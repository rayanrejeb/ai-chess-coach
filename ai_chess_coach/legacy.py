# Adapted from Chess Analyzer by yusufelgen07 (MIT License).
# See THIRD_PARTY_NOTICES.md.
# Temporary: the original ChessAnalyzer (engine + Gemini), kept running unchanged
# until engine/ and llm/ replace it.
import chess
import chess.engine
import chess.pgn
import chess.polyglot
import chess.svg
import chess.syzygy
import json
import logging
import math
import os
import re
import threading
import time
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import List, Optional, Tuple
import google.generativeai as genai
import google.api_core.exceptions

from ai_chess_coach.config import AnalysisConfig, Constants
from ai_chess_coach.engine.positional import PositionalAnalyzer
from ai_chess_coach.models import Classification, GeminiCommentary, LineAnalysis, PlyAnalysis, PositionalFeatures
from ai_chess_coach.tactics.analyzer import TacticalAnalyzer

log = logging.getLogger(__name__)


class ChessAnalyzer:
    """Handles all chess logic, including engine analysis and move classification."""
    def __init__(self, stockfish_path, gemini_key, gemini_model, opening_book_path=None, syzygy_path=None):
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self._engine_lock = threading.Lock()
        self._task_executor = ThreadPoolExecutor(max_workers=3)
        self._render_executor = ThreadPoolExecutor(max_workers=1)
        self._nav_generation = 0

        self.config = AnalysisConfig()
        self.positional_analyzer = PositionalAnalyzer(self.config)
        self.tactical_analyzer = TacticalAnalyzer()

        # --- CHANGE 1: Split the model into two instances ---
        self.commentary_model = None # For generating JSON commentary
        self.chat_model = None       # For interactive chat with tools

        if gemini_key and gemini_model:
            try:
                genai.configure(api_key=gemini_key)
                
                # Model for generating pure JSON (NO tools)
                self.commentary_model = genai.GenerativeModel(gemini_model)
                
                # Model for interactive chat (WITH tools)
                self.chat_model = genai.GenerativeModel(
                    gemini_model,
                    tools=[self.analyze_fen_for_chatbot]
                )
            except Exception as e:
                log.warning(f"Failed to configure Gemini: {e}. AI models disabled.")
        # --- END OF CHANGE 1 ---

        self.opening_book = None
        if opening_book_path and os.path.exists(opening_book_path):
            try:
                self.opening_book = chess.polyglot.open_reader(opening_book_path)
            except Exception as e:
                log.warning(f"Could not load opening book: {e}.")
        self.tablebases = None
        if syzygy_path and os.path.exists(syzygy_path):
            try:
                self.tablebases = chess.syzygy.open_tablebase(syzygy_path)
                log.info("Syzygy tablebases loaded successfully.")
            except Exception as e:
                log.warning(f"Could not load Syzygy tablebases: {e}.")

        self._san_cache: OrderedDict = OrderedDict()
        self._max_san_cache = 256

    def _probe_syzygy(self, board):
        if self.tablebases and len(board.piece_map()) <= self.tablebases.max_pieces:
            try:
                return self.tablebases.probe_wdl(board)
            except (IndexError, KeyError):
                return None
        return None

    def _get_analysis(self, board: chess.Board, depth: int, multipv: int, time_limit: float):
        with self._engine_lock:
            try:
                return self.engine.analyse(board, chess.engine.Limit(depth=depth, time=time_limit), multipv=multipv)
            except chess.engine.EngineTerminatedError:
                log.error("Engine terminated unexpectedly")
                return []
            except Exception as e:
                log.error(f"Engine analysis failed: {e}")
                return []

    def _cp_to_expected_points(self, cp):
        if cp is None:
            return 0.5
        cp = max(-self.config.MATE_SCORE, min(self.config.MATE_SCORE, cp))
        return 0.5 + 0.5 * (2 / (1 + math.exp(-0.004 * cp)) - 1)

    def _get_san_line(self, move_list: List[chess.Move], board: chess.Board) -> List[str]:
        if not move_list:
            return []
        key = (board.fen(), tuple(m.uci() for m in move_list))
        if key in self._san_cache:
            self._san_cache.move_to_end(key)
            return self._san_cache[key]
        san_moves = []
        temp_board = board.copy()
        for move in move_list:
            try:
                if move in temp_board.legal_moves:
                    san_moves.append(temp_board.san(move))
                    temp_board.push(move)
                else:
                    break
            except (chess.InvalidMoveError, chess.IllegalMoveError, AttributeError):
                break
        self._san_cache[key] = san_moves
        if len(self._san_cache) > self._max_san_cache:
            self._san_cache.popitem(last=False)
        return san_moves

    def classify_move(self, board: chess.Board, move: chess.Move, analysis_before_list: List[dict], analysis_after_dict: dict, opponent_move_classification: Optional[Classification] = None):
        try:
            if self.opening_book and self.opening_book.find(board, default=None):
                return Constants.BOOK_MOVE, 0.0, 0
        except Exception:
            pass
        player = board.turn
        if not analysis_before_list: return Constants.BLUNDER, 1.0, self.config.MATE_SCORE
        pv_before = analysis_before_list[0].get('pv') or []
        if not pv_before: return Constants.BLUNDER, 1.0, self.config.MATE_SCORE
        best_engine_move = pv_before[0]
        eval_before_top = analysis_before_list[0]['score'].pov(player)
        eval_after_player_move = analysis_after_dict['score'].pov(player)
        cp_before = eval_before_top.score(mate_score=self.config.MATE_SCORE)
        cp_after = eval_after_player_move.score(mate_score=self.config.MATE_SCORE)
        if self.tablebases and len(board.piece_map()) <= self.tablebases.max_pieces:
            wdl_before = self._probe_syzygy(board)
            temp_board = board.copy()
            temp_board.push(move)
            wdl_after = self._probe_syzygy(temp_board)
            if wdl_before is not None and wdl_after is not None:
                if wdl_before > 0 and wdl_after <= 0: return Constants.BLUNDER, 0.5, 500
                if wdl_before == 0 and wdl_after < 0: return Constants.MISTAKE, 0.2, 200
        if cp_before is None or cp_after is None: return Constants.BLUNDER, 1.0, self.config.MATE_SCORE
        points_loss = self._cp_to_expected_points(cp_before) - self._cp_to_expected_points(cp_after)
        cp_loss = cp_before - cp_after
        if eval_after_player_move.is_mate() and eval_after_player_move.mate() and eval_after_player_move.mate() > 0:
            if not (eval_before_top.is_mate() and eval_before_top.mate() and eval_before_top.mate() > 0):
                return Constants.BRILLIANT, points_loss, cp_loss
        if move == best_engine_move:
            points_before, points_after = self._cp_to_expected_points(cp_before), self._cp_to_expected_points(cp_after)
            is_great = (points_before < 0.4 and points_after >= 0.6) or (points_before < 0.5 and points_after >= 0.7)
            if not is_great and len(analysis_before_list) > 1 and points_before < 0.8:
                gaps = []
                for item in analysis_before_list[1:3]:
                    cp_second = item['score'].pov(player).score(mate_score=self.config.MATE_SCORE)
                    if cp_second is not None: gaps.append(points_before - self._cp_to_expected_points(cp_second))
                if gaps and max(gaps) > 0.25: is_great = True
            is_brilliant_sacrifice = False
            moved_piece = board.piece_at(move.from_square)
            if moved_piece and moved_piece.piece_type > chess.PAWN and not board.is_capture(move) and board.is_attacked_by(not player, move.to_square):
                if 0.1 < self._cp_to_expected_points(cp_before) < 0.9 and points_loss < 0.05:
                    is_brilliant_sacrifice = True
            if is_great and is_brilliant_sacrifice: return Constants.BRILLIANT, points_loss, cp_loss
            elif is_great: return Constants.GREAT_MOVE, points_loss, cp_loss
            else: return Constants.BEST, points_loss, cp_loss
        if eval_before_top.is_mate() and eval_before_top.mate() and eval_before_top.mate() > 0 and (not eval_after_player_move.is_mate() or (eval_after_player_move.mate() and eval_after_player_move.mate() <= 0)):
            return Constants.MISS, points_loss, cp_loss
        if eval_after_player_move.is_mate() and eval_after_player_move.mate() and eval_after_player_move.mate() < 0 and cp_before > -800:
            return Constants.BLUNDER, points_loss, cp_loss
        if opponent_move_classification in [Constants.BLUNDER, Constants.MISTAKE] and points_loss > 0.05:
            return Constants.MISS, points_loss, cp_loss
        if points_loss < 0.03: return Constants.EXCELLENT, points_loss, cp_loss
        if points_loss < 0.075: return Constants.GOOD, points_loss, cp_loss
        if points_loss < 0.15: return Constants.INACCURACY, points_loss, cp_loss
        if points_loss < 0.30: return Constants.MISTAKE, points_loss, cp_loss
        return Constants.BLUNDER, points_loss, cp_loss


    def _get_gemini_prompt(self, language: str, username: str, player_color: str, game_summary: dict, chunk_meta: Optional[dict] = None) -> str:
        summary_text = ""
        if game_summary.get('white_pattern'):
            summary_text += f"- White's recurring pattern: {game_summary['white_pattern']}\n"
        if game_summary.get('black_pattern'):
            summary_text += f"- Black's recurring pattern: {game_summary['black_pattern']}\n"
        chunk_info = f"\nCHUNK INFO: id={chunk_meta.get('id')}, total={chunk_meta.get('total')}, ply_range={chunk_meta.get('ply_range')}" if chunk_meta else ""
        
        return f'''You are an expert chess coach providing analysis for '{username}' ({player_color}) in {language}.
Your task is to provide detailed, insightful, and human-like commentary for each move in a chess game.
Your response must be a single, valid JSON object. Do not include any text or markdown formatting outside of the JSON.

**Your Coaching Philosophy:**
- **Be thorough and never skip details.** Your goal is to help the user understand every aspect of the move.
- **Explain the 'Why':** Don't just state a fact. Explain *why* a move is good or bad, *why* a tactic works, and *why* a certain position is advantageous.
- **Focus on Potential:** Discuss not only the immediate tactical results but also any *potential* threats or opportunities the move creates for the future. This includes forks, pins, discovered attacks that are not yet realized but are now possible.
- **Human-like Commentary:** Use natural, encouraging, and educational language. Avoid robotic lists of data. Weave the tactical motifs and positional analysis into a clear, narrative explanation.
- **Analyze Alternatives:** For every move, discuss the `missed_opportunities` (what the player could have played) and the `opponent_best_replies`. Explain what makes these moves strong and what the follow-up might look like. This is crucial for learning.
- **Connect to Data:** Ground your analysis in the provided tactical and positional data. For example, if you see a "Fork" motif, explain how it works in the position. If you see "bad bishop", explain why the bishop is considered bad.
- **Profitable Trades:** If you see a "Profitable Trade" tactic, it's critical to explain that the player's move allowed the opponent to initiate a favorable sequence of captures.

**JSON Response Format:**
Your entire output must be a single JSON object.

--- EXAMPLE JSON RESPONSE FORMAT ---
{{
  "game_commentary": [
    {{
      "ply": 1,
      "move_san": "e4",
      "move_commentary": "An excellent first move! By placing a pawn in the center, you immediately fight for control of key squares and open lines for your queen and bishop. This is a very popular and strong way to begin the game.",
      "missed_opportunities": [],
      "best_replies": ["e5", "c5"]
    }},
    {{
      "ply": 2,
      "move_san": "d5?!",
      "move_commentary": "This is an ambitious but risky move, known as the Scandinavian Defense. While it challenges White's center immediately, it brings your queen out very early, which can make it a target for your opponent's developing pieces.",
      "missed_opportunities": [
        "c5 was a great alternative. The Sicilian Defense is a top-tier response that fights for the center without exposing your queen so early."
      ],
      "best_replies": ["exd5"]
    }}
  ],
  "game_summary": {{ "white_takeaway": "...", "black_takeaway": "...", "white_pattern": "...", "black_pattern": "..." }}
}}
---

**Pre-computed Analysis for Your Reference:**
{summary_text}
{chunk_info}

Now, generate the JSON response for the provided game data. Be as detailed and helpful as possible.
'''



    def _send_gemini_request(self, prompt: str) -> Tuple[Optional[dict], Optional[dict]]:
        if not self.commentary_model: return None, None
        
        # --- FIX 1: Enforce Strict JSON Schema ---
        # This guarantees Gemini Flash won't rename keys (like 'Ply' instead of 'ply')
        # or skip the structure entirely, preventing silent failures.
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    "game_commentary": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "ply": {"type": "integer"},
                                "move_san": {"type": "string"},
                                "move_commentary": {"type": "string"},
                                "missed_opportunities": {"type": "array", "items": {"type": "string"}},
                                "best_replies": {"type": "array", "items": {"type": "string"}}
                            },
                            "required":["ply", "move_san", "move_commentary"]
                        }
                    },
                    "game_summary": {
                        "type": "object",
                        "properties": {
                            "white_takeaway": {"type": "string"},
                            "black_takeaway": {"type": "string"},
                            "white_pattern": {"type": "string"},
                            "black_pattern": {"type": "string"}
                        }
                    }
                },
                "required": ["game_commentary"]
            }
        )
        
        safety_settings = {
            'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
            'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
            'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
            'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
        }

        for attempt in range(5):
            try:
                response = self.commentary_model.generate_content(
                    prompt, 
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                
                if not response.parts:
                    log.warning(f"Gemini returned no content. Finish Reason: {response.candidates[0].finish_reason if response.candidates else 'N/A'}")
                    if attempt < 4: time.sleep(2)
                    continue

                raw_text = response.text or ""
                
                # Clean up the response
                match = re.search(r'```json\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
                cleaned_text = match.group(1) if match else raw_text.strip()
                if cleaned_text.startswith("```"):
                    cleaned_text = cleaned_text.strip("`").strip("json").strip()
                
                parsed_json = json.loads(cleaned_text)
                game_summary = parsed_json.get("game_summary")
                commentaries = {}
                
                for item in parsed_json.get("game_commentary",[]):
                    # Schema enforcement ensures 'ply' is always correct now
                    if 'ply' in item:
                        commentaries[str(item['ply'])] = GeminiCommentary.from_dict(item)
                
                if not commentaries:
                    log.warning("JSON parsed successfully but 'game_commentary' was empty.")
                    
                return commentaries, game_summary

            except google.api_core.exceptions.ResourceExhausted as e:
                # If the quota is 0 (Daily limit hit), don't waste 2 minutes retrying
                if 'daily' in str(e).lower() or 'quota' in str(e).lower():
                    log.error(f"Fatal Gemini Quota Limit hit: {e}")
                    return None, None
                
                wait_time = 5 * (2 ** attempt)
                log.warning(f"Gemini rate limit hit. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                
            except json.JSONDecodeError as e:
                log.error(f"JSON decoding failed (Attempt {attempt + 1}): {e}")
                if attempt < 4: time.sleep(2)
                
            except Exception as e:
                log.error(f"Gemini analysis failed (Attempt {attempt + 1}): {e}")
                if attempt < 4: time.sleep(2)
                
        return None, None

    def _create_ai_payload_for_ply(self, ply: PlyAnalysis, reported_features: set) -> dict:
        def summarize_positional(features: PositionalFeatures) -> list:
            summary = []
            if features.king_safety.get("white", {}).get("status") != "safe": summary.append(f"White king is {features.king_safety['white']['status']}")
            if features.king_safety.get("black", {}).get("status") != "safe": summary.append(f"Black king is {features.king_safety['black']['status']}")
            for p_type, items in features.pawn_structure.items():
                if items: summary.append(f"{p_type.replace('_', ' ')}: {', '.join(items)}")
            for r_color, r_items in features.rook_activity.items(): summary.extend(r_items)
            summary.extend(features.advantages)
            return summary
        moves_dict = {"player_move": ply.move_san}
        missed_opportunities = [self._get_simplified_line_analysis(line, ply.fen_before) for line in ply.best_engine_lines[:2] if line.san and line.san.split(' ')[0] != ply.move_san]
        for i, opp in enumerate(missed_opportunities):
            if opp.get("san"): moves_dict[f"engine_line_{i+1}"] = opp["san"]
        opponent_replies = [self._get_simplified_line_analysis(line, ply.fen_before) for line in ply.best_follow_up_lines[:2]]
        for i, rep in enumerate(opponent_replies):
            if rep.get("san"): moves_dict[f"follow_up_{i+1}"] = rep["san"]
        positional_summary_full = summarize_positional(ply.positional_analysis_after)
        newly_detected_positional_summary = []
        for item in positional_summary_full:
            if item not in reported_features:
                newly_detected_positional_summary.append(item)
                reported_features.add(item)
        player_tactics = [{"tactic": m['tactic'], "details": m.get('details', '')} for m in ply.player_line.tactical_motifs if m.get('tactic') != "None"][:3]
        return {
            "ply": ply.ply, "player": ply.player, "move_san": ply.move_san, "classification": ply.classification,
            "game_phase": ply.game_phase, "fen_before": ply.fen_before,
            "player_move_analysis": {"tactics": player_tactics, "positional_summary_after": newly_detected_positional_summary},
            "missed_opportunities": missed_opportunities, "opponent_best_replies": opponent_replies,
            "moves_for_template": moves_dict
        }

    def _get_simplified_line_analysis(self, line: LineAnalysis, board_fen: str, max_moves=3) -> dict:
        if not line or not line.san: return {}
        result, moves_analysis, temp_board = {"san": " ".join(line.san.split(' ')[:max_moves])}, [], chess.Board(board_fen)
        for i, uci_move in enumerate(line.uci[:max_moves]):
            move_obj = chess.Move.from_uci(uci_move)
            if move_obj not in temp_board.legal_moves: break
            board_before_move = temp_board.copy()
            temp_board.push(move_obj)
            tactical = self.tactical_analyzer.find_tactical_motifs(board_before_move, temp_board, move_obj)
            cleaned_tactical = [t for t in tactical if t.get('tactic') != "None"]
            positional = self.positional_analyzer.analyze_position(temp_board)
            cleaned_positional = {}
            pos_dict = asdict(positional)
            for k, v in pos_dict.items():
                if isinstance(v, (list, dict)) and not v: continue
                if isinstance(v, float) and v == 0.0: continue
                if k == "king_safety" and isinstance(v, dict) and v.get("status") == "safe": continue
                cleaned_positional[k] = v
            move_analysis_entry = {"move_san": board_before_move.san(move_obj)}
            if cleaned_tactical: move_analysis_entry["tactical_motifs"] = cleaned_tactical
            if cleaned_positional: move_analysis_entry["positional_analysis"] = cleaned_positional
            moves_analysis.append(move_analysis_entry)
        if moves_analysis: result["moves_analysis"] = moves_analysis
        return result

    def get_gemini_analysis(self, full_game_analysis: List[PlyAnalysis], language: str, username: str, player_color: str, game: chess.pgn.Game, status_callback=None) -> Tuple[dict, Optional[dict]]:
        if not self.commentary_model: return {}, None
        reported_ai_features = set()
        sanitized_analysis =[self._create_ai_payload_for_ply(ply, reported_ai_features) for ply in full_game_analysis]
        
        # --- FIX 2: Lower the token limit per request ---
        # Gemini has an 8192 max output limit. Asking for 60 moves at once causes it to 
        # truncate or return empty arrays. Breaking it into ~3 chunks solves this.
        TOKEN_LIMIT_PER_REQUEST = 15000 
        
        game_summary_data = self._summarize_recurring_patterns(full_game_analysis, username, game)
        base_prompt_str = self._get_gemini_prompt(language, username, player_color, game_summary_data, None)
        
        analysis_chunks = self._chunk_analysis_by_tokens(sanitized_analysis, base_prompt_str, TOKEN_LIMIT_PER_REQUEST)
        all_commentaries = {}
        final_game_summary = None
        
        for i, chunk in enumerate(analysis_chunks):
            is_final_chunk = (i == len(analysis_chunks) - 1)
            chunk_data_str = json.dumps(chunk)
            ply_range = (chunk[0]["ply"], chunk[-1]["ply"]) if chunk else (0, 0)
            chunk_meta = {"id": i + 1, "total": len(analysis_chunks), "ply_range": ply_range}
            prompt_for_chunk = self._get_gemini_prompt(language, username, player_color, game_summary_data, chunk_meta)
            
            if status_callback: status_callback(f"AI Part {i+1}/{len(analysis_chunks)}...")
            chunk_commentaries, chunk_summary = self._send_gemini_request(prompt_for_chunk + "\n--- GAME DATA TO ANALYZE ---\n" + chunk_data_str)
            
            if chunk_commentaries: 
                all_commentaries.update(chunk_commentaries)
            if is_final_chunk and chunk_summary: 
                final_game_summary = chunk_summary
                
        return all_commentaries, final_game_summary

    def _chunk_analysis_by_tokens(self, sanitized_analysis: list, base_prompt: str, token_limit: int) -> list:
        analysis_chunks = []
        current_chunk = []
        base_prompt_tokens = len(base_prompt) / 4
        current_chunk_tokens = 0
        
        for ply_analysis in sanitized_analysis:
            # FIX: Split into two lines so ply_str is created before taking its length
            ply_str = json.dumps(ply_analysis)
            ply_tokens = len(ply_str) / 4
            
            if current_chunk and (current_chunk_tokens + ply_tokens + base_prompt_tokens > token_limit):
                analysis_chunks.append(current_chunk)
                current_chunk = []
                current_chunk_tokens = 0
                
            current_chunk.append(ply_analysis)
            current_chunk_tokens += ply_tokens
            
        if current_chunk: 
            analysis_chunks.append(current_chunk)
            
        return analysis_chunks

    def _summarize_recurring_patterns(self, full_game_analysis: List[PlyAnalysis], username: str, game: chess.pgn.Game) -> dict:
        player_color_name = ""
        if username.lower() == game.headers.get("White", "").lower(): player_color_name = "white"
        elif username.lower() == game.headers.get("Black", "").lower(): player_color_name = "black"
        if not player_color_name: return {}
        mistake_causes = defaultdict(int)
        player_mistakes = [p for p in full_game_analysis if p.player.lower() == player_color_name and p.classification in [Constants.MISTAKE, Constants.BLUNDER]]
        for mistake in player_mistakes:
            if mistake.best_follow_up and mistake.best_follow_up.tactical_motifs:
                for motif in mistake.best_follow_up.tactical_motifs:
                    if motif['tactic'] != "None": mistake_causes[f"Allowed a {motif['tactic']} tactic"] += 1
            if mistake.best_engine_line and mistake.best_engine_line.tactical_motifs:
                for motif in mistake.best_engine_line.tactical_motifs:
                    if motif['tactic'] != "None": mistake_causes[f"Missed a {motif['tactic']} opportunity"] += 1
        if not mistake_causes: return {f"{player_color_name}_pattern": "No consistent mistakes found. Solid game!"}
        most_common_mistake = max(mistake_causes, key=mistake_causes.get)
        return {f"{player_color_name}_pattern": most_common_mistake}
    
    def _safe_see(self, board: chess.Board, move: chess.Move) -> int:
        try:
            return int(self.tactical_analyzer.see(board, move, with_sequence=False))
        except Exception:
            return 0

    def _analyze_opponent_choice(self, board: chess.Board, analysis_list: List[dict], player_move: chess.Move) -> Optional[dict]:
        opponent_captures = [m for m in board.legal_moves if board.is_capture(m)]
        if len(opponent_captures) < 2: return None
        profitable_choices = []
        for move in opponent_captures:
            see_value = self._safe_see(board, move)
            if see_value > 0: profitable_choices.append({'move': move, 'see': see_value})
        if len(profitable_choices) < 2: return None
        profitable_choices.sort(key=lambda x: x['see'], reverse=True)
        choice_a_move, choice_b_move, opponent_color = profitable_choices[0]['move'], profitable_choices[1]['move'], board.turn
        try:
            board_a = board.copy(); board_a.push(choice_a_move)
            analysis_a = self._get_analysis(board_a, 14, 1, 0.7)
            score_a = analysis_a[0]['score'].pov(opponent_color).score(mate_score=self.config.MATE_SCORE) if analysis_a else None
            board_b = board.copy(); board_b.push(choice_b_move)
            analysis_b = self._get_analysis(board_b, 14, 1, 0.7)
            score_b = analysis_b[0]['score'].pov(opponent_color).score(mate_score=self.config.MATE_SCORE) if analysis_b else None
        except Exception: return None
        if score_a is None or score_b is None or score_a < 50 or score_b < 50: return None
        san_a, san_b = board.san(choice_a_move), board.san(choice_b_move)
        details = f"Your move gives the opponent a choice between two good captures: {san_a} (eval for them: {score_a/100.0:+.2f}) and {san_b} (eval for them: {score_b/100.0:+.2f})."
        return {"tactic": "Opponent Choice", "details": details, "value": abs(score_a - score_b)}

    def _detect_zwischenzug(self, board: chess.Board, move: chess.Move, analysis_before: List[dict]) -> Optional[dict]:
        if not analysis_before or not board.is_capture(move): return None
        best_engine_move = analysis_before[0].get('pv', [None])[0]
        if not best_engine_move or best_engine_move == move: return None
        current_player, engine_best_score = board.turn, analysis_before[0]['score'].pov(board.turn).score(mate_score=self.config.MATE_SCORE)
        player_move_score = None
        try:
            temp_board = board.copy()
            temp_board.push(move)
            player_analysis = self._get_analysis(temp_board, 14, 1, 0.5)
            if player_analysis:
                opponent_pov_score = player_analysis[0]['score'].pov(temp_board.turn).score(mate_score=self.config.MATE_SCORE)
                if opponent_pov_score is not None: player_move_score = -opponent_pov_score
        except Exception: return None
        if engine_best_score is None or player_move_score is None: return None
        cp_diff = engine_best_score - player_move_score
        if cp_diff > 120:
            engine_move_board_after = board.copy()
            engine_move_board_after.push(best_engine_move)
            motifs = self.tactical_analyzer.find_tactical_motifs(board, engine_move_board_after, best_engine_move)
            is_tactical = engine_move_board_after.is_check() or any(m['tactic'] not in ["None", "Hanging Piece"] for m in motifs)
            if is_tactical:
                san_engine, san_player = board.san(best_engine_move), board.san(move)
                details = f"A powerful intermediate move was missed. Instead of the simple recapture {san_player} (eval: {player_move_score/100.0:+.2f}), the move {san_engine} (eval: {engine_best_score/100.0:+.2f}) creates a significant tactical advantage."
                return {"tactic": "Missed Zwischenzug", "details": details, "value": cp_diff}
        return None

    def quit(self):
        try:
            if self.engine: self.engine.quit()
            if self.tablebases: self.tablebases.close()
            self._task_executor.shutdown(wait=False, cancel_futures=True)
            self._render_executor.shutdown(wait=False, cancel_futures=True)
        except Exception: pass

    def get_chat_response(self, question: str, game_context: dict, chat_history: str) -> str:
        if not self.chat_model:
            return "The AI chat model is not configured. Please check your Gemini API key in the settings."

        # --- FIX: Smarter System Prompt ---
        system_prompt = f"""You are an expert Grandmaster chess coach.
A user is asking a question about a specific moment in their game.

Your Task:
1. Carefully analyze the user's question.
2. If the user asks a "what if" question about a different move (e.g., "what if I played Ng5 instead?" or "why not Nxe4?"), you MUST use the `analyze_fen_for_chatbot` tool.
3. Pass the `fen_before` string from the game context EXACTLY as it is, and pass the user's hypothetical move (e.g. 'Ng5') to the tool. DO NOT try to calculate the FEN yourself.
4. If the user asks a general question about the current position, answer it using the provided context.
5. Answer in a friendly, conversational, and educational tone. Do not output raw JSON to the user.
"""

        history = [
            {'role': 'user', 'parts': [system_prompt]},
            {'role': 'model', 'parts': ["Understood. I will use the tool by passing the current FEN and the hypothetical move to get accurate engine data."]},
            {'role': 'user', 'parts': [
                "Here is the game context:",
                json.dumps(game_context, indent=2),
                "\n--- CHAT HISTORY ---",
                chat_history,
                "--- END CHAT HISTORY ---"
            ]},
            {'role': 'model', 'parts': ["I have reviewed the context. What is the user's current question?"]}
        ]

        try:
            # Re-initialize the chat to clear any broken tool states
            chat = self.chat_model.start_chat(history=history, enable_automatic_function_calling=True)
            response = chat.send_message(question)
            return response.text
        except Exception as e:
            log.error(f"Gemini chat failed: {e}")
            return f"An error occurred while analyzing that move: {e}"

    def find_key_moments(self, full_game_analysis: List[PlyAnalysis], num_moments: int = 5) -> List[PlyAnalysis]:
        if not full_game_analysis: return []
        scored_plies = []
        for ply in full_game_analysis:
            impact_score = ply.points_loss * 100
            if ply.classification == Constants.BRILLIANT: impact_score += 50
            elif ply.classification == Constants.GREAT_MOVE: impact_score += 20
            scored_plies.append((impact_score, ply))
        scored_plies.sort(key=lambda x: x[0], reverse=True)
        key_moments = [ply for score, ply in scored_plies[:num_moments]]
        key_moments.sort(key=lambda p: p.ply)
        return key_moments

    def analyze_fen_for_chatbot(self, fen: str, hypothetical_move_san: str = "") -> dict:
        """Analyzes a chess position, optionally applying a 'what if' move first.

        Use this tool when the user asks a "what if" question about a chess move 
        (e.g., "what if I played Ng5?"). 

        Args:
            fen: The EXACT fen_before string provided in the game context. Do not alter it.
            hypothetical_move_san: The Standard Algebraic Notation of the move the user is asking about (e.g., 'Ng5', 'exd5', 'O-O'). Leave blank if they didn't suggest a move.
        """
        try:
            board = chess.Board(fen)
            
            # 1. If the AI passed a hypothetical move, let Python apply it mathematically
            if hypothetical_move_san:
                try:
                    move = board.parse_san(hypothetical_move_san)
                    board.push(move)
                except ValueError:
                    legal_moves = ", ".join([board.san(m) for m in board.legal_moves][:15])
                    return {"error": f"The move '{hypothetical_move_san}' is illegal or impossible in this position. Tell the user this move is illegal. Examples of legal moves are: {legal_moves}..."}

            new_fen = board.fen()
            
            # 2. Run Stockfish on the correct resulting board
            initial_positional_analysis = self.positional_analyzer.analyze_position(board)
            main_analysis_list = self._get_analysis(board, self.config.FULL_DEPTH, self.config.FULL_MULTIPV, self.config.FULL_TIME)
            
            if not main_analysis_list or not main_analysis_list[0].get('pv'):
                return {"error": "Could not get a best line from the engine."}
                
            best_line_pv = main_analysis_list[0].get('pv', [])
            # POV matches the player whose turn it is now
            best_line_score_cp = main_analysis_list[0]['score'].pov(board.turn).score(mate_score=self.config.MATE_SCORE)
            
            best_line_obj = LineAnalysis(
                san=" ".join(self._get_san_line(best_line_pv, board.copy())), 
                uci=[m.uci() for m in best_line_pv], 
                score_cp=best_line_score_cp or 0
            )
            
            detailed_best_line = self._get_simplified_line_analysis(best_line_obj, new_fen, max_moves=3)
            
            return {
                "success": "Engine analysis complete.",
                "resulting_fen": new_fen,
                "current_engine_evaluation_centipawns": best_line_score_cp,
                "engine_best_line_continuation": detailed_best_line,
                "positional_analysis": asdict(initial_positional_analysis)
            }
            
        except Exception as e:
            return {"error": f"Python backend failed to process the request: {str(e)}"}

