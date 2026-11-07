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
import logging
import math
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import List, Optional

from ai_chess_coach.llm.chat import Coach
from ai_chess_coach.config import AnalysisConfig, Constants
from ai_chess_coach.engine.positional import PositionalAnalyzer
from ai_chess_coach.models import Classification, LineAnalysis, PlyAnalysis
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

        # LLM layer (ai_chess_coach.llm); None when no Gemini key is configured.
        self.coach = None
        if gemini_key:
            self.coach = Coach(
                gemini_key,
                gemini_model,
                analyse=lambda board: next(iter(self._get_analysis(board, self.config.FULL_DEPTH, 1, self.config.FULL_TIME)), None),
                describe_line=self._describe_line,
            )

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


    def _describe_line(self, fen: str, uci_moves: List[str]) -> dict:
        san = " ".join(self._get_san_line([chess.Move.from_uci(u) for u in uci_moves], chess.Board(fen)))
        return self._get_simplified_line_analysis(LineAnalysis(san=san, uci=list(uci_moves)), fen)

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

