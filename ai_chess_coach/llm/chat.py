"""Interactive coach chat with Stockfish exposed as a tool, plus the Coach facade used by the GUI."""

import json
import logging
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import chess
import chess.engine
from google.genai import types

from ai_chess_coach.llm.client import GeminiClient, LLMError
from ai_chess_coach.llm.commentary import CommentaryContext, DescribeLine, generate_game_commentary
from ai_chess_coach.llm.grounding import GroundingReport, References, check, normalize_san, references_for_ply

log = logging.getLogger(__name__)

MATE_SCORE = 10_000
TOOL_LINE_PLIES = 6
ILLEGAL_EXAMPLES = 12

# analyse(board) -> python-chess InfoDict with "score" and "pv", or None if the engine failed
Analyse = Callable[[chess.Board], Optional[dict]]

SYSTEM_PROMPT = """You are a friendly chess coach answering a student's questions about one moment of their game.

Rules:
- Write every chess move in English Standard Algebraic Notation (SAN): K Q R B N for pieces, O-O and O-O-O for castling, e.g. Nf3, Bxb5+. Never translate piece letters, even if you answer in another language.
- Every move you name must come from the game context, from the game itself, or from a call to `evaluate_position`.
- When you refer to a move played earlier in the game, write it with its move number, e.g. 7.Qb3 or 13...Rxd7.
- For any "what if" question about a move not in the context, call `evaluate_position` with the context's `fen_before` copied exactly and the move in SAN. Never compute a FEN yourself.
- If the tool says a move is illegal, tell the student.
- Answer in plain, encouraging language. No raw JSON.
"""


class EngineTool:
    """Wraps the engine as a function the model can call, and remembers what it evaluated."""

    def __init__(self, analyse: Analyse, describe_line: Optional[DescribeLine] = None):
        self._analyse = analyse
        self._describe_line = describe_line
        self.evaluated = References()

    def as_function(self) -> Callable[..., dict]:
        tool = self

        def evaluate_position(fen: str, move_san: str = "") -> dict:
            """Evaluate a chess position with Stockfish, optionally after playing one move first.

            Args:
              fen: The position in FEN. Copy fen_before from the game context exactly.
              move_san: Optional move to play first, in English SAN (for example "Nf3" or "exd5"). Empty to evaluate the position as it is.
            """
            return tool.evaluate(fen, move_san)

        return evaluate_position

    def evaluate(self, fen: str, move_san: str = "") -> dict:
        try:
            board = chess.Board(fen)
        except ValueError:
            return {"error": "Invalid FEN. Use fen_before from the game context exactly."}
        start_fen = board.fen()
        if move_san:
            try:
                move = board.parse_san(move_san)
            except ValueError:
                self.evaluated.rejected.add(normalize_san(move_san.strip()))
                examples = ", ".join(board.san(m) for m in list(board.legal_moves)[:ILLEGAL_EXAMPLES])
                return {"error": f"{move_san} is not a legal move here. Legal moves include: {examples}."}
            self.evaluated.add_line(start_fen, [move.uci()])
            board.push(move)

        if board.is_game_over():
            return {"resulting_fen": board.fen(), "game_over": board.result(claim_draw=True)}
        info = self._analyse(board)
        if not info or not info.get("pv"):
            return {"error": "The engine could not analyse this position."}

        pv = info["pv"][:TOOL_LINE_PLIES]
        self.evaluated.add_line(board.fen(), [m.uci() for m in pv])
        score: chess.engine.PovScore = info["score"]
        result = {
            "resulting_fen": board.fen(),
            "side_to_move": "White" if board.turn == chess.WHITE else "Black",
            "eval_cp_white": score.white().score(mate_score=MATE_SCORE),
            "eval_cp_side_to_move": score.pov(board.turn).score(mate_score=MATE_SCORE),
            "mate_in_for_side_to_move": score.pov(board.turn).mate(),
            "best_line_san": board.variation_san(pv),
        }
        if self._describe_line:
            result["best_line_details"] = self._describe_line(board.fen(), [m.uci() for m in pv[:3]])
        return result


@dataclass
class CoachAnswer:
    text: str
    first_report: GroundingReport
    final_report: GroundingReport
    retried: bool


def _correction(report: GroundingReport) -> str:
    parts = []
    if report.illegal:
        parts.append(f"These moves are not legal in the positions discussed: {', '.join(report.illegal)}.")
    if report.unverified:
        parts.append(f"These moves were not checked by the engine: {', '.join(report.unverified)}.")
    parts.append("Call evaluate_position for any of them you want to keep, or remove them. Then write the full answer again.")
    return " ".join(parts)


class CoachChat:
    def __init__(self, client: GeminiClient, analyse: Analyse, describe_line: Optional[DescribeLine] = None):
        self.client = client
        self.analyse = analyse
        self.describe_line = describe_line

    def answer(self, question: str, game_context: dict, history_text: str, refs: References) -> CoachAnswer:
        tool = EngineTool(self.analyse, self.describe_line)
        message = (
            "Game context:\n" + json.dumps(game_context, ensure_ascii=False, indent=1)
            + ("\n\nConversation so far:\n" + history_text if history_text else "")
            + "\n\nStudent's question: " + question
        )
        text = self.client.chat(SYSTEM_PROMPT, [], message, tools=[tool.as_function()])
        first = check(text, refs, tool.evaluated)
        final, retried = first, False
        if not first.ok:
            retried = True
            history = [
                types.Content(role="user", parts=[types.Part(text=message)]),
                types.Content(role="model", parts=[types.Part(text=text)]),
            ]
            text = self.client.chat(SYSTEM_PROMPT, history, _correction(first), tools=[tool.as_function()])
            final = check(text, refs, tool.evaluated)
        log.info(
            "grounding chat first_illegal=%s first_unverified=%s final_illegal=%s final_unverified=%s retried=%s",
            first.illegal, first.unverified, final.illegal, final.unverified, retried,
        )
        if not final.ok:
            flagged = final.illegal + final.unverified
            text += f"\n\n[Not verified by the engine: {', '.join(flagged)}]"
        return CoachAnswer(text, first, final, retried)


class Coach:
    """What the GUI uses: game commentary and chat, with errors turned into messages."""

    def __init__(self, api_key: str, model: Optional[str], analyse: Analyse, describe_line: DescribeLine, *, client: Optional[GeminiClient] = None):
        self.client = client or GeminiClient(api_key, model)
        self.describe_line = describe_line
        self.chat = CoachChat(self.client, analyse, describe_line)

    def comment_game(self, plies, language: str, username: str, player_color: str, game_moves: Sequence[chess.Move] = (), on_status=None):
        ctx = CommentaryContext(language, username, player_color)
        return generate_game_commentary(self.client, plies, ctx, self.describe_line, game_moves, on_status)

    def answer(self, question: str, game_context: dict, history_text: str, ply, game_moves: Sequence[chess.Move] = ()) -> str:
        try:
            return self.chat.answer(question, game_context, history_text, references_for_ply(ply, game_moves)).text
        except LLMError as err:
            log.error("Coach chat failed (%s): %s", err.kind, err)
            if err.kind == "quota_exhausted":
                return "The Gemini daily quota is used up. Try again tomorrow or use another API key."
            if err.kind == "model_not_found":
                return f"The Gemini model '{self.client.model}' is not available. Change GEMINI_MODEL in .env."
            return f"The coach could not answer right now ({err.kind}). Please try again."
