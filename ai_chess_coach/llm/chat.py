"""Interactive coach chat with Stockfish exposed as a tool, plus the Coach facade used by the GUI."""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import chess
import chess.engine
from google.genai import types

from ai_chess_coach.llm.client import GeminiClient, LLMError
from ai_chess_coach.llm.commentary import CommentaryContext, DescribeLine, generate_game_commentary
from ai_chess_coach.llm.grounding import GroundingReport, References, _mentions, check, normalize_san, references_for_ply

log = logging.getLogger(__name__)

MATE_SCORE = 10_000
TOOL_LINE_PLIES = 6
ILLEGAL_EXAMPLES = 12
CHECK_PREFIX = "Checked with Stockfish: "

# analyse(board) -> python-chess InfoDict with "score" and "pv", or None if the engine failed
Analyse = Callable[[chess.Board], Optional[dict]]

SYSTEM_PROMPT = """You are a friendly chess coach answering a student's questions about one moment of their game.

Rules:
- Answer in {language}.
- Write every chess move in English Standard Algebraic Notation (SAN): K Q R B N for pieces, O-O and O-O-O for castling, e.g. Nf3, Bxb5+. Never translate piece letters, even if you answer in another language.
- Every move you name must come from the game context, from the game itself, or from a call to `evaluate_position`.
- When you refer to a move played earlier in the game, write it with its move number, e.g. 7.Qb3 or 13...Rxd7.
- For any "what if" question about a move not in the context, call `evaluate_position` with the context's `fen_before` copied exactly and the move in SAN. Never compute a FEN yourself.
- To check a move from earlier in the game, call `evaluate_position` with `move_number` instead of a FEN: "10." is the position before White's 10th move, "10..." the position before Black's 10th move. Put the move to test in `move_san`.
- If the tool says a move is illegal, tell the student.
- The context gives the app's rating of the move discussed (`move_info`) and of game moves named in the question
  (`rated_game_moves`), with the win chance and centipawns it lost. That rating is the verdict: state it once, plainly,
  and never contradict it. Do not call a move rated Best, Excellent or Good a mistake or a weakness, and do not praise
  a move rated Inaccuracy, Mistake or Blunder.
- If `engine_top_choice` is true for the move discussed, the engine had nothing better: explain why the move works,
  from the engine data. Do not invent a weakness or a better move.
- If `engine_top_choice` is false, the engine data shows a loss or a move it preferred: after the verdict, name at least
  one concrete downside of the move or the alternative the engine preferred, taken from `best_alternative_line`,
  `best_opponent_follow_up`, the centipawns lost, or an `evaluate_position` call. For a game move other than the
  current one, call `evaluate_position` with its `move_number` and no `move_san` to see what the engine preferred there.
- Write calmly and factually, like a coach reviewing the game with the student. No exclamation marks and no cheering
  ("Great move", "Nice play"). No raw JSON.
"""


def move_rating(ply) -> dict:
    """The app's verdict on a played move, for the chat context. Small negative losses are engine noise: shown as 0.
    engine_top_choice: rated Best, no centipawns lost, or the engine's first move. The prompt then asks why the move
    works instead of for a downside."""
    cp_lost = max(0, ply.cp_loss)
    played = ply.player_line.uci[:1]
    return {
        "app_rating": ply.classification,
        "win_chance_lost_pct": round(max(0.0, ply.points_loss) * 100, 1),
        "centipawns_lost": cp_lost,
        "engine_top_choice": ply.classification == "Best" or cp_lost == 0 or (bool(played) and ply.best_engine_line.uci[:1] == played),
    }


def rated_game_moves(question: str, plies: Sequence) -> list[dict]:
    """The app's rating of each game move named in the question: "13.f3" by number, or a bare "f3"
    when only one game move matches."""
    by_cite, by_san = {}, {}
    for ply in plies:
        board = chess.Board(ply.fen_before)
        by_cite[(board.fullmove_number, board.turn == chess.BLACK)] = ply
        by_san.setdefault(normalize_san(ply.move_san), []).append(ply)
    found = []
    for san, cite, _ in _mentions(question):
        if cite:
            ply = by_cite.get(cite)
        else:
            matches = by_san.get(san, [])
            ply = matches[0] if len(matches) == 1 else None
        if ply is not None and normalize_san(ply.move_san) == san and ply not in found:
            found.append(ply)
    rated = []
    for p in found:
        board = chess.Board(p.fen_before)
        rated.append({"move": f"{board.fullmove_number}{'.' if board.turn == chess.WHITE else '...'} {p.move_san}", **move_rating(p)})
    return rated


class EngineTool:
    """Wraps the engine as a function the model can call, and remembers what it evaluated."""

    def __init__(self, analyse: Analyse, describe_line: Optional[DescribeLine] = None,
                 game_moves: Sequence[chess.Move] = (), start_fen: str = chess.STARTING_FEN):
        self._analyse = analyse
        self._describe_line = describe_line
        self._game_moves = list(game_moves)
        self._start_fen = start_fen
        self.evaluated = References()
        self.checks: list[str] = []  # one line per successful call, shown under the answer

    def as_function(self) -> Callable[..., dict]:
        tool = self

        def evaluate_position(fen: str = "", move_san: str = "", move_number: str = "") -> dict:
            """Evaluate a chess position with Stockfish, optionally after playing one move first.

            Args:
              fen: The position in FEN. Copy fen_before from the game context exactly. Leave empty when using move_number.
              move_san: Optional move to play first, in English SAN (for example "Nf3" or "exd5"). Empty to evaluate the position as it is.
              move_number: Optional, to evaluate an earlier position of this game instead of fen: "10." is the position before White's 10th move, "10..." the position before Black's 10th move.
            """
            return tool.evaluate(fen, move_san, move_number)

        return evaluate_position

    def game_position(self, move_number: str) -> Optional[chess.Board]:
        """The game position before the move numbered "10." (White) or "10..." (Black), or None."""
        match = re.fullmatch(r"\s*(\d+)\s*(\.\.\.|\.|…)\s*", move_number)
        if not match:
            return None
        wanted = (int(match.group(1)), chess.WHITE if match.group(2) == "." else chess.BLACK)
        board = chess.Board(self._start_fen)
        for move in [*self._game_moves, None]:
            if (board.fullmove_number, board.turn) == wanted:
                return board
            if move is None or move not in board.legal_moves:
                return None
            board.push(move)

    def evaluate(self, fen: str = "", move_san: str = "", move_number: str = "") -> dict:
        if move_number:
            board = self.game_position(move_number)
            if board is None:
                return {"error": f"No position for move number {move_number!r} in this game. Use \"10.\" (before White's 10th move) or \"10...\" (before Black's)."}
        else:
            try:
                board = chess.Board(fen)
            except ValueError:
                return {"error": "Invalid FEN. Use fen_before from the game context exactly, or move_number for an earlier position."}
        start_fen = board.fen()
        number = f"{board.fullmove_number}{'.' if board.turn == chess.WHITE else '...'}"
        label = f"position before {number}"
        if move_san:
            try:
                move = board.parse_san(move_san)
            except ValueError:
                self.evaluated.rejected.add(normalize_san(move_san.strip()))
                examples = ", ".join(board.san(m) for m in list(board.legal_moves)[:ILLEGAL_EXAMPLES])
                return {"error": f"{move_san} is not a legal move here. Legal moves include: {examples}."}
            self.evaluated.add_line(start_fen, [move.uci()])
            label = number + board.san(move)  # written by the tool, not copied from the model's argument
            board.push(move)

        if board.is_game_over():
            self.checks.append(f"{CHECK_PREFIX}{label}, game over ({board.result(claim_draw=True)})")
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
        self.checks.append(f"{CHECK_PREFIX}{label}, {_format_eval(score.white())}")
        return result


def _format_eval(white: chess.engine.Score) -> str:
    """White's point of view, like the evaluation bar: +0.54, or mate in 3 for Black."""
    mate = white.mate()
    if mate is not None:
        return f"mate in {abs(mate)} for {'White' if mate > 0 else 'Black'}"
    return f"{white.score() / 100:+.2f}"


@dataclass
class CoachAnswer:
    text: str
    first_report: GroundingReport
    final_report: GroundingReport
    retried: bool
    checks: list[str] = field(default_factory=list)  # EngineTool.checks: "Checked with Stockfish: 15.Qd2, +0.54"


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

    def answer(self, question: str, game_context: dict, history_text: str, refs: References, language: str = "English",
               game_moves: Sequence[chess.Move] = ()) -> CoachAnswer:
        tool = EngineTool(self.analyse, self.describe_line, game_moves)
        system = SYSTEM_PROMPT.format(language=language)
        message = (
            "Game context:\n" + json.dumps(game_context, ensure_ascii=False, indent=1)
            + ("\n\nConversation so far:\n" + history_text if history_text else "")
            + "\n\nStudent's question: " + question
        )
        text = self.client.chat(system, [], message, tools=[tool.as_function()])
        first = check(text, refs, tool.evaluated)
        final, retried = first, False
        if not first.ok:
            retried = True
            history = [
                types.Content(role="user", parts=[types.Part(text=message)]),
                types.Content(role="model", parts=[types.Part(text=text)]),
            ]
            text = self.client.chat(system, history, _correction(first), tools=[tool.as_function()])
            final = check(text, refs, tool.evaluated)
        log.info(
            "grounding chat first_illegal=%s first_unverified=%s final_illegal=%s final_unverified=%s retried=%s",
            first.illegal, first.unverified, final.illegal, final.unverified, retried,
        )
        if not final.ok:
            flagged = final.illegal + final.unverified
            text += f"\n\n[Not verified by the engine: {', '.join(flagged)}]"
        return CoachAnswer(text, first, final, retried, tool.checks)


class Coach:
    """What the GUI uses: game commentary and chat, with errors turned into messages."""

    def __init__(self, api_key: str, model: Optional[str], analyse: Analyse, describe_line: DescribeLine, *, client: Optional[GeminiClient] = None):
        self.client = client or GeminiClient(api_key, model)
        self.describe_line = describe_line
        self.chat = CoachChat(self.client, analyse, describe_line)

    def comment_game(self, plies, language: str, username: str, player_color: str, game_moves: Sequence[chess.Move] = (), on_status=None):
        ctx = CommentaryContext(language, username, player_color)
        return generate_game_commentary(self.client, plies, ctx, self.describe_line, game_moves, on_status)

    def answer(self, question: str, game_context: dict, history_text: str, ply, game_moves: Sequence[chess.Move] = (),
               language: str = "English") -> tuple[str, list[str]]:
        """(answer text, one line per engine tool call). Errors become a message with no tool lines."""
        try:
            answer = self.chat.answer(question, game_context, history_text, references_for_ply(ply, game_moves), language, game_moves)
            return answer.text, answer.checks
        except LLMError as err:
            log.error("Coach chat failed (%s): %s", err.kind, err)
            if err.kind == "quota_exhausted":
                return "The Gemini daily quota is used up. Try again tomorrow or use another API key.", []
            if err.kind == "too_slow":
                return f"The Gemini model '{self.client.model}' is too slow right now. Try again later or change GEMINI_MODEL in .env.", []
            if err.kind == "model_not_found":
                return f"The Gemini model '{self.client.model}' is not available. Change GEMINI_MODEL in .env.", []
            return f"The coach could not answer right now ({err.kind}). Please try again.", []
