"""Per-move coaching commentary for a whole game, generated in chunks and grounding-checked."""

import json
import logging
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import chess
from pydantic import BaseModel

from ai_chess_coach.llm.client import GeminiClient, LLMError
from ai_chess_coach.llm.grounding import check, references_for_ply
from ai_chess_coach.models import GeminiCommentary, PositionalFeatures

log = logging.getLogger(__name__)

MAX_PLIES_PER_CHUNK = 24       # keeps each answer well inside the model's output limit
MAX_CHARS_PER_CHUNK = 40_000   # input budget per request (prompt data, not tokens)
MAX_LINES_PER_PLY = 2          # engine alternatives and replies sent per move
MAX_TACTICS_PER_PLY = 3

# describe_line(fen, uci_moves) -> dict with SAN and per-move tactical/positional notes
DescribeLine = Callable[[str, list[str]], dict]


class MoveComment(BaseModel):
    ply: int
    move_san: str
    move_commentary: str
    missed_opportunities: list[str] = []
    best_replies: list[str] = []


class GameSummary(BaseModel):
    white_takeaway: str = ""
    black_takeaway: str = ""
    white_pattern: str = ""
    black_pattern: str = ""


class CommentaryBatch(BaseModel):
    comments: list[MoveComment]  # a list, not a dict: the Developer API rejects map schemas
    summary: Optional[GameSummary] = None


@dataclass
class CommentaryContext:
    language: str
    username: str
    player_color: str


SYSTEM_PROMPT = """You are a chess coach writing move-by-move feedback for a student.

Rules:
- Write explanations in {language}.
- Write every chess move in English Standard Algebraic Notation (SAN): K Q R B N for pieces, O-O and O-O-O for castling, e.g. Nf3, Bxb5+, exd5, e8=Q. Never translate piece letters, whatever the language.
- Only name moves that appear in the data you are given (the move played, engine alternatives, engine replies) or that were played in the game. Do not invent variations.
- Base each comment on the data: classification, engine lines, tactical motifs and positional notes. Explain why, not just what.
- Return exactly one comment per ply in the input, with the same ply number.
"""


def _positional_notes(features: PositionalFeatures) -> list[str]:
    notes = []
    for color in ("white", "black"):
        status = features.king_safety.get(color, {}).get("status")
        if status and status != "safe":
            notes.append(f"{color} king {status}")
    for kind, squares in features.pawn_structure.items():
        if squares:
            notes.append(f"{kind.replace('_', ' ')}: {', '.join(squares)}")
    for items in features.rook_activity.values():
        notes.extend(items)
    notes.extend(features.advantages)
    return notes


def build_ply_payload(ply, describe_line: DescribeLine, already_reported: set[str]) -> dict:
    """What the model sees for one ply. Positional notes already sent earlier are dropped to save space."""
    after_fen = ply.fen_before
    if ply.player_line.uci:
        board = chess.Board(ply.fen_before)
        board.push(chess.Move.from_uci(ply.player_line.uci[0]))
        after_fen = board.fen()

    new_notes = [n for n in _positional_notes(ply.positional_analysis_after) if n not in already_reported]
    already_reported.update(new_notes)
    alternatives = [line for line in ply.best_engine_lines if line.uci and line.uci[:1] != ply.player_line.uci[:1]]
    return {
        "ply": ply.ply,
        "side": ply.player,
        "move": ply.move_san,
        "classification": ply.classification,
        "phase": ply.game_phase,
        "tactics": [
            {"tactic": m["tactic"], "details": m.get("details", "")}
            for m in ply.player_line.tactical_motifs
            if m.get("tactic") != "None"
        ][:MAX_TACTICS_PER_PLY],
        "positional_notes": new_notes,
        "engine_alternatives": [describe_line(ply.fen_before, line.uci) for line in alternatives[:MAX_LINES_PER_PLY]],
        "engine_replies": [describe_line(after_fen, line.uci) for line in ply.best_follow_up_lines[:MAX_LINES_PER_PLY] if line.uci],
    }


def chunk_payloads(payloads: list[dict], max_plies: Optional[int] = None, max_chars: Optional[int] = None) -> list[list[dict]]:
    max_plies = max_plies or MAX_PLIES_PER_CHUNK
    max_chars = max_chars or MAX_CHARS_PER_CHUNK
    chunks, current, size = [], [], 0
    for payload in payloads:
        length = len(json.dumps(payload))
        if current and (len(current) >= max_plies or size + length > max_chars):
            chunks.append(current)
            current, size = [], 0
        current.append(payload)
        size += length
    if current:
        chunks.append(current)
    return chunks


def recurring_patterns(plies) -> dict[str, str]:
    """Most frequent tactic behind each side's mistakes and blunders."""
    patterns = {}
    for side in ("White", "Black"):
        causes = Counter()
        for ply in plies:
            if ply.player != side or ply.classification not in ("Mistake", "Blunder"):
                continue
            causes.update(f"allowed a {m['tactic']}" for m in ply.best_follow_up.tactical_motifs if m.get("tactic") != "None")
            causes.update(f"missed a {m['tactic']}" for m in ply.best_engine_line.tactical_motifs if m.get("tactic") != "None")
        if causes:
            patterns[f"{side.lower()}_pattern"] = causes.most_common(1)[0][0]
    return patterns


def _chunk_prompt(ctx: CommentaryContext, patterns: dict, chunk: list[dict], index: int, total: int) -> str:
    last = index == total - 1
    parts = [
        f"Student: {ctx.username}, playing {ctx.player_color}.",
        f"Part {index + 1} of {total} of the game (plies {chunk[0]['ply']}-{chunk[-1]['ply']}).",
    ]
    if patterns:
        parts.append("Recurring patterns found by the engine: " + json.dumps(patterns))
    parts.append(
        "This is the last part: also fill `summary` with one takeaway and one recurring pattern per side."
        if last
        else "Leave `summary` empty."
    )
    parts.append("Moves to comment on:\n" + json.dumps(chunk, ensure_ascii=False))
    return "\n".join(parts)


def generate_game_commentary(
    client: GeminiClient,
    plies: Sequence,
    ctx: CommentaryContext,
    describe_line: DescribeLine,
    game_moves: Sequence[chess.Move] = (),
    on_status: Optional[Callable[[str], None]] = None,
) -> tuple[dict[str, GeminiCommentary], Optional[dict]]:
    """Returns ({ply number as str: commentary}, summary dict or None). Stops early if the daily quota runs out."""
    reported: set[str] = set()
    payloads = [build_ply_payload(ply, describe_line, reported) for ply in plies]
    chunks = chunk_payloads(payloads)
    patterns = recurring_patterns(plies)
    system = SYSTEM_PROMPT.format(language=ctx.language)
    by_ply = {ply.ply: ply for ply in plies}

    comments: dict[str, GeminiCommentary] = {}
    summary = None
    for i, chunk in enumerate(chunks):
        if on_status:
            on_status(f"AI Part {i + 1}/{len(chunks)}...")
        try:
            batch = client.generate_json(_chunk_prompt(ctx, patterns, chunk, i, len(chunks)), CommentaryBatch, system=system)
        except LLMError as err:
            log.error("Commentary part %d/%d failed (%s): %s", i + 1, len(chunks), err.kind, err)
            if err.kind == "quota_exhausted":
                break
            continue
        for c in batch.comments:
            ply = by_ply.get(c.ply)
            if ply is None:
                continue
            report = check(" ".join([c.move_commentary, *c.missed_opportunities, *c.best_replies]), references_for_ply(ply, game_moves))
            if not report.ok:
                log.info("grounding commentary ply=%d illegal=%s unverified=%s", c.ply, report.illegal, report.unverified)
            comments[str(c.ply)] = GeminiCommentary(
                ply=c.ply,
                move_san=c.move_san,
                move_commentary=c.move_commentary,
                missed_opportunities=c.missed_opportunities,
                best_replies=c.best_replies,
                illegal_moves=report.illegal,
                unverified_moves=report.unverified,
            )
        if batch.summary and i == len(chunks) - 1:
            summary = batch.summary.model_dump()
    return comments, summary
