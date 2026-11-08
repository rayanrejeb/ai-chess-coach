"""Check that every move an LLM answer mentions is backed by the position and the engine.

A mentioned move is:
  - verified:   cited with its move number and matching the game move at that number
                ("7.Qb3", "13...Rxd7"), part of an engine line, or evaluated by the engine tool
  - unverified: legal in a reference position, but not backed by the engine
  - illegal:    not legal in any reference position (an invented move)
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional, Sequence

import chess

log = logging.getLogger(__name__)

_MOVE_RE = re.compile(
    r"""
    (?<![\w.])                                   # not glued to a word or a previous token
    (?:(?P<num>\d+)\s*(?P<dots>\.\.\.|\.|…)\s*)?  # optional move number: 13. / 13... / 13…
    (?P<san>
        [O0]-[O0](?:-[O0])?                      # castling (also written with zeros)
      | [KQRBN][a-h]?[1-8]?x?[a-h][1-8]          # piece move: Nf3, Nbd7, R1e2, Rxd7
      | [a-h](?:x[a-h])?[1-8](?:=?[QRBN])?       # pawn move: e4, exd5, e8=Q, e8Q
    )
    (?P<suffix>[+#]?(?:!!|\?\?|!\?|\?!|!|\?)?)   # check/mate and annotation: +, #, !, ?, !!, ??, !?, ?!
    (?![\w])
    """,
    re.VERBOSE,
)

_BARE_SQUARE_RE = re.compile(r"[a-h][1-8]")
_SQUARE_CONTEXT_RE = re.compile(r"\b(on|to|the|at|from|of|square|squares|towards|into|onto|via)\s*$", re.IGNORECASE)


def normalize_san(san: str) -> str:
    """Canonical form for comparison: no check/annotation marks, letter-O castling, '=' before promotion."""
    san = san.rstrip("+#!?").replace("0", "O")
    if re.fullmatch(r"[a-h](?:x[a-h])?[1-8][QRBN]", san):
        san = san[:-1] + "=" + san[-1]
    return san


# (fullmove number, is Black's move), e.g. "13...Rxd7" -> (13, True)
Cite = tuple[int, bool]


def _mentions(text: str) -> Iterator[tuple[str, Optional[Cite]]]:
    """Every move written in English SAN, normalized, with its move number if one was written."""
    for m in _MOVE_RE.finditer(text):
        token = m.group("san")
        is_bare_square = _BARE_SQUARE_RE.fullmatch(token) and not m.group("suffix") and m.group(0) == token
        # LIMITATION: a bare square ("e4") after words like "on", "to", "the" is read as a square name,
        # not a pawn move, so "the knight on f3" is not counted. A real pawn move written that way is missed.
        if is_bare_square and _SQUARE_CONTEXT_RE.search(text[: m.start()]):
            continue
        cite = (int(m.group("num")), m.group("dots") != ".") if m.group("num") else None
        yield normalize_san(token), cite


def extract_moves(text: str) -> list[str]:
    """Moves written in English SAN, normalized, in order of first appearance."""
    return list(dict.fromkeys(san for san, _ in _mentions(text)))


@dataclass
class References:
    """Positions where a mentioned move may be played, and moves the engine or the game backs."""

    positions: list[chess.Board] = field(default_factory=list)
    verified: set[str] = field(default_factory=set)
    game_moves: dict[Cite, str] = field(default_factory=dict)

    def add_position(self, board: chess.Board) -> None:
        if all(b.fen() != board.fen() for b in self.positions):
            self.positions.append(board.copy(stack=False))

    def add_line(self, fen: str, uci_moves: Iterable[str]) -> None:
        """Every position along an engine line becomes a reference; every move in it is verified."""
        board = chess.Board(fen)
        self.add_position(board)
        for uci in uci_moves:
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                break
            self.verified.add(normalize_san(board.san(move)))
            board.push(move)
            self.add_position(board)

    def add_game_moves(self, moves: Sequence[chess.Move], start_fen: str = chess.STARTING_FEN) -> None:
        """Moves actually played, by move number. They are verified only when cited with the right number."""
        board = chess.Board(start_fen)
        for move in moves:
            if move not in board.legal_moves:
                break
            self.game_moves[(board.fullmove_number, board.turn == chess.BLACK)] = normalize_san(board.san(move))
            board.push(move)


def references_for_ply(ply, game_moves: Sequence[chess.Move] = (), start_fen: str = chess.STARTING_FEN) -> References:
    """References for one analyzed ply (a models.PlyAnalysis): positions before/after it and its engine lines."""
    refs = References()
    before = chess.Board(ply.fen_before)
    refs.add_position(before)
    after = before.copy()
    if ply.player_line.uci:
        played = chess.Move.from_uci(ply.player_line.uci[0])
        if played in before.legal_moves:
            refs.verified.add(normalize_san(before.san(played)))
            after.push(played)
            refs.add_position(after)
    for line in ply.best_engine_lines or [ply.best_engine_line]:
        refs.add_line(before.fen(), line.uci)
    for line in ply.best_follow_up_lines or [ply.best_follow_up]:
        refs.add_line(after.fen(), line.uci)
    refs.add_game_moves(game_moves, start_fen)
    return refs


@dataclass
class GroundingReport:
    moves: list[str] = field(default_factory=list)
    illegal: list[str] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.illegal and not self.unverified


def _is_legal_somewhere(san: str, positions: Iterable[chess.Board]) -> bool:
    for board in positions:
        try:
            board.parse_san(san)
            return True
        except chess.AmbiguousMoveError:
            return True  # the notation fits more than one piece, but such a move exists
        except ValueError:
            continue
    return False


def check(text: str, refs: References, extra: Optional[References] = None) -> GroundingReport:
    positions = refs.positions + (extra.positions if extra else [])
    verified = refs.verified | (extra.verified if extra else set())
    report = GroundingReport(moves=extract_moves(text))
    for san, cite in _mentions(text):
        if san in verified or (cite and refs.game_moves.get(cite) == san):
            continue
        bucket = report.unverified if _is_legal_somewhere(san, positions) else report.illegal
        if san not in bucket:
            bucket.append(san)
    return report
