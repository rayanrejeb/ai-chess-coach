"""Move classification from engine scores, by win-probability loss.

Every score is turned into the mover's win probability with a logistic curve; the drop in that
probability caused by the move ("loss", 0..1) decides the class. Special cases (book moves,
tablebase results, forced mates, only moves, sacrifices) are checked first.
"""

import math
from typing import Optional

import chess
import chess.engine

from ai_chess_coach.config import Constants

MATE_SCORE = 10_000              # centipawn value given to a forced mate
WIN_PROB_SLOPE = 0.004           # logistic slope: win probability = 1 / (1 + e^(-slope * cp))

# Loss thresholds for an ordinary move that is not the engine's best move (upper bounds, exclusive)
EXCELLENT_MAX_LOSS = 0.03
GOOD_MAX_LOSS = 0.075
INACCURACY_MAX_LOSS = 0.15
MISTAKE_MAX_LOSS = 0.30          # anything above is a blunder

# The engine's best move is "great" if it turns the game around: (win prob before below, after at least)
GREAT_TURNAROUNDS = ((0.4, 0.6), (0.5, 0.7))
# ...or if it is the only good move: every other engine line loses more than this much win probability
ONLY_MOVE_MIN_GAP = 0.25
ONLY_MOVE_MAX_WIN_PROB = 0.8     # not counted when the position is already winning
ONLY_MOVE_LINES_COMPARED = 2     # second and third engine lines

# A great move is "brilliant" if it also leaves a piece en prise in a balanced position at no cost
SACRIFICE_WIN_PROB_RANGE = (0.1, 0.9)
SACRIFICE_MAX_LOSS = 0.05

MISS_MIN_LOSS = 0.05             # failing to punish an opponent's mistake or blunder
MATED_BLUNDER_MIN_CP = -800      # walking into a forced mate is a blunder unless already lost

# Tablebase verdicts: (class, loss, cp loss)
TABLEBASE_WIN_THROWN = (Constants.BLUNDER, 0.5, 500)
TABLEBASE_DRAW_THROWN = (Constants.MISTAKE, 0.2, 200)
NO_ANALYSIS = (Constants.BLUNDER, 1.0, MATE_SCORE)


def win_probability(cp: Optional[int]) -> float:
    """Win probability (0..1) for the side the centipawn score belongs to; 0.5 if unknown."""
    if cp is None:
        return 0.5
    cp = max(-MATE_SCORE, min(MATE_SCORE, cp))
    return 1 / (1 + math.exp(-WIN_PROB_SLOPE * cp))


def _mate_in(score: chess.engine.Score) -> int:
    """Moves to mate (> 0: the side wins, < 0: it gets mated), 0 if no forced mate."""
    return score.mate() or 0


def _probe_wdl(tablebases, board: chess.Board) -> Optional[int]:
    if tablebases is None or len(board.piece_map()) > tablebases.max_pieces:
        return None
    try:
        return tablebases.probe_wdl(board)
    except (IndexError, KeyError):
        return None


def _is_free_sacrifice(board: chess.Board, move: chess.Move, cp_before: int, loss: float) -> bool:
    piece = board.piece_at(move.from_square)
    leaves_piece_en_prise = (
        piece is not None and piece.piece_type != chess.PAWN
        and not board.is_capture(move) and board.is_attacked_by(not board.turn, move.to_square)
    )
    low, high = SACRIFICE_WIN_PROB_RANGE
    return leaves_piece_en_prise and low < win_probability(cp_before) < high and loss < SACRIFICE_MAX_LOSS


def _is_great(lines_before: list, mover: chess.Color, wp_before: float, wp_after: float) -> bool:
    if any(wp_before < below and wp_after >= at_least for below, at_least in GREAT_TURNAROUNDS):
        return True
    if len(lines_before) < 2 or wp_before >= ONLY_MOVE_MAX_WIN_PROB:
        return False
    others = [line["score"].pov(mover).score(mate_score=MATE_SCORE) for line in lines_before[1:1 + ONLY_MOVE_LINES_COMPARED]]
    gaps = [wp_before - win_probability(cp) for cp in others if cp is not None]
    return bool(gaps) and max(gaps) > ONLY_MOVE_MIN_GAP


def classify_move(
    board: chess.Board,
    move: chess.Move,
    lines_before: list,
    info_after: dict,
    opponent_class: Optional[str] = None,
    *,
    opening_book=None,
    tablebases=None,
) -> tuple[str, float, int]:
    """(class, win-probability loss, centipawn loss) for `move` played in `board`.

    lines_before: engine lines (python-chess InfoDicts, best first) for the position before the move.
    info_after: the engine's InfoDict for the position after the move.
    opponent_class: class of the opponent's previous move, to detect missed punishments.
    """
    if opening_book is not None:
        try:
            if opening_book.find(board, default=None):
                return Constants.BOOK_MOVE, 0.0, 0
        except Exception:
            pass  # a broken book entry must not stop the analysis

    if not lines_before or not lines_before[0].get("pv"):
        return NO_ANALYSIS
    mover = board.turn
    best_move = lines_before[0]["pv"][0]
    score_before = lines_before[0]["score"].pov(mover)
    score_after = info_after["score"].pov(mover)
    cp_before = score_before.score(mate_score=MATE_SCORE)
    cp_after = score_after.score(mate_score=MATE_SCORE)

    wdl_before = _probe_wdl(tablebases, board)
    if wdl_before is not None:
        after_board = board.copy()
        after_board.push(move)
        wdl_after = _probe_wdl(tablebases, after_board)
        if wdl_after is not None:
            if wdl_before > 0 and wdl_after <= 0:
                return TABLEBASE_WIN_THROWN
            if wdl_before == 0 and wdl_after < 0:
                return TABLEBASE_DRAW_THROWN

    if cp_before is None or cp_after is None:
        return NO_ANALYSIS
    wp_before, wp_after = win_probability(cp_before), win_probability(cp_after)
    loss = wp_before - wp_after
    cp_loss = cp_before - cp_after
    mate_before, mate_after = _mate_in(score_before), _mate_in(score_after)

    if mate_after > 0 and mate_before <= 0:
        return Constants.BRILLIANT, loss, cp_loss  # found a forced mate that was not there before
    if move == best_move:
        if not _is_great(lines_before, mover, wp_before, wp_after):
            return Constants.BEST, loss, cp_loss
        if _is_free_sacrifice(board, move, cp_before, loss):
            return Constants.BRILLIANT, loss, cp_loss
        return Constants.GREAT_MOVE, loss, cp_loss
    if mate_before > 0 and (not score_after.is_mate() or mate_after < 0):
        return Constants.MISS, loss, cp_loss  # had a forced mate and let it go
    if mate_after < 0 and cp_before > MATED_BLUNDER_MIN_CP:
        return Constants.BLUNDER, loss, cp_loss
    if opponent_class in (Constants.BLUNDER, Constants.MISTAKE) and loss > MISS_MIN_LOSS:
        return Constants.MISS, loss, cp_loss

    for limit, name in ((EXCELLENT_MAX_LOSS, Constants.EXCELLENT), (GOOD_MAX_LOSS, Constants.GOOD),
                        (INACCURACY_MAX_LOSS, Constants.INACCURACY), (MISTAKE_MAX_LOSS, Constants.MISTAKE)):
        if loss < limit:
            return name, loss, cp_loss
    return Constants.BLUNDER, loss, cp_loss
