import chess
import chess.engine
import pytest

from ai_chess_coach.config import Constants
from ai_chess_coach.engine.classify import (
    EXCELLENT_MAX_LOSS, GOOD_MAX_LOSS, INACCURACY_MAX_LOSS, MISTAKE_MAX_LOSS, classify_move, win_probability,
)


def test_win_probability_curve():
    assert win_probability(0) == 0.5 and win_probability(None) == 0.5
    assert win_probability(300) == pytest.approx(1 - win_probability(-300))
    assert win_probability(10**9) == win_probability(10_000)  # mate scores are capped


def _cp_for_loss(loss):
    """Centipawns after the move so that the win-probability loss from 0 cp is `loss`."""
    import math
    return -math.log(1 / (0.5 - loss) - 1) / 0.004


@pytest.mark.parametrize("limit, name", [
    (EXCELLENT_MAX_LOSS, Constants.EXCELLENT), (GOOD_MAX_LOSS, Constants.GOOD),
    (INACCURACY_MAX_LOSS, Constants.INACCURACY), (MISTAKE_MAX_LOSS, Constants.MISTAKE),
])
def test_loss_thresholds(limit, name):
    board = chess.Board()
    best, played = chess.Move.from_uci("e2e4"), chess.Move.from_uci("a2a3")
    lines = [{"score": chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE), "pv": [best]}]

    def classify(loss):
        after = {"score": chess.engine.PovScore(chess.engine.Cp(round(_cp_for_loss(loss))), chess.WHITE)}
        return classify_move(board, played, lines, after)[0]

    assert classify(limit - 0.005) == name
    assert classify(limit + 0.005) != name
