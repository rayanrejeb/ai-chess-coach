import json
import os
import threading
import time
from dataclasses import asdict

import chess
import pytest

from ai_chess_coach.config import Constants
from ai_chess_coach.engine import game as game_module
from ai_chess_coach.engine.game import GameAnalyzer, LineDescriber, find_key_moments, game_phase, san_line
from ai_chess_coach.models import PlyAnalysis
from fakes import FakePool
from golden_plies_cases import GOLDEN_PLIES_PATH, OPENING, PLIES, opera_moves


def test_matches_the_old_ply_builder():
    """Same PlyAnalysis records as the old builder (gui/app.py + legacy.py) on the same fake engine."""
    moves = opera_moves()
    plies = GameAnalyzer(FakePool()).analyze_game(moves[:PLIES], OPENING, total_plies=len(moves))
    got = json.loads(json.dumps([asdict(p) for p in plies], sort_keys=True, default=str))
    want = json.loads(GOLDEN_PLIES_PATH.read_text(encoding="utf-8"))
    for g, w in zip(got, want):
        assert g == w, f"ply {w['ply']} ({w['move_san']}) differs"
    assert len(got) == len(want)


def test_progress_is_reported_per_ply():
    seen = []
    moves = opera_moves()[:3]
    GameAnalyzer(FakePool()).analyze_game(moves, on_progress=lambda done, total: seen.append((done, total)))
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_game_phase():
    assert game_phase(0, 80) == "opening" and game_phase(23, 80) == "opening"
    assert game_phase(24, 80) == "middlegame" and game_phase(51, 80) == "endgame"


def test_san_line_stops_at_an_illegal_move():
    moves = [chess.Move.from_uci(u) for u in ("e2e4", "e7e5", "e4e5")]
    assert san_line(moves, chess.Board()) == ["e4", "e5"]


def test_key_moments_rank_losses_and_bonuses():
    plies = [PlyAnalysis(ply=1, points_loss=0.1), PlyAnalysis(ply=2, points_loss=0.4),
             PlyAnalysis(ply=3, points_loss=0.0, classification=Constants.BRILLIANT), PlyAnalysis(ply=4, points_loss=0.2)]
    assert [p.ply for p in find_key_moments(plies, 2)] == [2, 3]


def test_reused_motifs_respect_the_tactics_state():
    """Motifs already reported for the previous game move are filtered out, so a reused result must not cross states."""
    board = chess.Board()
    for san in "e4 e5 Nf3 d6 d4".split():
        board.push_san(san)
    pin = board.parse_san("Bg4")
    after = board.copy()
    after.push(pin)
    game = GameAnalyzer(FakePool())
    fresh = game._motifs(board, after, pin, None)
    assert any(m["tactic"] == "Pin" for m in fresh)
    game.tactics.find_tactical_motifs(board.copy(), after.copy(), pin, ply_index=5)  # the game move: updates the state
    assert game._motifs(board, after, pin, None) != fresh  # same move, new state: recomputed and filtered


def _opening_lines():
    """A few (fen, uci) lines from the first plies of the Opera game, cheap to describe."""
    board, lines = chess.Board(), []
    for move in opera_moves()[:4]:
        lines.append((board.fen(), [m.uci() for m in FakePool().analyse(board, 1)[0]["pv"]]))
        board.push(move)
    return lines


def test_describer_parallel_equals_serial():
    lines = _opening_lines()
    serial = LineDescriber(workers=1)
    parallel = LineDescriber(workers=2)
    parallel.prefetch(lines)
    assert [parallel(f, u) for f, u in lines] == [serial(f, u) for f, u in lines]
    assert all(serial(f, u)["san"] for f, u in lines)


def test_describer_returns_copies_and_ignores_game_state():
    fen, uci = _opening_lines()[2]
    describer = LineDescriber(workers=1)
    first = describer(fen, uci)
    first["san"] = "changed"
    game = GameAnalyzer(FakePool(), describer=describer)
    game.analyze_game(opera_moves()[:3])  # moves the game's tactics state, not the describer's
    assert describer(fen, uci) == LineDescriber(workers=1)(fen, uci) != first


def _middlegame_lines(count=10):
    """Lines from later Opera positions: slower to describe, so workers are busy for a while."""
    board, lines = chess.Board(), []
    for i, move in enumerate(opera_moves()[:24]):
        if i >= 24 - count:
            lines.append((board.fen(), [m.uci() for m in FakePool().analyse(board, 1)[0]["pv"]]))
        board.push(move)
    return lines


class _RecordingExecutor(game_module.ProcessPoolExecutor):
    """Keeps a handle on every worker process it started, to check none is left running."""
    processes: list = []

    def shutdown(self, *args, **kwargs):
        _RecordingExecutor.processes.extend((self._processes or {}).values())
        super().shutdown(*args, **kwargs)


def _wait_until(condition, timeout=60.0):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.05)
    return condition()


def test_worker_count_is_capped_by_the_cores():
    assert 1 <= game_module.DESCRIBE_WORKERS <= (os.cpu_count() or 1)


def test_close_stops_busy_workers():
    describer = LineDescriber(workers=2)
    worker = threading.Thread(target=describer.prefetch, args=(_middlegame_lines(),))
    worker.start()
    assert _wait_until(lambda: describer._executor is not None and describer._executor._processes)
    processes = list(describer._executor._processes.values())
    describer.close()
    worker.join(30)
    assert not worker.is_alive()
    assert processes and not any(p.is_alive() for p in processes)
    assert describer._done == {}  # interrupted: nothing stored
    describer.prefetch(_opening_lines())  # closed: does nothing
    assert describer._executor is None


def test_workers_are_stopped_after_an_error(monkeypatch):
    monkeypatch.setattr(game_module, "ProcessPoolExecutor", _RecordingExecutor)
    _RecordingExecutor.processes = []
    with pytest.raises(ValueError):
        LineDescriber(workers=2).prefetch(_opening_lines() + [("not a fen", ["e2e4"])])
    assert _RecordingExecutor.processes and not any(p.is_alive() for p in _RecordingExecutor.processes)


def test_falls_back_to_one_process_if_workers_cannot_start(monkeypatch):
    def no_workers(*args, **kwargs):
        raise OSError("no processes for you")

    monkeypatch.setattr(game_module, "ProcessPoolExecutor", no_workers)
    lines = _opening_lines()
    describer = LineDescriber(workers=2)
    describer.prefetch(lines)
    assert len(describer._done) == len({(f, tuple(u[:3])) for f, u in lines})
    assert [describer(f, u) for f, u in lines] == [LineDescriber(workers=1)(f, u) for f, u in lines]


def test_closing_the_analyzer_closes_the_describer():
    """The window's close handler calls GameAnalyzer.close(): no worker may outlive it."""
    game = GameAnalyzer(FakePool())
    game.close()
    assert game.describe_line._closed
