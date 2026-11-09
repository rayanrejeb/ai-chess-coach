"""GUI behavior with a fake engine: no Stockfish, no Gemini. Skipped when there is no display."""

import os
import subprocess
import sys
import textwrap
import tkinter as tk
from pathlib import Path

import chess
import chess.engine
import pytest

from ai_chess_coach.config import AnalysisConfig
from ai_chess_coach.models import LineAnalysis, PlyAnalysis

ROOT = Path(__file__).resolve().parent.parent


class FakeAnalyzer:
    """Always gives the same line, like the engine pool answering from its cache."""

    def __init__(self):
        self.config = AnalysisConfig()
        self.calls = 0

    def analyse(self, board, depth, multipv, time_limit):
        self.calls += 1
        return [{"pv": [next(iter(board.legal_moves))], "score": chess.engine.PovScore(chess.engine.Cp(30), chess.WHITE)}]

    def describe_line(self, fen, uci):
        return {"san": [], "moves": []}

    def close(self):
        pass


def one_move_game():
    ply = PlyAnalysis(ply=1, move_number_str="1", player="White", move_san="e4", fen_before=chess.STARTING_FEN,
                      classification="Best", player_line=LineAnalysis(uci=["e2e4"]),
                      best_engine_line=LineAnalysis(uci=["e2e4"]), best_follow_up=LineAnalysis(uci=["e7e5"]))
    return [chess.Move.from_uci("e2e4")], [ply]


def open_app(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # config.json, fetched_games/ and analyzed_games/ stay out of the repo
    for var in ("STOCKFISH_PATH", "GEMINI_API_KEY", "DEFAULT_USERNAME", "LICHESS_TOKEN"):
        monkeypatch.setenv(var, "")
    from ai_chess_coach.gui.app import ChessGUI

    try:
        return ChessGUI()
    except tk.TclError as e:
        pytest.skip(f"no display: {e}")


def test_selecting_a_move_refines_its_engine_lines_once(monkeypatch, tmp_path):
    """The refresh after selecting a move redraws once; the redraw's own refresh finds the same lines and stops.
    (Window resizes at start-up also redraw, so the exact count varies; it must stop growing once idle.)"""
    app = open_app(monkeypatch, tmp_path)

    def run_for(ms):
        app.after(ms, app.quit)
        app.mainloop()  # workers call app.after(), which needs a running mainloop

    try:
        app.analyzer = FakeAnalyzer()
        app.moves, app.analysis_data = one_move_game()
        app.after(0, app.go_to_ply, 1)
        run_for(1500)
        settled = app.analyzer.calls
        run_for(1000)
        first = next(iter(chess.Board().legal_moves)).uci()
        assert app.analysis_data[0].raw_engine_lines_for_display == [{"line": [first], "score": "+30"}]
        assert 0 < settled == app.analyzer.calls
    finally:
        app.on_closing()


CLOSE_SCRIPT = textwrap.dedent("""
    import os, sys
    sys.path[:0] = [{root!r}, {tests!r}]
    os.chdir({tmp!r})
    from test_gui import FakeAnalyzer, one_move_game
    from ai_chess_coach.gui.app import ChessGUI
    app = ChessGUI()
    app.analyzer = FakeAnalyzer()
    app.moves, app.analysis_data = one_move_game()
    app.after(0, app.go_to_ply, 1)
    app.after(1000, app.on_closing)
    app.mainloop()
    print("closed")
""")


def test_process_exits_after_closing_with_a_move_selected(tmp_path):
    env = {**os.environ, "STOCKFISH_PATH": "", "GEMINI_API_KEY": "", "DEFAULT_USERNAME": "", "LICHESS_TOKEN": ""}
    script = CLOSE_SCRIPT.format(root=str(ROOT), tests=str(ROOT / "tests"), tmp=str(tmp_path))
    try:
        result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        pytest.fail("the process was still running 60 s after the window closed")
    if "TclError" in result.stderr and "display" in result.stderr:
        pytest.skip("no display")
    assert result.returncode == 0, result.stderr[-2000:]
    assert "closed" in result.stdout
