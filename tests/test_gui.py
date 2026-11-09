"""GUI behavior with a fake engine: no Stockfish, no Gemini. Skipped when there is no display.

Each test runs the window in its own process: a second Tk window in one process sometimes fails to load
Tcl on Windows ("Can't find a usable init.tcl"), which would make test_smoke's window test skip.
"""

import json
import os
import subprocess
import sys
import textwrap
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


PROLOGUE = """
import json, os, sys
sys.path[:0] = [{root!r}, {tests!r}]
os.chdir({tmp!r})
from test_gui import FakeAnalyzer, one_move_game
from ai_chess_coach.gui.app import ChessGUI
app = ChessGUI()
app.analyzer = FakeAnalyzer()
app.moves, app.analysis_data = one_move_game()
out = {{}}

def run_for(ms):
    app.after(ms, app.quit)
    app.mainloop()  # worker threads call app.after(), which needs a running mainloop
"""


def run_gui(tmp_path, body):
    """Runs PROLOGUE + body in a new process; body fills `out`, which is returned."""
    env = {**os.environ, "STOCKFISH_PATH": "", "GEMINI_API_KEY": "", "DEFAULT_USERNAME": "", "LICHESS_TOKEN": ""}
    script = PROLOGUE.format(root=str(ROOT), tests=str(ROOT / "tests"), tmp=str(tmp_path)) + textwrap.dedent(body) + \
        "\nprint('OUT=' + json.dumps(out))\n"
    try:
        result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        pytest.fail("the GUI process was still running after 60 s")
    if "TclError" in result.stderr and ("display" in result.stderr or "init.tcl" in result.stderr):
        pytest.skip(f"no usable Tk: {result.stderr.strip().splitlines()[-1]}")
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.split("OUT=")[-1])


def test_selecting_a_move_stops_redrawing_once_idle(tmp_path):
    """The refresh after selecting a move redraws once; the redraw's own refresh finds the same lines and stops.
    Resizes at start-up also redraw, so the count varies; it must stop growing once the window is idle."""
    out = run_gui(tmp_path, """
        app.after(0, app.go_to_ply, 1)
        previous = -1
        for _ in range(20):  # start-up redraws take longer on a busy machine: wait (up to 10 s) for a stable count
            run_for(500)
            if app.analyzer.calls == previous:
                break
            previous = app.analyzer.calls
        out["settled"] = app.analyzer.calls
        run_for(1000)
        out["later"] = app.analyzer.calls
        out["lines"] = app.analysis_data[0].raw_engine_lines_for_display
        app.on_closing()
    """)
    first = next(iter(chess.Board().legal_moves)).uci()
    assert out["lines"] == [{"line": [first], "score": "+30"}]
    assert 0 < out["settled"] == out["later"]


def test_process_exits_after_closing_with_a_move_selected(tmp_path):
    out = run_gui(tmp_path, """
        app.after(0, app.go_to_ply, 1)
        app.after(1000, app.on_closing)
        app.mainloop()
        out["closed"] = True
    """)
    assert out == {"closed": True}  # and run_gui saw the process end


SEND_BODY = """
    class FakeCoach:
        def __init__(self, reply):
            self.reply = reply

        def answer(self, *args):
            if isinstance(self.reply, Exception):
                raise self.reply
            return self.reply

    app.after(0, app.go_to_ply, 1)
    run_for(300)
    for reply in ("Fine move.", RuntimeError("boom")):
        app.coach = FakeCoach(reply)
        app.chat_entry.insert(0, "Was 1.e4 good?")
        app.send_chat_message()
        out.setdefault("while_waiting", app.send_chat_btn.cget("text"))
        run_for(500)
        out.setdefault("after", []).append([app.send_chat_btn.cget("text"), app.send_chat_btn.cget("state")])
    out["chat"] = app.chat_history.get("1.0", "end")
    app.on_closing()
"""


def test_send_button_comes_back_after_an_answer_and_after_an_error(tmp_path):
    out = run_gui(tmp_path, SEND_BODY)
    assert out["while_waiting"] == "..."
    assert out["after"] == [["Send", "normal"], ["Send", "normal"]]
    assert "Coach: Fine move." in out["chat"] and "System Error: An error occurred in the chat task: boom" in out["chat"]


def test_chat_shows_double_asterisks_as_bold(tmp_path):
    out = run_gui(tmp_path, """
        import tkinter.font as tkfont
        app._display_chat_message("Coach", "Yes, **13.f3** was rated **Excellent**. Unpaired ** stays.")
        box = app.chat_history._textbox
        out["text"] = box.get("1.0", "end").strip()
        ranges = [str(r) for r in box.tag_ranges("bold")]
        out["bold"] = [box.get(a, b) for a, b in zip(ranges[::2], ranges[1::2])]
        bold, normal = tkfont.Font(font=box.tag_cget("bold", "font")), tkfont.Font(font=box.cget("font"))
        out["fonts"] = [bold.actual("weight"), bold.actual("size") == normal.actual("size"), bold.actual("family") == normal.actual("family")]
        app.on_closing()
    """)
    assert out["text"] == "Coach: Yes, 13.f3 was rated Excellent. Unpaired ** stays."
    assert out["bold"] == ["13.f3", "Excellent"]
    assert out["fonts"] == ["bold", True, True]
