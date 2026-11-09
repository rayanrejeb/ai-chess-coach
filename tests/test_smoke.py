import importlib
import json

import pytest

from golden_cases import GOLDEN_PATH, compute, stockfish_path

MODULES = [
    "ai_chess_coach.config",
    "ai_chess_coach.models",
    "ai_chess_coach.engine.positional",
    "ai_chess_coach.engine.classify",
    "ai_chess_coach.engine.pool",
    "ai_chess_coach.engine.game",
    "ai_chess_coach.tactics.analyzer",
    "ai_chess_coach.llm.client",
    "ai_chess_coach.llm.grounding",
    "ai_chess_coach.llm.commentary",
    "ai_chess_coach.llm.chat",
    "ai_chess_coach.sync",
    "ai_chess_coach.sync.__main__",
    "ai_chess_coach.gui.app",
    "main",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def _current(with_engine):
    from ai_chess_coach.config import AnalysisConfig
    from ai_chess_coach.engine.classify import classify_move, win_probability
    from ai_chess_coach.engine.positional import PositionalAnalyzer
    from ai_chess_coach.tactics.analyzer import TacticalAnalyzer

    return compute(PositionalAnalyzer, TacticalAnalyzer, classify_move, win_probability, AnalysisConfig, with_engine=with_engine)


def _strip_engine(data):
    data = json.loads(json.dumps(data))
    data.pop("engine_id", None)
    for p in data["positions"]:
        p.pop("engine_before", None)
        p.pop("engine_after", None)
    return data


@pytest.fixture(scope="module")
def golden():
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def test_analysis_matches_golden(golden):
    """Positional features, tactical motifs, SEE and move classification are unchanged."""
    assert _current(with_engine=False) == _strip_engine(golden)


@pytest.mark.skipif(not stockfish_path(), reason="Stockfish not found (set STOCKFISH_PATH)")
def test_engine_matches_golden(golden):
    current = _current(with_engine=True)
    if current["engine_id"] != golden["engine_id"]:
        pytest.skip(f"golden made with {golden['engine_id']}, found {current['engine_id']}")
    for got, want in zip(current["positions"], golden["positions"]):
        assert got["engine_before"] == want["engine_before"]
        assert got["engine_after"] == want["engine_after"]


def test_sync_count_cli(tmp_path, monkeypatch, capsys):
    """Counts the saved games per platform offline, without printing the usernames."""
    from ai_chess_coach.config import Constants
    from ai_chess_coach.sync.__main__ import main

    (tmp_path / "ccname_chess.com_games.json").write_text(json.dumps([{"url": "a"}, {"url": "b"}]), encoding="utf-8")
    (tmp_path / "liname_lichess_games.json").write_text(json.dumps([{"url": "c"}]), encoding="utf-8")
    monkeypatch.setattr(Constants, "GAMES_DIR", str(tmp_path))
    monkeypatch.setenv("CHESSCOM_USERNAME", "CCName")
    monkeypatch.setenv("LICHESS_USERNAME", "LIName")
    main(["count"])
    out = capsys.readouterr().out
    assert out.splitlines() == ["chess.com: 2 games", "lichess: 1 games", "total: 3 games"]
    assert "ccname" not in out.lower() and "liname" not in out.lower()


def test_gui_opens_and_closes_without_stockfish(tmp_path, monkeypatch):
    """The window builds with nothing configured, the settings window opens, and closing saves nothing unwanted."""
    import tkinter as tk

    monkeypatch.chdir(tmp_path)  # config.json, fetched_games/ and analyzed_games/ stay out of the repo
    for var in ("STOCKFISH_PATH", "GEMINI_API_KEY", "DEFAULT_USERNAME", "LICHESS_TOKEN"):
        monkeypatch.setenv(var, "")
    from ai_chess_coach.gui.app import ChessGUI

    try:
        app = ChessGUI()
    except tk.TclError as e:
        pytest.skip(f"no display: {e}")
    try:
        app.update()
        assert app.analyzer is None
        app.open_settings_window()
        app.update()

        def texts(widget):
            for child in widget.winfo_children():
                try:
                    yield str(child.cget("text"))
                except (tk.TclError, ValueError):
                    pass
                yield from texts(child)

        labels = list(texts(app))
        assert "Stockfish Path:" in labels
        assert not any("Depth" in t for t in labels)
        settings = next(w for w in app.winfo_children() if w.__class__.__name__ == "CTkToplevel")
        assert settings.winfo_height() == settings.winfo_reqheight()  # fits its contents, no empty space
    finally:
        app.on_closing()
    assert not (tmp_path / "config.json").exists()  # no username entered, so nothing to save
