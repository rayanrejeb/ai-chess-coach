import importlib
import json

import pytest

from golden_cases import GOLDEN_PATH, compute, stockfish_path

MODULES = [
    "ai_chess_coach.config",
    "ai_chess_coach.models",
    "ai_chess_coach.engine.positional",
    "ai_chess_coach.tactics.analyzer",
    "ai_chess_coach.llm.client",
    "ai_chess_coach.llm.grounding",
    "ai_chess_coach.llm.commentary",
    "ai_chess_coach.llm.chat",
    "ai_chess_coach.legacy",
    "ai_chess_coach.gui.app",
    "main",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def _current(with_engine):
    from ai_chess_coach.config import AnalysisConfig
    from ai_chess_coach.engine.positional import PositionalAnalyzer
    from ai_chess_coach.legacy import ChessAnalyzer
    from ai_chess_coach.tactics.analyzer import TacticalAnalyzer

    return compute(PositionalAnalyzer, TacticalAnalyzer, ChessAnalyzer, AnalysisConfig, with_engine=with_engine)


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
