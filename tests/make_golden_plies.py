"""Write tests/golden_plies.json: the old ply builder's output on the Opera game with a fake engine.

Run once, before engine/ replaced the old builder:  python tests/make_golden_plies.py
The old builder (gui/app.py _build_ply_analysis + legacy.ChessAnalyzer) was removed afterwards;
to regenerate, check out commit 3ce8dc1 (the commit that added this file).
"""

import json
import sys
import time
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

import chess  # noqa: E402

from ai_chess_coach.config import AnalysisConfig  # noqa: E402
from ai_chess_coach.engine.positional import PositionalAnalyzer  # noqa: E402
from ai_chess_coach.gui.app import ChessGUI  # noqa: E402
from ai_chess_coach.legacy import ChessAnalyzer  # noqa: E402
from ai_chess_coach.tactics.analyzer import TacticalAnalyzer  # noqa: E402
from fakes import fake_engine_lines  # noqa: E402
from golden_plies_cases import GOLDEN_PLIES_PATH, OPENING, PLIES, opera_moves  # noqa: E402

analyzer = ChessAnalyzer.__new__(ChessAnalyzer)  # no Stockfish: the fake engine answers
analyzer.config = AnalysisConfig()
analyzer.positional_analyzer = PositionalAnalyzer(analyzer.config)
analyzer.tactical_analyzer = TacticalAnalyzer()
analyzer.opening_book = analyzer.tablebases = None
analyzer._san_cache, analyzer._max_san_cache = OrderedDict(), 256
analyzer._get_analysis = lambda board, depth, multipv, time_limit: fake_engine_lines(board, depth, multipv, time_limit)
gui = SimpleNamespace(analyzer=analyzer, _get_game_phase=lambda p, t: ChessGUI._get_game_phase(None, p, t))

moves = opera_moves()
board, plies, last_class, previous = chess.Board(), [], None, None
start = time.perf_counter()
for i, move in enumerate(moves[:PLIES]):
    ply = ChessGUI._build_ply_analysis(gui, board.copy(), move, i, len(moves), last_class, OPENING, previous, fast_pass=True)
    plies.append(asdict(ply))
    last_class, previous = ply.classification, move
    board.push(move)
    print(f"ply {i + 1}: {ply.move_san} {ply.classification} ({time.perf_counter() - start:.1f} s)")
GOLDEN_PLIES_PATH.write_text(json.dumps(plies, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
print(f"wrote {GOLDEN_PLIES_PATH}")
