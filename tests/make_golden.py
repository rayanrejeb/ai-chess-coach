"""Write tests/golden.json from the original single-file app (chess_analyzer.py).

Run once, before the refactor:  python tests/make_golden.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess_analyzer as old  # noqa: E402

from golden_cases import GOLDEN_PATH, compute  # noqa: E402

data = compute(old.PositionalAnalyzer, old.TacticalAnalyzer, old.ChessAnalyzer, old.AnalysisConfig)
GOLDEN_PATH.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"wrote {GOLDEN_PATH} ({len(data['positions'])} positions, engine: {data.get('engine_id')})")
