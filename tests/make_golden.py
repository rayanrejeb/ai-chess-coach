"""Write tests/golden.json from the original single-file app (chess_analyzer.py).

Run once, before the refactor:  python tests/make_golden.py
The single file was removed afterwards; to regenerate, check out commit 3438f55.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess_analyzer as old  # noqa: E402

from golden_cases import GOLDEN_PATH, compute  # noqa: E402

# The old ChessAnalyzer without an engine: classify_move only needs these attributes.
classifier = old.ChessAnalyzer.__new__(old.ChessAnalyzer)
classifier.config, classifier.opening_book, classifier.tablebases = old.AnalysisConfig(), None, None
data = compute(old.PositionalAnalyzer, old.TacticalAnalyzer, classifier.classify_move, classifier._cp_to_expected_points, old.AnalysisConfig)
GOLDEN_PATH.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"wrote {GOLDEN_PATH} ({len(data['positions'])} positions, engine: {data.get('engine_id')})")
