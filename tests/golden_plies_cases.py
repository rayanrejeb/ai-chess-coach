"""Shared settings for tests/golden_plies.json (see make_golden_plies.py)."""

from pathlib import Path

import chess.pgn

GOLDEN_PLIES_PATH = Path(__file__).with_name("golden_plies.json")
PGN = Path(__file__).resolve().parent.parent / "sample_data" / "opera_game_1858.pgn"
OPENING = "Philidor Defense"
PLIES = 12


def opera_moves():
    with open(PGN, encoding="utf-8") as f:
        return list(chess.pgn.read_game(f).mainline_moves())
