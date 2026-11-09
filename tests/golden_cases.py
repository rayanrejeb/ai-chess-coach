"""Reference outputs on 3 sample positions, used to prove refactors don't change behavior.

`compute()` takes the analyzers and functions as arguments so the same code runs against the old
single-file app (to create golden.json) and against the new package (to check it).
"""

import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path

import chess
import chess.engine
import chess.pgn

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = Path(__file__).with_name("golden.json")
ENGINE_DEPTH = 12

# (sample game, plies played before the position, expected SAN of the game move)
POSITIONS = [
    ("opera_game_1858.pgn", 24, "Rxd7"),
    ("immortal_game_1851.pgn", 32, "Nd5"),
    ("evergreen_game_1852.pgn", 32, "Nf6+"),
]

# Synthetic engine results for classify_move: (cp_before, cp_after, move_is_best, n_lines, second_line_cp, opponent_class)
# All scores are centipawns from the mover's point of view; "M3" / "M-2" mean mate in N.
CLASSIFY_CASES = [
    (30, 25, True, 1, None, None),
    (-120, 160, True, 1, None, None),
    (40, 35, True, 3, -200, None),
    (20, 10, False, 1, None, None),
    (20, -30, False, 1, None, None),
    (20, -120, False, 1, None, None),
    (20, -250, False, 1, None, None),
    (20, -700, False, 1, None, None),
    (300, 250, False, 1, None, "Blunder"),
    ("M3", 400, False, 1, None, None),
    (50, "M-2", False, 1, None, None),
    (-50, "M2", False, 1, None, None),
]


def stockfish_path():
    path = os.environ.get("STOCKFISH_PATH")  # .env.example's placeholder path does not exist: skip, don't fail
    return path if path and os.path.isfile(path) else shutil.which("stockfish")


def load_position(pgn_name, plies, expected_san):
    with open(ROOT / "sample_data" / pgn_name, encoding="utf-8") as f:
        game = chess.pgn.read_game(f)
    moves = list(game.mainline_moves())
    board = game.board()
    for m in moves[:plies]:
        board.push(m)
    move = moves[plies]
    assert board.san(move) == expected_san, (pgn_name, board.san(move))
    return board, move


def _score(value, turn):
    if isinstance(value, str):
        n = int(value[1:])
        return chess.engine.PovScore(chess.engine.Mate(n), turn)
    return chess.engine.PovScore(chess.engine.Cp(value), turn)


def _normalize(obj):
    return json.loads(json.dumps(obj, sort_keys=True, default=str))


def compute(PositionalAnalyzer, TacticalAnalyzer, classify_move, win_probability, AnalysisConfig, with_engine=True):
    """classify_move(board, move, lines_before, info_after, opponent_class) -> (class, loss, cp_loss);
    win_probability(cp) -> 0..1 (stored as "expected_points")."""
    config = AnalysisConfig()
    positional = PositionalAnalyzer(config)

    out = {"positions": [], "expected_points": {}}
    for cp in (-1000, -300, -100, 0, 100, 300, 1000):
        out["expected_points"][str(cp)] = round(win_probability(cp), 6)

    engine = None
    if with_engine:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path())
        engine.configure({"Threads": 1, "Hash": 16})
        out["engine_id"] = engine.id.get("name")

    try:
        for pgn_name, plies, san in POSITIONS:
            board, move = load_position(pgn_name, plies, san)
            after = board.copy()
            after.push(move)
            tactics = TacticalAnalyzer()
            entry = {
                "name": pgn_name,
                "fen": board.fen(),
                "move": move.uci(),
                "positional_before": asdict(positional.analyze_position(board.copy())),
                "positional_after": asdict(positional.analyze_position(after.copy())),
                "motifs": tactics.find_tactical_motifs(board.copy(), after.copy(), move),
                "see": {m.uci(): TacticalAnalyzer().see(board.copy(), m) for m in sorted(board.legal_moves, key=lambda m: m.uci()) if board.is_capture(m)},
                "classify": [],
            }

            other = next(m for m in sorted(board.legal_moves, key=lambda m: m.uci()) if m != move)
            for cp_before, cp_after, is_best, n_lines, second_cp, opp in CLASSIFY_CASES:
                best = move if is_best else other
                lines = [{"score": _score(cp_before, board.turn), "pv": [best]}]
                for _ in range(n_lines - 1):
                    lines.append({"score": _score(second_cp, board.turn), "pv": [other]})
                after_info = {"score": _score(cp_after, board.turn), "pv": []}
                cls, points_loss, cp_loss = classify_move(board.copy(), move, lines, after_info, opp)
                entry["classify"].append([cls, round(points_loss, 6), cp_loss])

            if engine:
                for label, b in (("before", board), ("after", after)):
                    engine.configure({"Clear Hash": None})
                    info = engine.analyse(b, chess.engine.Limit(depth=ENGINE_DEPTH))
                    entry[f"engine_{label}"] = {
                        "best": info["pv"][0].uci(),
                        "score_white": info["score"].white().score(mate_score=10000),
                    }
            out["positions"].append(entry)
    finally:
        if engine:
            engine.quit()
    return _normalize(out)
