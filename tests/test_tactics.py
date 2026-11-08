import chess

from ai_chess_coach.tactics.analyzer import TacticalAnalyzer
from golden_plies_cases import opera_moves


def test_material_balance_matches_a_piece_by_piece_count():
    analyzer, board = TacticalAnalyzer(), chess.Board()
    for move in opera_moves():  # includes captures by both sides and a final mate
        board.push(move)
        by_piece = sum(TacticalAnalyzer.PIECE_VALUES[p.piece_type] * (1 if p.color == chess.WHITE else -1)
                       for p in board.piece_map().values())
        assert analyzer.get_material_balance(board) == by_piece
