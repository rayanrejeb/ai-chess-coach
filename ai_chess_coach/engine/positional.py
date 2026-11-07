# Adapted from Chess Analyzer by yusufelgen07 (MIT License).
# See THIRD_PARTY_NOTICES.md.
import chess
import chess.engine
import chess.pgn
import chess.polyglot
import chess.svg
import chess.syzygy
import logging
from typing import Dict, List, Tuple, Any

from ai_chess_coach.config import AnalysisConfig
from ai_chess_coach.models import PositionalFeatures

log = logging.getLogger(__name__)


class PositionalAnalyzer:
    """A streamlined positional analyzer focusing on core, high-impact features."""
    def __init__(self, config: AnalysisConfig):
        self.config = config

    def analyze_position(self, board: chess.Board) -> PositionalFeatures:
        features = PositionalFeatures()
        features.material_balance = self._calculate_material_balance(board)
        features.king_safety = self._analyze_king_safety(board)
        features.pawn_structure = self._analyze_pawn_structure(board)
        features.rook_activity = self._analyze_rook_activity(board)
        features.knight_outposts = self._analyze_knight_outposts(board)
        features.bad_bishops = self._analyze_bad_bishops(board)
        features.has_bishop_pair = self._analyze_bishop_pair(board)
        
        advantages = self._identify_advantages(board)
        if features.has_bishop_pair.get("white"):
            advantages.append("White has the bishop pair.")
        if features.has_bishop_pair.get("black"):
            advantages.append("Black has the bishop pair.")
        features.advantages = advantages
        
        return features

    def _calculate_material_balance(self, board: chess.Board) -> float:
        piece_values = {chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.2, chess.ROOK: 5.0, chess.QUEEN: 9.0}
        white_material = sum(piece_values.get(p.piece_type, 0) for p in board.piece_map().values() if p.color == chess.WHITE)
        black_material = sum(piece_values.get(p.piece_type, 0) for p in board.piece_map().values() if p.color == chess.BLACK)
        return white_material - black_material

    def _analyze_king_safety(self, board: chess.Board) -> Dict[str, Any]:
        safety = {}
        for color in [chess.WHITE, chess.BLACK]:
            color_key = "white" if color == chess.WHITE else "black"
            king_square = board.king(color)
            if king_square is None:
                safety[color_key] = {"status": "missing", "attackers": 0, "pawn_shield_holes": 0}
                continue

            attackers = board.attackers(not color, king_square)
            pawn_shield_holes = self._count_pawn_shield_holes(board, king_square, color)

            danger_score = len(attackers) * 2 + pawn_shield_holes
            status = "safe"
            if danger_score > 3:
                status = "dangerous"
            elif danger_score > 1:
                status = "exposed"

            safety[color_key] = {
                "status": status,
                "attackers": len(attackers),
                "pawn_shield_holes": pawn_shield_holes
            }
        return safety

    def _count_pawn_shield_holes(self, board: chess.Board, king_square: int, color: chess.Color) -> int:
        holes = 0
        king_file, king_rank = chess.square_file(king_square), chess.square_rank(king_square)
        shield_files = range(max(0, king_file - 1), min(8, king_file + 2))
        shield_rank = king_rank + (1 if color == chess.WHITE else -1)

        if not (0 <= shield_rank <= 7): return 3

        for file in shield_files:
            pawn = board.piece_at(chess.square(file, shield_rank))
            if not pawn or pawn.piece_type != chess.PAWN or pawn.color != color:
                holes += 1
        return holes

    def _analyze_pawn_structure(self, board: chess.Board) -> Dict[str, List[str]]:
        structure = {"isolated_pawns": [], "doubled_pawns": [], "passed_pawns": [], "backward_pawns": [], "pawn_islands": []}
        for color in [chess.WHITE, chess.BLACK]:
            color_name = "White" if color == chess.WHITE else "Black"
            for pawn_square in board.pieces(chess.PAWN, color):
                sq_name = chess.square_name(pawn_square)
                if self._is_isolated_pawn(board, pawn_square, color):
                    structure["isolated_pawns"].append(f"{color_name}'s pawn on {sq_name}")
                if self._is_doubled_pawn(board, pawn_square, color):
                    structure["doubled_pawns"].append(f"{color_name}'s pawn on {sq_name}")
                if self._is_passed_pawn(board, pawn_square, color):
                    structure["passed_pawns"].append(f"{color_name}'s pawn on {sq_name}")
                if self._is_backward_pawn(board, pawn_square, color):
                    structure["backward_pawns"].append(f"{color_name}'s pawn on {sq_name}")
            
            islands = self._count_pawn_islands(board, color)
            structure["pawn_islands"].append(f"{color_name} has {islands} pawn island(s).")
        return structure

    def _is_isolated_pawn(self, board: chess.Board, square: int, color: chess.Color) -> bool:
        file = chess.square_file(square)
        adjacent_files = [f for f in [file - 1, file + 1] if 0 <= f <= 7]
        return not any(board.pieces(chess.PAWN, color) & chess.BB_FILES[adj_file] for adj_file in adjacent_files)

    def _is_doubled_pawn(self, board: chess.Board, square: int, color: chess.Color) -> bool:
        file = chess.square_file(square)
        return len(list(board.pieces(chess.PAWN, color) & chess.BB_FILES[file])) > 1

    def _is_passed_pawn(self, board: chess.Board, square: int, color: chess.Color) -> bool:
        file, rank = chess.square_file(square), chess.square_rank(square)
        files_to_check = [f for f in [file - 1, file, file + 1] if 0 <= f <= 7]
        ranks_to_check = range(rank + 1, 8) if color == chess.WHITE else range(0, rank)
        for check_file in files_to_check:
            for check_rank in ranks_to_check:
                piece = board.piece_at(chess.square(check_file, check_rank))
                if piece and piece.piece_type == chess.PAWN and piece.color != color:
                    return False
        return True

    def _is_backward_pawn(self, board: chess.Board, square: int, color: chess.Color) -> bool:
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        
        # No friendly pawns on adjacent files on same or previous ranks
        for adj_file in [f for f in [file - 1, file + 1] if 0 <= f <= 7]:
            # Check ranks BEHIND the pawn for support
            ranks_to_check = range(0, rank) if color == chess.WHITE else range(rank + 1, 8)
            for adj_rank in ranks_to_check:
                 if board.piece_at(chess.square(adj_file, adj_rank)) == chess.Piece(chess.PAWN, color):
                     return False # Supported by another pawn, so not backward

        # Square in front is attacked by enemy pawn, preventing advance
        forward_rank = rank + 1 if color == chess.WHITE else rank - 1
        if not (0 <= forward_rank <= 7): return False
        forward_square = chess.square(file, forward_rank)

        attackers = board.attackers(not color, forward_square)
        for attacker_sq in attackers:
            piece = board.piece_at(attacker_sq)
            if piece and piece.piece_type == chess.PAWN:
                return True # Blocked by enemy pawn, so it is backward
        return False

    def _count_pawn_islands(self, board: chess.Board, color: chess.Color) -> int:
        islands = 0
        pawn_files = {chess.square_file(sq) for sq in board.pieces(chess.PAWN, color)}
        if not pawn_files:
            return 0
        
        last_file = -2
        for file in sorted(list(pawn_files)):
            if file > last_file + 1:
                islands += 1
            last_file = file
        return islands

    def _analyze_rook_activity(self, board: chess.Board) -> Dict[str, List[str]]:
        rook_info = {"white": [], "black": []}
        for color in [chess.WHITE, chess.BLACK]:
            color_name = "White" if color == chess.WHITE else "Black"
            key = "white" if color == chess.WHITE else "black"
            seventh_rank = 6 if color == chess.WHITE else 1
            
            for rook_square in board.pieces(chess.ROOK, color):
                file = chess.square_file(rook_square)
                rank = chess.square_rank(rook_square)
                sq_name = chess.square_name(rook_square)

                if rank == seventh_rank:
                    rook_info[key].append(f"{color_name}'s Rook on {sq_name} is on the 7th rank, a major threat.")
                    continue

                is_open = not (board.pieces(chess.PAWN, chess.WHITE) & chess.BB_FILES[file]) and not (board.pieces(chess.PAWN, chess.BLACK) & chess.BB_FILES[file])
                is_semi_open = not (board.pieces(chess.PAWN, color) & chess.BB_FILES[file])

                if is_open:
                    rook_info[key].append(f"{color_name}'s Rook on {sq_name} controls an open file.")
                elif is_semi_open:
                    rook_info[key].append(f"{color_name}'s Rook on {sq_name} controls a semi-open file.")
        return rook_info

    def _analyze_bishop_pair(self, board: chess.Board) -> Dict[str, bool]:
        white_bishops = len(board.pieces(chess.BISHOP, chess.WHITE))
        black_bishops = len(board.pieces(chess.BISHOP, chess.BLACK))
        return {
            "white": white_bishops >= 2,
            "black": black_bishops >= 2
        }

    def _analyze_bad_bishops(self, board: chess.Board) -> List[str]:
        bad_bishops = []
        for color in [chess.WHITE, chess.BLACK]:
            color_name = "White" if color == chess.WHITE else "Black"
            own_pawns = list(board.pieces(chess.PAWN, color))
            if not own_pawns: continue

            for bishop_square in board.pieces(chess.BISHOP, color):
                is_light_square_bishop = (chess.square_file(bishop_square) + chess.square_rank(bishop_square)) % 2 != 0
                
                pawns_on_same_color = 0
                for pawn_square in own_pawns:
                    is_light_square_pawn = (chess.square_file(pawn_square) + chess.square_rank(pawn_square)) % 2 != 0
                    if is_light_square_pawn == is_light_square_bishop:
                        pawns_on_same_color += 1
                
                if pawns_on_same_color / len(own_pawns) > self.config.BAD_BISHOP_PAWN_RATIO:
                    bishop_type = "light-square" if is_light_square_bishop else "dark-square"
                    bad_bishops.append(f"{color_name}'s {bishop_type} bishop on {chess.square_name(bishop_square)} is a bad bishop.")
        return bad_bishops

    def _analyze_knight_outposts(self, board: chess.Board) -> List[str]:
        outposts = []
        for color in [chess.WHITE, chess.BLACK]:
            color_name = "White" if color == chess.WHITE else "Black"
            # Ranks are 0-7. White outposts are on ranks 5,6,7 (indices 4,5,6).
            # Black outposts are on ranks 4,3,2 (indices 3,2,1).
            outpost_ranks = [4, 5, 6] if color == chess.WHITE else [3, 2, 1]
            
            for knight_square in board.pieces(chess.KNIGHT, color):
                rank = chess.square_rank(knight_square)
                file = chess.square_file(knight_square)
                if rank not in outpost_ranks:
                    continue

                # Must be supported by a friendly pawn
                is_supported = False
                for supporter_sq in board.attackers(color, knight_square):
                    piece = board.piece_at(supporter_sq)
                    if piece and piece.piece_type == chess.PAWN:
                        is_supported = True
                        break
                if not is_supported:
                    continue

                # Cannot be attacked by an enemy pawn
                can_be_attacked_by_pawn = False
                if color == chess.WHITE:
                    pawn_rank = rank + 1
                    for pawn_file in [file - 1, file + 1]:
                        if 0 <= pawn_file <= 7 and pawn_rank < 8:
                            pawn_square = chess.square(pawn_file, pawn_rank)
                            piece = board.piece_at(pawn_square)
                            if piece and piece.piece_type == chess.PAWN and piece.color == chess.BLACK:
                                can_be_attacked_by_pawn = True
                                break
                else: # Black Knight
                    pawn_rank = rank - 1
                    for pawn_file in [file - 1, file + 1]:
                        if 0 <= pawn_file <= 7 and pawn_rank >= 0:
                            pawn_square = chess.square(pawn_file, pawn_rank)
                            piece = board.piece_at(pawn_square)
                            if piece and piece.piece_type == chess.PAWN and piece.color == chess.WHITE:
                                can_be_attacked_by_pawn = True
                                break
                
                if not can_be_attacked_by_pawn:
                    outposts.append(f"{color_name}'s knight on {chess.square_name(knight_square)} is on a strong outpost.")
        return outposts

    def _identify_advantages(self, board: chess.Board) -> List[str]:
        advantages = []
        white_space, black_space = self._calculate_space(board)
        if white_space > black_space * self.config.SPACE_ADVANTAGE_RATIO:
            advantages.append("White has a space advantage.")
        if black_space > white_space * self.config.SPACE_ADVANTAGE_RATIO:
            advantages.append("Black has a space advantage.")
        return advantages

    def _calculate_space(self, board: chess.Board) -> Tuple[int, int]:
        white_space = 0
        black_space = 0
        for rank in range(3, 6):
            for file in range(8):
                sq = chess.square(file, rank)
                if not board.is_attacked_by(chess.BLACK, sq):
                    white_space += 1
                if not board.is_attacked_by(chess.WHITE, sq):
                    black_space += 1
        return white_space, black_space

