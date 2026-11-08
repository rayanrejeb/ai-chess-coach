# Adapted from Chess Analyzer by yusufelgen07 (MIT License).
# See THIRD_PARTY_NOTICES.md.
import chess
import chess.engine
import chess.pgn
import chess.polyglot
import chess.svg
import chess.syzygy
import logging
from typing import Dict, List, Optional, Tuple, Union, Set

log = logging.getLogger(__name__)


class TacticalAnalyzer:
    """
    Tactical analyzer focused on detecting actual tactical patterns and potential profitable opportunities.
    Includes original methods plus new detection for discovered attacks, overloading, interference, and more.
    """

    # Standard piece values in centipawns
    PIECE_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 300,
        chess.BISHOP: 300,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }

    def __init__(self):
        self.previous_motifs: Set[Tuple[str, str]] = set()

    def reset_state(self):
        """Resets stateful properties for a new game analysis."""
        self.previous_motifs = set()

    # --- NEW QUIESCENCE SEARCH METHODS ---
    def get_material_balance(self, board: chess.Board) -> int:
        """Calculates the material balance of the board from White's perspective."""
        # Counted from the bitboards: the quiescence search calls this very often, and building
        # piece_map() twice per call was the largest single cost of the tactical analysis.
        return sum(
            value * (chess.popcount(board.pieces_mask(piece_type, chess.WHITE)) - chess.popcount(board.pieces_mask(piece_type, chess.BLACK)))
            for piece_type, value in self.PIECE_VALUES.items()
        )

    def quiescence_search(self, board: chess.Board, alpha: float, beta: float, depth: int, max_depth: int = 10) -> Tuple[int, int, List[chess.Move]]:
        """
        Performs a quiescence search to find the tactical value of a position.
        Returns a tuple of (final_score, max_advantage_seen, principal_variation).
        Both values are from the perspective of the current player.
        """
        pv = []
        if depth >= max_depth:
            eval_score = self.get_material_balance(board) if board.turn == chess.WHITE else -self.get_material_balance(board)
            return eval_score, eval_score, []

        stand_pat_score = self.get_material_balance(board) if board.turn == chess.WHITE else -self.get_material_balance(board)
        max_advantage = stand_pat_score

        if stand_pat_score >= beta:
            return beta, max_advantage, []
        if alpha < stand_pat_score:
            alpha = stand_pat_score

        capture_moves = [move for move in board.legal_moves if board.is_capture(move)]

        if not capture_moves:
            if board.is_check() and board.is_checkmate():
                return -self.PIECE_VALUES[chess.KING], max_advantage, []
            return stand_pat_score, max_advantage, []

        capture_moves.sort(key=lambda move: self.PIECE_VALUES.get(board.piece_at(move.to_square).piece_type, 0), reverse=True)

        for move in capture_moves:
            board.push(move)
            score, opp_max_advantage, sub_pv = self.quiescence_search(board, -beta, -alpha, depth + 1, max_depth)
            score = -score
            board.pop()

            max_advantage = max(max_advantage, score)

            if score >= beta:
                return beta, max_advantage, [move] + sub_pv
            if score > alpha:
                alpha = score
                pv = [move] + sub_pv

        return alpha, max_advantage, pv

    def evaluate_move_with_quiescence(self, board: chess.Board, move: chess.Move) -> Tuple[int, int, List[chess.Move]]:
        """
        Evaluates the position after a move by performing a quiescence search.
        Returns a tuple of (final_score, max_advantage, sequence) from the perspective
        of the player whose turn it was BEFORE the move.
        """
        if not board.is_capture(move):
            current_eval = self.get_material_balance(board) if board.turn == chess.WHITE else -self.get_material_balance(board)
            return current_eval, current_eval, []

        try:
            board.push(move)
            final_score, max_adv, pv = self.quiescence_search(board, -float('inf'), float('inf'), 0)
            board.pop()
            return -final_score, -max_adv, [move] + pv
        except Exception as e:
            if board.move_stack and board.peek() == move:
                board.pop()
            log.warning(f"Error during quiescence evaluation for move {move.uci()} on FEN {board.fen()}: {e}")
            return 0, 0, []

    # ========================================================================================
    # STATIC EXCHANGE EVALUATION (SEE) - NOW POWERED BY QUIESCENCE SEARCH
    # ========================================================================================

    def see(self, board: chess.Board, move: chess.Move, with_sequence: bool = False) -> Union[int, Tuple[int, List[chess.Move]]]:
        """
        Calculates the tactical gain of a move using a quiescence search, which is more
        robust than traditional SEE. It evaluates the outcome of the likely capture sequence
        that follows the move.

        Args:
            board: The chess board before the move.
            move: The move to evaluate.
            with_sequence: If True, returns the principal variation (sequence of moves) along with the score.

        Returns:
            If with_sequence is False, returns the material gain in centipawns.
            If with_sequence is True, returns a tuple of (gain_in_centipawns, sequence_of_moves).
        """
        # A non-capture move has a SEE value of 0 by definition.
        if not board.is_capture(move):
            return (0, []) if with_sequence else 0

        # Ensure the move is legal before proceeding.
        if move not in board.legal_moves:
            return (0, []) if with_sequence else 0

        # Get the material evaluation from the current player's perspective before the move.
        initial_score = self.get_material_balance(board) if board.turn == chess.WHITE else -self.get_material_balance(board)

        # evaluate_move_with_quiescence returns the final score from the perspective
        # of the player making the move, after the tactical sequence.
        final_score, _, sequence = self.evaluate_move_with_quiescence(board, move)

        # The gain is the difference between the evaluation after the tactical sequence and the
        # evaluation before the move was made.
        gain = final_score - initial_score

        if with_sequence:
            return gain, sequence
        else:
            return gain





    # ========================================================================================
    # TACTICAL MOTIF DETECTION - UPDATED MAIN FUNCTION
    # ========================================================================================

    def find_tactical_motifs(self, board_before: chess.Board, board_after: chess.Board, 
                           move: chess.Move, ply_index: Optional[int] = None, previous_move: Optional[chess.Move] = None, find_potential: bool = True) -> List[Dict[str, str]]:
        if not self._validate_inputs(board_before, board_after, move):
            return [{"tactic": "Error", "details": "Invalid input parameters"}]
        motifs: List[Dict[str, str]] = []
        moving_side = board_before.turn
        last_move_to_square = move.to_square
        test_board = board_before.copy()
        if move not in test_board.legal_moves:
            return [{"tactic": "Error", "details": "Illegal move"}]
        test_board.push(move)
        if test_board.fen() != board_after.fen():
            return [{"tactic": "Error", "details": "Board states don't match move"}]
        prev_move_to_sq = previous_move.to_square if previous_move else None
        original_move_san = board_before.san(move)
        motifs.extend(self._detect_potential_profitable_trades(board_after, not moving_side, last_move_to_square)) #checked
        motifs.extend(self._find_removal_of_guard(board_before, move)) #checked
        if find_potential:
            motifs.extend(self._detect_potential_removal_of_guard_for_both_sides(board_after, moving_side, original_move_san)) #checked
        motifs.extend(self._find_forks_on_board(board_after, chess.WHITE))
        motifs.extend(self._find_forks_on_board(board_after, chess.BLACK))
        if find_potential:
            motifs.extend(self._detect_potential_forks_for_both_sides(board_after, moving_side, original_move_san)) # not checked yet
        motifs.extend(self._find_pins_on_board(board_after, chess.WHITE)) #checked
        motifs.extend(self._find_pins_on_board(board_after, chess.BLACK))
        if find_potential:
            motifs.extend(self._detect_potential_pins_for_both_sides(board_after, moving_side, original_move_san))
        motifs.extend(self._find_skewers_on_board(board_after, chess.WHITE))
        motifs.extend(self._find_skewers_on_board(board_after, chess.BLACK))
        if find_potential:
            motifs.extend(self._detect_potential_skewers_for_both_sides(board_after, moving_side, original_move_san)) # not checked yet
        motifs.extend(self._detect_hanging_pieces(board_after, moving_side, last_move_to_square, board_before.is_capture(move))) #checked
        motifs.extend(self._detect_hanging_pieces(board_after, not moving_side, -1, False))
        motifs.extend(self._detect_discovered_attacks(board_before, move)) # checked
        if find_potential:
            motifs.extend(self._detect_potential_discovered_attacks_for_both_sides(board_after, moving_side, original_move_san)) #checked
        motifs.extend(self._detect_overloaded_pieces(board_after, chess.WHITE))
        motifs.extend(self._detect_overloaded_pieces(board_after, chess.BLACK))
        motifs.extend(self._find_trapped_pieces_on_board(board_after, not moving_side)) #checked
        if find_potential:
            motifs.extend(self._detect_potential_trapped_pieces_for_both_sides(board_after, moving_side, original_move_san))
        motifs.extend(self._detect_clearance_sacrifices(board_before, move)) # not checked yet
        motifs.extend(self._detect_decoy_and_attraction(board_after, moving_side)) # not checked yet
        motifs.extend(self._detect_battery_threats(board_after, chess.WHITE))
        motifs.extend(self._detect_battery_threats(board_after, chess.BLACK))
        
        current_motifs = self._clean_motifs(motifs)

        def get_key(m):
            # Use the stable canonical_key if it exists, otherwise fall back to the old method.
            return m.get('canonical_key', (m.get('tactic', ''), m.get('details', '')))

        current_motif_keys = {get_key(m) for m in current_motifs}
        new_motifs = [m for m in current_motifs if get_key(m) not in self.previous_motifs]
        
        # THE FIX: Only update state for actual game moves, not hypothetical engine lines.
        if ply_index is not None:
            self.previous_motifs = current_motif_keys

        # Clean up the internal-only key before returning the motifs
        for m in new_motifs:
            m.pop('canonical_key', None)

        if ply_index is not None and new_motifs and new_motifs[0].get("tactic") != "None":
            motifs_for_display = [{k: v for k, v in m.items() if k != 'value'} for m in new_motifs]
            log.debug(f"Ply {ply_index + 1} ({move.uci()}): Detected New Motifs -> {motifs_for_display}")
        
        return new_motifs if new_motifs else [{"tactic": "None", "details": "No tactical opportunities detected"}]

    # ========================================================================================
    # DETECTION METHODS (ORIGINAL AND UPDATED)
    # ========================================================================================

    def _validate_inputs(self, board_before: chess.Board, board_after: chess.Board, move: chess.Move) -> bool:
        try:
            return (isinstance(board_before, chess.Board) and 
                   isinstance(board_after, chess.Board) and 
                   isinstance(move, chess.Move) and
                   board_before.is_valid() and 
                   board_after.is_valid())
        except:
            return False

    def _is_move_sound(self, board: chess.Board, move: chess.Move) -> Tuple[bool, Optional[chess.Move]]:
        """
        Checks if a move is tactically sound, ignoring it if it leads to a bad trade.
        - For captures, it's sound if the SEE is non-negative (>= 0).
        - For quiet moves, it's sound if it doesn't allow any profitable captures by the opponent.
        """
        if not move in board.legal_moves:
            return False, None

        if board.is_capture(move):
            # For captures, the move is sound if it's at least an even trade for us.
            return self.see(board, move) >= 0, None
        else:  # For quiet moves
            temp_board = board.copy()
            temp_board.push(move)
            # Check if the opponent has any profitable captures now.
            # If they do, our move was unsound.
            for opponent_move in temp_board.legal_moves:
                if temp_board.is_capture(opponent_move):
                    if self.see(temp_board, opponent_move) >= 0:
                        # Opponent has a profitable reply, so our move is not sound.
                        return False, opponent_move
            # No profitable replies for the opponent were found, so the move is sound.
            return True, None

    def _detect_potential_profitable_trades(self, board: chess.Board, side: chess.Color, prev_move_to_sq: Optional[int] = None) -> List[Dict[str, str]]:
        """Detects if the move just made created opportunities for the opponent to initiate a profitable trade, including the full move sequence."""
        motifs = []
        
        if board.turn != side:
            return []

        initial_value = self.get_material_balance(board) if side == chess.WHITE else -self.get_material_balance(board)

        for move in board.legal_moves:
            if board.is_capture(move):
                if move.to_square == prev_move_to_sq:
                    continue
                
                final_score, max_swing, sequence = self.evaluate_move_with_quiescence(board, move)
                gain = final_score - initial_value
                
                if gain > 0 and sequence:
                    captured_piece = board.piece_at(move.to_square)
                    moving_piece = board.piece_at(move.from_square)
                    
                    if captured_piece and moving_piece:
                        sequence_san_parts = []
                        temp_board_for_san = board.copy()
                        try:
                            for seq_move in sequence:
                                if seq_move in temp_board_for_san.legal_moves:
                                    sequence_san_parts.append(temp_board_for_san.san(seq_move))
                                    temp_board_for_san.push(seq_move)
                                else:
                                    break
                        except Exception:
                            sequence_san_parts = [m.uci() for m in sequence]

                        sequence_str = f" The full sequence is: {' '.join(sequence_san_parts)}." if sequence_san_parts else ""
                        details = (f"The move made created a profitable trade potential for the opponent. "
                                   f"The move {board.san(move)} starts a favorable tactical sequence, improving the evaluation by {gain/100.0:+.2f}.{sequence_str}")
                        
                        motifs.append({
                            "tactic": "Profitable Trade",
                            "details": details,
                            "value": gain,
                            "sequence_uci": [m.uci() for m in sequence]
                        })
        
        motifs.sort(key=lambda x: x.get("value", 0), reverse=True)
        return motifs[:2]
    def _find_removal_of_guard(self, board_before: chess.Board, move: chess.Move) -> List[Dict[str, str]]:
        motifs = []
        moving_piece = board_before.piece_at(move.from_square)
        if not moving_piece or moving_piece.piece_type == chess.KING: return motifs

        board_after = board_before.copy()
        board_after.push(move)

        # Case 1: Moving your own piece, which was guarding a friendly piece.
        for defended_sq, defended_piece in board_before.piece_map().items():
            if defended_piece.color == moving_piece.color and move.from_square in board_before.attackers(moving_piece.color, defended_sq):
                if board_after.is_attacked_by(not moving_piece.color, defended_sq):
                    temp_board_see = board_after.copy()
                    temp_board_see.turn = not moving_piece.color # Opponent's turn to capture
                    
                    # Iterate through all possible opponent captures on the now-vulnerable square
                    for attacker_sq in temp_board_see.attackers(not moving_piece.color, defended_sq):
                        capture_move = chess.Move(attacker_sq, defended_sq)
                        if capture_move in temp_board_see.legal_moves:
                            see_value = self.see(temp_board_see, capture_move)
                            if see_value > 0:
                                details = (f"The move {board_before.san(move)} removes a defender. "
                                           f"The {chess.piece_name(moving_piece.piece_type)} moved, leaving the friendly "
                                           f"{chess.piece_name(defended_piece.piece_type)} on {chess.square_name(defended_sq)} vulnerable to a capture by the {chess.piece_name(temp_board_see.piece_at(attacker_sq).piece_type)}.")
                                motifs.append({"tactic": "Removal of the Guard (Self)", "details": details, "value": see_value, "squares": [move.from_square, defended_sq]})
                                # Found a profitable capture, no need to check other attackers for this defended piece.
                                break 

        # Case 2: Capturing an opponent's piece that was guarding another enemy piece.
        if board_before.is_capture(move):
            captured_piece = board_before.piece_at(move.to_square)
            if not captured_piece and board_before.is_en_passant(move):
                captured_piece = chess.Piece(chess.PAWN, not moving_piece.color)

            if captured_piece:
                for defended_square in board_before.attacks(move.to_square):
                    defended_piece_target = board_before.piece_at(defended_square)
                    if defended_piece_target and defended_piece_target.color == captured_piece.color:
                        if board_after.is_attacked_by(moving_piece.color, defended_square):
                            temp_board_see = board_after.copy()
                            temp_board_see.turn = moving_piece.color # Our turn to capture
                            
                            # Iterate through all our possible captures on the now-vulnerable square
                            for attacker_sq in temp_board_see.attackers(moving_piece.color, defended_square):
                                capture_move = chess.Move(attacker_sq, defended_square)
                                if capture_move in temp_board_see.legal_moves:
                                    see_value = self.see(temp_board_see, capture_move)
                                    if see_value > 0:
                                        details = (f"The capture {board_before.san(move)} removes a key enemy defender. "
                                                   f"This exposes the {chess.piece_name(defended_piece_target.piece_type)} on {chess.square_name(defended_square)} to a profitable attack.")
                                        motifs.append({"tactic": "Removal of the Guard (Capture)", "details": details, "value": see_value, "squares": [move.to_square, defended_square]})
                                        # Found a profitable capture, no need to check other attackers for this defended piece.
                                        break
        return motifs

    def _detect_potential_removal_of_guard(self, board: chess.Board, side: chess.Color, original_move_san: str, is_opportunity: bool) -> List[Dict[str, str]]:
        motifs = []

        work_board = board
        if board.turn != side:
            b = board.copy()
            if b.is_check():
                return motifs
            try:
                b.push(chess.Move.null())
            except ValueError:
                return motifs
            if b.turn != side:
                return motifs
            work_board = b

        for move in work_board.legal_moves:
            created_motifs = self._find_removal_of_guard(work_board, move)
            if created_motifs:
                if is_opportunity:
                    # VALIDITY CHECK: An opportunity is only valid if the opponent has no
                    # non-losing move on ply 2 that can disrupt it.
                    opportunity_is_disruptable = False
                    for opponent_move in board.legal_moves:
                        board_after_opponent_move = board.copy()
                        board_after_opponent_move.push(opponent_move)

                        tactic_is_broken = False
                        if move not in board_after_opponent_move.legal_moves:
                            tactic_is_broken = True
                        else:
                            # This tactic check needs the board *before* the tactical move.
                            if not self._find_removal_of_guard(board_after_opponent_move, move):
                                tactic_is_broken = True
                        
                        if not tactic_is_broken: continue

                        is_non_losing = False
                        if board.is_capture(opponent_move):
                            if self.see(board, opponent_move) >= 0: is_non_losing = True
                        else: # Quiet move
                            board_after_quiet_move = board.copy()
                            board_after_quiet_move.push(opponent_move)
                            has_profitable_reply = False
                            for my_reply in board_after_quiet_move.legal_moves:
                                if board_after_quiet_move.is_capture(my_reply) and self.see(board_after_quiet_move, my_reply) > 0:
                                    has_profitable_reply = True; break
                            if not has_profitable_reply: is_non_losing = True
                        
                        if is_non_losing:
                            opportunity_is_disruptable = True; break
                    
                    if opportunity_is_disruptable:
                        continue
                else: # is_opportunity=False (threat check)
                    is_sound, _ = self._is_move_sound(work_board, move)
                    if not is_sound:
                        continue

                motif_info = created_motifs[0]
                move_san = work_board.san(move)
                if is_opportunity:
                    details = (f"Your move {original_move_san} sets up a potential 'Removal of the Guard'. "
                               f"On your next turn, the move {move_san} would actualize it: {motif_info['details']}")
                else:
                    details = (f"A potential 'Removal of the Guard' is available with the move {move_san}. "
                               f"The resulting situation is: {motif_info['details']}")

                motifs.append({
                    "tactic": f"Potential {motif_info['tactic']}",
                    "details": details.strip(),
                    "value": motif_info.get('value', 0),
                    "creating_move_san": move_san,
                    "canonical_key": (f"Potential {motif_info['tactic']}", move_san, tuple(motif_info.get("squares", [])))
                })
        return motifs

    def _detect_potential_removal_of_guard_for_both_sides(self, board: chess.Board, moving_side: chess.Color, original_move_san: str) -> List[Dict[str, str]]:
        motifs = []
        motifs.extend(self._detect_potential_removal_of_guard(board, moving_side, original_move_san, is_opportunity=True))
        motifs.extend(self._detect_potential_removal_of_guard(board, not moving_side, original_move_san, is_opportunity=False))
        return motifs

    def _find_forks_on_board(self, board: chess.Board, side: chess.Color) -> List[Dict[str, str]]:
        motifs = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None and piece.color == side:
                attacked_squares = board.attacks(square)
                valuable_targets = []
                for target_square in attacked_squares:
                    target = board.piece_at(target_square)
                    if (target is not None and target.color != side and 
                        target.piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.KING]):
                        temp_board = board.copy(); temp_board.turn = side
                        capture_move = chess.Move(square, target_square)
                        if capture_move in temp_board.legal_moves:
                            see_value = self.see(temp_board, capture_move)
                            valuable_targets.append((target, target_square, see_value))
                
                if len(valuable_targets) >= 2:
                    is_forker_safe = True
                    if board.is_attacked_by(not side, square):
                        opponent_board = board.copy()
                        opponent_board.turn = not side
                        for attacker_square in opponent_board.attackers(not side, square):
                            capture_forker_move = chess.Move(attacker_square, square)
                            if capture_forker_move in opponent_board.legal_moves:
                                if self.see(opponent_board, capture_forker_move) >= 0:
                                    is_forker_safe = False
                                    break
                    if not is_forker_safe:
                        continue

                    # NEW LOGIC: Check for escaping checks by forked pieces
                    has_safe_escaping_check = False
                    forked_squares = {t[1] for t in valuable_targets}
                    opponent_board = board.copy()
                    opponent_board.turn = not side
                    for move in opponent_board.legal_moves:
                        if move.from_square in forked_squares and opponent_board.gives_check(move):
                            is_safe_check = True
                            board_after_escape = opponent_board.copy()
                            board_after_escape.push(move)
                            board_after_escape.turn = side
                            for our_reply_move in board_after_escape.legal_moves:
                                if our_reply_move.to_square == move.to_square:
                                    if self.see(board_after_escape, our_reply_move) >= 0:
                                        is_safe_check = False
                                        break
                            if is_safe_check:
                                has_safe_escaping_check = True
                                break
                    
                    if has_safe_escaping_check:
                        continue

                    valuable_targets.sort(key=lambda x: x[2])
                    if valuable_targets[0][2] > 0:
                        forking_piece_color = "White" if piece.color == chess.WHITE else "Black"
                        target_details = []
                        for t_piece, t_square, _ in valuable_targets[:2]:
                            t_color = "White" if t_piece.color == chess.WHITE else "Black"
                            target_details.append(f"the {t_color} {chess.piece_name(t_piece.piece_type)} on {chess.square_name(t_square)}")
                        details = f"The {forking_piece_color} {chess.piece_name(piece.piece_type)} on {chess.square_name(square)} forks {' and '.join(target_details)}."
                        motifs.append({"tactic": "Fork", "details": details, "value": valuable_targets[0][2], "squares": [square] + [t[1] for t in valuable_targets[:2]]})
        return motifs

    def _detect_potential_forks(self, board: chess.Board, side: chess.Color, original_move_san: str, is_opportunity: bool) -> List[Dict[str, str]]:
        motifs = []

        work_board = board
        if board.turn != side:
            b = board.copy()
            if b.is_check():
                return motifs
            try:
                b.push(chess.Move.null())
            except ValueError:
                return motifs
            if b.turn != side:
                return motifs
            work_board = b

        existing_forks = {f['details'] for f in self._find_forks_on_board(work_board, side)}

        for move in work_board.legal_moves:
            board_after_move = work_board.copy()
            board_after_move.push(move)
            created_forks = self._find_forks_on_board(board_after_move, side)

            if created_forks:
                if is_opportunity:
                    # VALIDITY CHECK: An opportunity is only valid if the opponent has no
                    # non-losing move on ply 2 that can disrupt it.
                    opportunity_is_disruptable = False
                    for opponent_move in board.legal_moves:
                        board_after_opponent_move = board.copy()
                        board_after_opponent_move.push(opponent_move)

                        tactic_is_broken = False
                        if move not in board_after_opponent_move.legal_moves:
                            tactic_is_broken = True
                        else:
                            board_after_our_move = board_after_opponent_move.copy()
                            board_after_our_move.push(move)
                            if not self._find_forks_on_board(board_after_our_move, side):
                                tactic_is_broken = True
                        
                        if not tactic_is_broken: continue

                        is_non_losing = False
                        if board.is_capture(opponent_move):
                            if self.see(board, opponent_move) >= 0: is_non_losing = True
                        else: # Quiet move
                            board_after_quiet_move = board.copy()
                            board_after_quiet_move.push(opponent_move)
                            has_profitable_reply = False
                            for my_reply in board_after_quiet_move.legal_moves:
                                if board_after_quiet_move.is_capture(my_reply) and self.see(board_after_quiet_move, my_reply) > 0:
                                    has_profitable_reply = True; break
                            if not has_profitable_reply: is_non_losing = True
                        
                        if is_non_losing:
                            opportunity_is_disruptable = True; break
                    
                    if opportunity_is_disruptable:
                        continue
                else: # is_opportunity=False (threat check)
                    is_sound, _ = self._is_move_sound(work_board, move)
                    if not is_sound:
                        continue

                for fork_info in created_forks:
                    if fork_info['details'] not in existing_forks:
                        if is_opportunity:
                            details = (f"Your move {original_move_san} sets up a potential Fork. "
                                       f"On your next turn, the move {work_board.san(move)} would actualize it: {fork_info['details']}")
                        else:
                            details = (f"The move {work_board.san(move)} creates a potential fork. "
                                       f"The new threat is: {fork_info['details']}")

                        motifs.append({
                            "tactic": "Potential Fork",
                            "details": details.strip(),
                            "value": fork_info.get('value', 0),
                            "canonical_key": ("Potential Fork", work_board.san(move), tuple(fork_info.get("squares", [])))
                        })
        return motifs

    def _detect_potential_forks_for_both_sides(self, board: chess.Board, moving_side: chess.Color, original_move_san: str) -> List[Dict[str, str]]:
        motifs = []
        motifs.extend(self._detect_potential_forks(board, moving_side, original_move_san, is_opportunity=True))
        motifs.extend(self._detect_potential_forks(board, not moving_side, original_move_san, is_opportunity=False))
        return motifs

    def _find_pins_on_board(self, board: chess.Board, side: chess.Color) -> List[Dict[str, str]]:

        motifs = []

        for pinner_square in board.pieces(chess.QUEEN, side) | board.pieces(chess.ROOK, side) | board.pieces(chess.BISHOP, side):

            pinner_piece = board.piece_at(pinner_square)

            if not pinner_piece: continue



            is_pinner_safe = True

            if board.is_attacked_by(not side, pinner_square):

                opponent_board = board.copy()

                opponent_board.turn = not side

                for attacker_square in opponent_board.attackers(not side, pinner_square):

                    capture_pinner_move = chess.Move(attacker_square, pinner_square)

                    if capture_pinner_move in opponent_board.legal_moves:

                        if self.see(opponent_board, capture_pinner_move) >= 0:

                            is_pinner_safe = False

                            break

            if not is_pinner_safe:

                continue



            if pinner_piece.piece_type == chess.ROOK:

                directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

            elif pinner_piece.piece_type == chess.BISHOP:

                directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

            else: # Queen

                directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]

            

            for dr, df in directions:

                front_piece_info = None; back_piece_info = None

                current_rank, current_file = chess.square_rank(pinner_square), chess.square_file(pinner_square)

                for _ in range(8):

                    current_rank += dr; current_file += df

                    if not (0 <= current_rank <= 7 and 0 <= current_file <= 7): break

                    ray_square = chess.square(current_file, current_rank)

                    piece_on_ray = board.piece_at(ray_square)

                    if piece_on_ray:

                        if front_piece_info is None:

                            front_piece_info = (piece_on_ray, ray_square)

                        else:

                            back_piece_info = (piece_on_ray, ray_square); break

                

                if front_piece_info and back_piece_info:

                    front_piece, front_square = front_piece_info

                    back_piece, back_square = back_piece_info



                    if front_piece.piece_type == chess.PAWN or front_piece.color == side or back_piece.color == side:

                        continue



                    is_exploitable_pin = False

                    temp_board = board.copy()

                    temp_board.turn = not side



                    for move in temp_board.legal_moves:

                        if move.from_square == front_square:

                            board_after_front_moved = temp_board.copy()

                            board_after_front_moved.push(move)

                            board_after_front_moved.turn = side

                            capture_back_piece_move = chess.Move(pinner_square, back_square)



                            if capture_back_piece_move in board_after_front_moved.legal_moves:

                                see_value = self.see(board_after_front_moved, capture_back_piece_move)

                                if see_value > 0:

                                    is_exploitable_pin = True

                                    break

                    

                    if is_exploitable_pin:

                        pinner_color = "White" if pinner_piece.color == chess.WHITE else "Black"

                        pinned_color = "White" if front_piece.color == chess.WHITE else "Black"

                        valuable_piece_color = "White" if back_piece.color == chess.WHITE else "Black"

                        details = (f"The {pinner_color} {chess.piece_name(pinner_piece.piece_type)} on {chess.square_name(pinner_square)} "

                                    f"pins the enemy {pinned_color} {chess.piece_name(front_piece.piece_type)} on {chess.square_name(front_square)} "

                                    f"to the {valuable_piece_color} {chess.piece_name(back_piece.piece_type)} on {chess.square_name(back_square)}. "

                                    "Moving the pinned piece would allow a profitable capture.")

                        motifs.append({"tactic": "Pin", "details": details, "value": self.PIECE_VALUES.get(front_piece.piece_type, 0), "squares": [pinner_square, front_square, back_square]})

        return motifs

    def _detect_potential_pins(self, board: chess.Board, side: chess.Color, original_move_san: str, is_opportunity: bool) -> List[Dict[str, str]]:
        motifs = []

        work_board = board
        if board.turn != side:
            b = board.copy()
            if b.is_check():
                return motifs
            try:
                b.push(chess.Move.null())
            except ValueError:
                return motifs

            if b.turn != side:
                return motifs
            work_board = b

        existing_pins = {p['details'] for p in self._find_pins_on_board(work_board, side)}

        for move in work_board.legal_moves:
            board_after_move = work_board.copy()
            board_after_move.push(move)
            created_pins = self._find_pins_on_board(board_after_move, side)

            if created_pins:
                if is_opportunity:
                    # VALIDITY CHECK: An opportunity is only valid if the opponent has no
                    # non-losing move on ply 2 that can disrupt it.
                    opportunity_is_disruptable = False
                    for opponent_move in board.legal_moves:
                        board_after_opponent_move = board.copy()
                        board_after_opponent_move.push(opponent_move)

                        tactic_is_broken = False
                        if move not in board_after_opponent_move.legal_moves:
                            tactic_is_broken = True
                        else:
                            board_after_our_move = board_after_opponent_move.copy()
                            board_after_our_move.push(move)
                            if not self._find_pins_on_board(board_after_our_move, side):
                                tactic_is_broken = True
                        
                        if not tactic_is_broken: continue

                        is_non_losing = False
                        if board.is_capture(opponent_move):
                            if self.see(board, opponent_move) >= 0: is_non_losing = True
                        else: # Quiet move
                            board_after_quiet_move = board.copy()
                            board_after_quiet_move.push(opponent_move)
                            has_profitable_reply = False
                            for my_reply in board_after_quiet_move.legal_moves:
                                if board_after_quiet_move.is_capture(my_reply) and self.see(board_after_quiet_move, my_reply) > 0:
                                    has_profitable_reply = True; break
                            if not has_profitable_reply: is_non_losing = True
                        
                        if is_non_losing:
                            opportunity_is_disruptable = True; break
                    
                    if opportunity_is_disruptable:
                        continue
                else: # is_opportunity=False (threat check)
                    is_sound, _ = self._is_move_sound(work_board, move)
                    if not is_sound:
                        continue

                for pin_info in created_pins:
                    if pin_info['details'] not in existing_pins:
                        move_san = work_board.san(move)
                        tactic_name = "Potential Pin"
                        if is_opportunity:
                            details = (f"Your move {original_move_san} sets up a potential Pin. "
                                       f"On your next turn, the move {move_san} would actualize it: {pin_info['details']}")
                        else:
                            details = (f"The move {move_san} creates a potential pin. "
                                       f"The new threat is: {pin_info['details']}")

                        motifs.append({
                            "tactic": tactic_name,
                            "details": details.strip(),
                            "canonical_key": (tactic_name, move_san, tuple(pin_info.get("squares", []))),
                            "value": pin_info.get('value', 0),
                        })
        return motifs

    def _detect_potential_pins_for_both_sides(self, board: chess.Board, moving_side: chess.Color, original_move_san: str) -> List[Dict[str, str]]:
        motifs = []
        # Find potential pins for the side that just moved (opportunities)
        motifs.extend(self._detect_potential_pins(board, moving_side, original_move_san, is_opportunity=True))
        # Find potential pins for the side whose turn it is now (threats)
        motifs.extend(self._detect_potential_pins(board, not moving_side, original_move_san, is_opportunity=False))
        return motifs

    def _find_skewers_on_board(self, board: chess.Board, side: chess.Color) -> List[Dict[str, str]]:
        motifs = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if (piece is not None and piece.color == side and piece.piece_type in [chess.BISHOP, chess.ROOK, chess.QUEEN]):
                attacked_squares = board.attacks(square)
                for target_square in attacked_squares:
                    target = board.piece_at(target_square)
                    if (target is not None and target.color != side and target.piece_type != chess.PAWN):
                        behind_pieces = self._get_pieces_behind(board, square, target_square)
                        for behind_square, behind_piece in behind_pieces:
                            if (behind_piece.color != side and behind_piece.piece_type != chess.PAWN and
                                self.PIECE_VALUES[behind_piece.piece_type] >= 300):
                                attacker_value = self.PIECE_VALUES[piece.piece_type]
                                front_value = self.PIECE_VALUES[target.piece_type]
                                if front_value <= attacker_value: continue
                                temp_board = board.copy(); temp_board.turn = side
                                attack_move = chess.Move(square, target_square)
                                if attack_move not in temp_board.legal_moves: continue
                                temp_board.turn = not side
                                best_outcome_for_us = float('inf')
                                front_piece_moves = [m for m in temp_board.legal_moves if m.from_square == target_square]
                                if not front_piece_moves: continue
                                for escape_move in front_piece_moves:
                                    test_board = temp_board.copy(); test_board.push(escape_move); test_board.turn = side
                                    capture_behind = chess.Move(square, behind_square)
                                    if capture_behind in test_board.legal_moves:
                                        see_value, _ = self.see(test_board, capture_behind, with_sequence=True)
                                        best_outcome_for_us = min(best_outcome_for_us, see_value)
                                    else:
                                        best_outcome_for_us = -1; break
                                if best_outcome_for_us > 0:
                                    skewerer_color = "White" if piece.color == chess.WHITE else "Black"
                                    front_color = "White" if target.color == chess.WHITE else "Black"
                                    behind_color = "White" if behind_piece.color == chess.WHITE else "Black"
                                    details = f"The {skewerer_color} {chess.piece_name(piece.piece_type)} on {chess.square_name(square)} skewers the {front_color} {chess.piece_name(target.piece_type)} on {chess.square_name(target_square)} and the {behind_color} {chess.piece_name(behind_piece.piece_type)} on {chess.square_name(behind_square)}."
                                    motifs.append({"tactic": "Skewer", "details": details, "value": best_outcome_for_us, "squares": [square, target_square, behind_square]})
        return motifs

    def _detect_potential_skewers(self, board: chess.Board, side: chess.Color, original_move_san: str, is_opportunity: bool) -> List[Dict[str, str]]:
        motifs = []

        work_board = board
        if board.turn != side:
            b = board.copy()
            if b.is_check():
                return motifs
            try:
                b.push(chess.Move.null())
            except ValueError:
                return motifs
            if b.turn != side:
                return motifs
            work_board = b

        existing_skewers = {s['details'] for s in self._find_skewers_on_board(work_board, side)}

        for move in work_board.legal_moves:
            board_after_move = work_board.copy()
            board_after_move.push(move)
            created_skewers = self._find_skewers_on_board(board_after_move, side)

            if created_skewers:
                if is_opportunity:
                    # VALIDITY CHECK: An opportunity is only valid if the opponent has no
                    # non-losing move on ply 2 that can disrupt it.
                    opportunity_is_disruptable = False
                    for opponent_move in board.legal_moves:
                        board_after_opponent_move = board.copy()
                        board_after_opponent_move.push(opponent_move)

                        tactic_is_broken = False
                        if move not in board_after_opponent_move.legal_moves:
                            tactic_is_broken = True
                        else:
                            board_after_our_move = board_after_opponent_move.copy()
                            board_after_our_move.push(move)
                            if not self._find_skewers_on_board(board_after_our_move, side):
                                tactic_is_broken = True
                        
                        if not tactic_is_broken: continue

                        is_non_losing = False
                        if board.is_capture(opponent_move):
                            if self.see(board, opponent_move) >= 0: is_non_losing = True
                        else: # Quiet move
                            board_after_quiet_move = board.copy()
                            board_after_quiet_move.push(opponent_move)
                            has_profitable_reply = False
                            for my_reply in board_after_quiet_move.legal_moves:
                                if board_after_quiet_move.is_capture(my_reply) and self.see(board_after_quiet_move, my_reply) > 0:
                                    has_profitable_reply = True; break
                            if not has_profitable_reply: is_non_losing = True
                        
                        if is_non_losing:
                            opportunity_is_disruptable = True; break
                    
                    if opportunity_is_disruptable:
                        continue
                else: # is_opportunity=False (threat check)
                    is_sound, _ = self._is_move_sound(work_board, move)
                    if not is_sound:
                        continue

                for skewer_info in created_skewers:
                    if skewer_info['details'] not in existing_skewers:
                        if is_opportunity:
                            details = (f"Your move {original_move_san} sets up a potential Skewer. "
                                       f"On your next turn, the move {work_board.san(move)} would actualize it: {skewer_info['details']}")
                        else:
                            details = (f"The move {work_board.san(move)} creates a potential skewer. "
                                       f"The new threat is: {skewer_info['details']}")
                        motifs.append({
                            "tactic": "Potential Skewer",
                            "details": details.strip(),
                            "value": skewer_info.get('value', 0),
                            "canonical_key": ("Potential Skewer", work_board.san(move), tuple(skewer_info.get("squares", [])))
                        })
        return motifs

    def _detect_potential_skewers_for_both_sides(self, board: chess.Board, moving_side: chess.Color, original_move_san: str) -> List[Dict[str, str]]:
        motifs = []
        motifs.extend(self._detect_potential_skewers(board, moving_side, original_move_san, is_opportunity=True))
        motifs.extend(self._detect_potential_skewers(board, not moving_side, original_move_san, is_opportunity=False))
        return motifs

    def _detect_hanging_pieces(self, board: chess.Board, side: chess.Color, last_move_to_square: int, was_capture: bool = False) -> List[Dict[str, str]]:
        motifs = []; hanging_pieces = []
        for square in chess.SQUARES:
            if was_capture and square == last_move_to_square:
                continue

            piece = board.piece_at(square)
            if piece is None or piece.color != side or piece.piece_type == chess.KING: continue
            if not board.is_attacked_by(not side, square): continue
            if board.is_attacked_by(side, square): continue
            attackers = board.attackers(not side, square)
            if not attackers: continue
            can_be_captured_profitably = False; best_capture_value = 0
            temp_board = board.copy(); temp_board.turn = not side
            for attacker_square in attackers:
                capture_move = chess.Move(attacker_square, square)
                if capture_move in temp_board.legal_moves:
                    see_value = self.see(temp_board, capture_move)
                    if see_value > 0:
                        can_be_captured_profitably = True
                        best_capture_value = max(best_capture_value, see_value)
            if can_be_captured_profitably:
                piece_value = self.PIECE_VALUES[piece.piece_type]
                hanging_pieces.append((piece, square, piece_value, best_capture_value))
        if hanging_pieces:
            hanging_pieces.sort(key=lambda x: x[2], reverse=True)
            best_hanging = hanging_pieces[0]
            piece, square, piece_value, capture_value = best_hanging
            hanging_piece_color = "White" if piece.color == chess.WHITE else "Black"
            square_name = chess.square_name(square)
            piece_name = chess.piece_name(piece.piece_type)
            details = (
                    f"The {hanging_piece_color} {piece_name} on {square_name} is hanging "
                    f"and can be captured profitably (SEE: +{capture_value/100.0:.2f}).")
            motifs.append({
                "tactic": "Hanging Piece", 
                "details": details, 
                "value": piece_value
            })
        return motifs

    

    def _detect_discovered_attacks(self, board_before: chess.Board, move: chess.Move) -> List[Dict[str, str]]:
        moving_piece = board_before.piece_at(move.from_square)
        if not moving_piece: return []

        board_after = board_before.copy()
        board_after.push(move)

        # === STEP 1: Analyze the Setup Move ===
        is_setup_move_sound = False
        if board_before.is_capture(move):
            # Path A: The setup move is a capture. Use the powerful tool.
            initial_value = self.get_material_balance(board_before) if board_before.turn == chess.WHITE else -self.get_material_balance(board_before)
            final_score, _, _ = self.evaluate_move_with_quiescence(board_before, move)
            gain = final_score - initial_value
            if gain >= 0:  # Condition: Even or Profitable
                is_setup_move_sound = True
        else:
            # Path B: The setup move is quiet. Check for opponent's tactical replies.
            opponent_trades = self._detect_potential_profitable_trades(board_after, board_after.turn)
            # If the list of trades is empty or just contains the "None" motif, the move is safe.
            if not opponent_trades or opponent_trades[0].get("tactic") == "None":
                is_setup_move_sound = True

        if not is_setup_move_sound:
            return [] # Stop if the setup move was not sound.

        # === STEP 2: Find and Analyze the Discovered Threat ===
        best_discovered_attack = None
        for revealed_square in chess.SQUARES:
            revealed_piece = board_before.piece_at(revealed_square)
            if not revealed_piece or revealed_piece.color != moving_piece.color: continue
            if revealed_piece.piece_type not in [chess.BISHOP, chess.ROOK, chess.QUEEN]: continue
            if not self._are_on_same_line(move.from_square, revealed_square): continue
            
            newly_attacked_squares = board_after.attacks(revealed_square) - board_before.attacks(revealed_square)
            for target_square in newly_attacked_squares:
                target_piece = board_after.piece_at(target_square)
                if target_piece and target_piece.color != moving_piece.color:
                    # Condition: Discovered threat must be strictly profitable
                    board_for_see = board_after.copy()
                    board_for_see.turn = moving_piece.color
                    revealed_capture = chess.Move(revealed_square, target_square)
                    if revealed_capture in board_for_see.legal_moves and self.see(board_for_see, revealed_capture) > 0:
                        current_value = self.PIECE_VALUES.get(target_piece.piece_type, 0)
                        if best_discovered_attack is None or current_value > best_discovered_attack['value']:
                            best_discovered_attack = {
                                'value': current_value, 'revealed_piece': revealed_piece,
                                'revealed_square': revealed_square, 'target_piece': target_piece,
                                'target_square': target_square, 'is_check': board_after.is_check()
                            }

        if not best_discovered_attack:
            return [] # No profitable discovered attack found.

        # === STEP 3: Check for Double Attack & Classify ===
        is_primary_threat_profitable = False
        primary_threat_details = ""
        
        best_primary_target_value = 0
        best_primary_target_sq = None
        # Find the best primary threat
        for attacked_sq in board_after.attacks(move.to_square):
            threatened_piece = board_after.piece_at(attacked_sq)
            if threatened_piece and threatened_piece.color != moving_piece.color:
                value = self.PIECE_VALUES.get(threatened_piece.piece_type, 0)
                if value > best_primary_target_value:
                    best_primary_target_value = value
                    best_primary_target_sq = attacked_sq
        
        # Check if the best primary threat is profitable
        if best_primary_target_sq is not None:
            primary_threat_move = chess.Move(move.to_square, best_primary_target_sq)
            temp_board_see = board_after.copy()
            temp_board_see.turn = moving_piece.color
            # Condition: Primary threat must be strictly profitable
            if primary_threat_move in temp_board_see.legal_moves and self.see(temp_board_see, primary_threat_move) > 0:
                is_primary_threat_profitable = True
                threatened_piece = board_after.piece_at(best_primary_target_sq)
                moving_color = "White" if moving_piece.color == chess.WHITE else "Black"
                threatened_color = "White" if threatened_piece.color == chess.WHITE else "Black"
                primary_threat_details = (f"the {moving_color} {chess.piece_name(moving_piece.piece_type)} on {chess.square_name(move.to_square)} now threatens the "
                                          f"{threatened_color} {chess.piece_name(threatened_piece.piece_type)} on {chess.square_name(best_primary_target_sq)}")

        # Final classification logic...
        motifs = []
        moving_color = "White" if moving_piece.color == chess.WHITE else "Black"
        info = best_discovered_attack
        rev_piece, rev_sq = info['revealed_piece'], info['revealed_square']
        tar_piece, tar_sq = info['target_piece'], info['target_square']
        revealed_color = "White" if rev_piece.color == chess.WHITE else "Black"
        target_color = "White" if tar_piece.color == chess.WHITE else "Black"
        secondary_attack_details = (f"a discovered attack from the {revealed_color} {chess.piece_name(rev_piece.piece_type)} on {chess.square_name(rev_sq)} "
                                    f"to the {target_color} {chess.piece_name(tar_piece.piece_type)} on {chess.square_name(tar_sq)}.")
        
        if is_primary_threat_profitable:
            tactic = "Double Attack"
            details = (f"The move {board_before.san(move)} is a Double Attack. "
                       f"The primary threat is that {primary_threat_details}. Simultaneously, the move unleashes {secondary_attack_details}")
        elif info['is_check']:
            tactic = "Discovered Check"
            details = (f"Moving the {moving_color} {chess.piece_name(moving_piece.piece_type)} to {chess.square_name(move.to_square)} unleashes a Discovered Check "
                       f"from the {revealed_color} {chess.piece_name(rev_piece.piece_type)} on {chess.square_name(rev_sq)}.")
        else:
            tactic = "Discovered Attack"
            details = (f"Moving the {moving_color} {chess.piece_name(moving_piece.piece_type)} to {chess.square_name(move.to_square)} unleashes {secondary_attack_details}")
        
        motifs.append({"tactic": tactic, "details": details, "value": info['value'], "secondary_attack": secondary_attack_details, "squares": [move.to_square, info['revealed_square'], info['target_square']]})
        return motifs
            
  

    def _detect_potential_discovered_attacks(self, board: chess.Board, side: chess.Color, original_move_san: str, is_opportunity: bool) -> List[Dict[str, str]]:
        motifs = []

        work_board = board
        if board.turn != side:
            b = board.copy()
            if b.is_check():
                return motifs
            try:
                b.push(chess.Move.null())
            except ValueError:
                return motifs
            if b.turn != side:
                return motifs
            work_board = b

        for move in work_board.legal_moves:
            created_discoveries = self._detect_discovered_attacks(work_board, move)

            if created_discoveries:
                if is_opportunity:
                    # VALIDITY CHECK: An opportunity is only valid if the opponent has no
                    # non-losing move on ply 2 that can disrupt it.
                    opportunity_is_disruptable = False
                    for opponent_move in board.legal_moves:
                        board_after_opponent_move = board.copy()
                        board_after_opponent_move.push(opponent_move)

                        tactic_is_broken = False
                        if move not in board_after_opponent_move.legal_moves:
                            tactic_is_broken = True
                        else:
                            # This tactic check needs the board *before* the tactical move.
                            if not self._detect_discovered_attacks(board_after_opponent_move, move):
                                tactic_is_broken = True
                        
                        if not tactic_is_broken: continue

                        is_non_losing = False
                        if board.is_capture(opponent_move):
                            if self.see(board, opponent_move) >= 0: is_non_losing = True
                        else: # Quiet move
                            board_after_quiet_move = board.copy()
                            board_after_quiet_move.push(opponent_move)
                            has_profitable_reply = False
                            for my_reply in board_after_quiet_move.legal_moves:
                                if board_after_quiet_move.is_capture(my_reply) and self.see(board_after_quiet_move, my_reply) > 0:
                                    has_profitable_reply = True; break
                            if not has_profitable_reply: is_non_losing = True
                        
                        if is_non_losing:
                            opportunity_is_disruptable = True; break
                    
                    if opportunity_is_disruptable:
                        continue
                else: # is_opportunity=False (threat check)
                    is_sound, _ = self._is_move_sound(work_board, move)
                    if not is_sound:
                        continue

                best_discovery = created_discoveries[0]
                secondary_attack = best_discovery.get("secondary_attack", "a powerful discovered attack.")
                
                if is_opportunity:
                    details = (f"Your move {original_move_san} sets up a potential discovered attack. "
                               f"On your next turn, the move {work_board.san(move)} would unleash {secondary_attack}")
                else:
                    details = (f"The move {work_board.san(move)} creates a potential discovered attack, "
                               f"unleashing {secondary_attack}")

                tactic_type = best_discovery['tactic']
                if "Check" in tactic_type:
                    potential_tactic = "Potential Discovered Check"
                elif "Double Attack" in tactic_type:
                    potential_tactic = "Potential Double Attack"
                else:
                    potential_tactic = "Potential Discovered Attack"

                motifs.append({
                    "tactic": potential_tactic,
                    "details": details,
                    "value": best_discovery.get('value', 0),
                    "canonical_key": (potential_tactic, work_board.san(move), tuple(best_discovery.get("squares", [])))
                })
        return motifs

    def _detect_potential_discovered_attacks_for_both_sides(self, board: chess.Board, moving_side: chess.Color, original_move_san: str) -> List[Dict[str, str]]:
        motifs = []
        motifs.extend(self._detect_potential_discovered_attacks(board, moving_side, original_move_san, is_opportunity=True))
        motifs.extend(self._detect_potential_discovered_attacks(board, not moving_side, original_move_san, is_opportunity=False))
        return motifs

    def _find_trapped_pieces_on_board(self, board: chess.Board, side: chess.Color) -> List[Dict[str, str]]:
        if board.is_check(): return []
        motifs = []
        for square, piece in board.piece_map().items():
            if piece.color != side: continue
            if piece.piece_type in [chess.PAWN, chess.KING] or board.is_pinned(side, square): continue
            if board.is_attacked_by(not side, square):
                safe_moves_count = 0
                temp_board = board.copy(); temp_board.turn = side
                for move in temp_board.legal_moves:
                    if move.from_square == square:
                        if not temp_board.is_attacked_by(not side, move.to_square) or (temp_board.is_capture(move) and self.see(temp_board, move) >= 0):
                            safe_moves_count += 1
                if safe_moves_count < 1:
                    # A piece is only truly trapped if the opponent has a profitable capture.
                    # Let's verify that first.
                    opponent_board = board.copy()
                    opponent_board.turn = not side
                    attackers = board.attackers(not side, square)

                    highest_see_for_opponent = -99999
                    for attacker_sq in attackers:
                        capture_move = chess.Move(attacker_sq, square)
                        if capture_move in opponent_board.legal_moves:
                            see_value = self.see(opponent_board, capture_move)
                            if see_value > highest_see_for_opponent:
                                highest_see_for_opponent = see_value
                    
                    # If the opponent's best capture isn't profitable for them, our piece isn't trapped.
                    if highest_see_for_opponent <= 0:
                        continue

                    # If the opponent's capture IS profitable, now we check for our own desperate measures.
                    attackers_on_piece = list(board.attackers(not side, square))
                    profitable_threat_removals = []; losing_threat_removals = []
                    temp_board_for_see = board.copy(); temp_board_for_see.turn = side
                    for attacker_sq in attackers_on_piece:
                        our_recapturers = board.attackers(side, attacker_sq)
                        for recapturer_sq in our_recapturers:
                            capture_move = chess.Move(recapturer_sq, attacker_sq)
                            if capture_move in temp_board_for_see.legal_moves:
                                see_value = self.see(temp_board_for_see, capture_move)
                                if see_value >= 0:
                                    profitable_threat_removals.append(capture_move)
                                else:
                                    losing_threat_removals.append({'move': capture_move, 'see': see_value})
                    if profitable_threat_removals: continue
                    loss_from_inaction = self.PIECE_VALUES[piece.piece_type]
                    if losing_threat_removals:
                        best_losing_trade = max(losing_threat_removals, key=lambda x: x['see'])
                        loss_from_trade = -best_losing_trade['see']
                        if loss_from_trade < loss_from_inaction:
                            color_name = "White" if piece.color == chess.WHITE else "Black"
                            move_san = board.san(best_losing_trade['move'])
                            details = (f"The {color_name} {chess.piece_name(piece.piece_type)} on {chess.square_name(square)} is trapped. "
                                       f"However, the forced capture {move_san} is the 'Lesser of Two Evils'. "
                                       f"It results in a material loss of {loss_from_trade/100.0:.2f}, which is better than losing the "
                                       f"{chess.piece_name(piece.piece_type)} (a loss of {loss_from_inaction/100.0:.2f}).")
                            motifs.append({"tactic": "Lesser of Two Evils", "details": details, "value": loss_from_trade, "squares": [square]})
                    color_name = "White" if piece.color == chess.WHITE else "Black"
                    details = (f"The {color_name} {chess.piece_name(piece.piece_type)} on {chess.square_name(square)} is trapped. "
                               f"It is under attack and has no safe squares or favorable trades to escape.")
                    motifs.append({"tactic": "Trapped Piece", "details": details, "value": loss_from_inaction, "squares": [square]})
        return motifs

    def _detect_potential_trapped_pieces(self, board: chess.Board, side: chess.Color, original_move_san: str, is_opportunity: bool) -> List[Dict[str, str]]:
        motifs = []

        work_board = board
        if board.turn != side:
            b = board.copy()
            if b.is_check():
                return motifs
            try:
                b.push(chess.Move.null())
            except ValueError:
                return motifs
            if b.turn != side:
                return motifs
            work_board = b

        existing_traps = {t['details'] for t in self._find_trapped_pieces_on_board(work_board, not side)}

        for move in work_board.legal_moves:
            board_after_move = work_board.copy()
            board_after_move.push(move)
            created_traps = self._find_trapped_pieces_on_board(board_after_move, not side)

            if created_traps:
                if is_opportunity:
                    # VALIDITY CHECK: An opportunity is only valid if the opponent has no
                    # non-losing move on ply 2 that can disrupt it.
                    opportunity_is_disruptable = False
                    for opponent_move in board.legal_moves:
                        board_after_opponent_move = board.copy()
                        board_after_opponent_move.push(opponent_move)

                        tactic_is_broken = False
                        if move not in board_after_opponent_move.legal_moves:
                            tactic_is_broken = True
                        else:
                            board_after_our_move = board_after_opponent_move.copy()
                            board_after_our_move.push(move)
                            if not self._find_trapped_pieces_on_board(board_after_our_move, not side):
                                tactic_is_broken = True
                        
                        if not tactic_is_broken: continue

                        is_non_losing = False
                        if board.is_capture(opponent_move):
                            if self.see(board, opponent_move) >= 0: is_non_losing = True
                        else: # Quiet move
                            board_after_quiet_move = board.copy()
                            board_after_quiet_move.push(opponent_move)
                            has_profitable_reply = False
                            for my_reply in board_after_quiet_move.legal_moves:
                                if board_after_quiet_move.is_capture(my_reply) and self.see(board_after_quiet_move, my_reply) > 0:
                                    has_profitable_reply = True; break
                            if not has_profitable_reply: is_non_losing = True
                        
                        if is_non_losing:
                            opportunity_is_disruptable = True; break
                    
                    if opportunity_is_disruptable:
                        continue
                else: # is_opportunity=False (threat check)
                    is_sound, _ = self._is_move_sound(work_board, move)
                    if not is_sound:
                        continue

                for trap_info in created_traps:
                    if trap_info['details'] not in existing_traps:
                        if is_opportunity:
                            details = (f"Your move {original_move_san} sets up a potential Trap. "
                                       f"On your next turn, the move {work_board.san(move)} would actualize it: {trap_info['details']}")
                        else:
                            details = (f"The move {work_board.san(move)} creates a potential trap. "
                                       f"The new situation is: {trap_info['details']}")
                        motifs.append({
                            "tactic": "Potential Trapped Piece",
                            "details": details.strip(),
                            "value": trap_info.get('value', 0),
                            "canonical_key": ("Potential Trapped Piece", work_board.san(move), tuple(trap_info.get("squares", [])))
                        })
        return motifs

    def _detect_potential_trapped_pieces_for_both_sides(self, board: chess.Board, moving_side: chess.Color, original_move_san: str) -> List[Dict[str, str]]:
        motifs = []
        motifs.extend(self._detect_potential_trapped_pieces(board, moving_side, original_move_san, is_opportunity=True))
        motifs.extend(self._detect_potential_trapped_pieces(board, not moving_side, original_move_san, is_opportunity=False))
        return motifs

    def _detect_battery_threats(self, board: chess.Board, side: chess.Color) -> List[Dict[str, str]]:
        motifs = []
        player_pieces = [(sq, p) for sq, p in board.piece_map().items() if p.color == side]
        for i in range(len(player_pieces)):
            for j in range(i + 1, len(player_pieces)):
                sq1, p1 = player_pieces[i]
                sq2, p2 = player_pieces[j]
                t1, t2 = p1.piece_type, p2.piece_type

                is_valid_battery = False
                
                # Check for Rook/Queen batteries on ranks or files
                is_rook_family = t1 in {chess.ROOK, chess.QUEEN} and t2 in {chess.ROOK, chess.QUEEN}
                if is_rook_family and (chess.square_file(sq1) == chess.square_file(sq2) or chess.square_rank(sq1) == chess.square_rank(sq2)):
                    is_valid_battery = True

                # Check for Bishop/Queen batteries on diagonals
                is_bishop_family = t1 in {chess.BISHOP, chess.QUEEN} and t2 in {chess.BISHOP, chess.QUEEN}
                if not is_valid_battery and is_bishop_family and abs(chess.square_file(sq1) - chess.square_file(sq2)) == abs(chess.square_rank(sq1) - chess.square_rank(sq2)):
                    is_valid_battery = True

                if is_valid_battery and not self._get_pieces_between(board, sq1, sq2):
                    color_name = "White" if p1.color == chess.WHITE else "Black"
                    details = (f"A battery has been formed by the {color_name} {chess.piece_name(p1.piece_type)} on {chess.square_name(sq1)} "
                               f"and the {color_name} {chess.piece_name(p2.piece_type)} on {chess.square_name(sq2)}.")
                    
                    motifs.append({"tactic": "Battery", "details": details, "value": 50})
        return motifs

    def _detect_overloaded_pieces(self, board: chess.Board, side: chess.Color) -> List[Dict[str, str]]:
        motifs = []; enemy_color = not side
        for square, piece in board.piece_map().items():
            if piece.color != enemy_color or piece.piece_type == chess.KING:
                continue
            solely_defended_targets = []
            attacked_squares = board.attacks(square)
            for def_sq in attacked_squares:
                def_piece = board.piece_at(def_sq)
                if def_piece and def_piece.color == enemy_color and def_piece.piece_type != chess.KING:
                    if board.is_attacked_by(side, def_sq):
                        defenders = board.attackers(enemy_color, def_sq)
                        if len(defenders) == 1 and list(defenders)[0] == square:
                            solely_defended_targets.append(def_piece)
            if len(solely_defended_targets) >= 2:
                overloaded_piece_color = "White" if piece.color == chess.WHITE else "Black"
                piece_name = chess.piece_name(piece.piece_type)
                target_descs = []
                for t in solely_defended_targets:
                    t_color = "White" if t.color == chess.WHITE else "Black"
                    target_descs.append(f"the {t_color} {chess.piece_name(t.piece_type)}")
                details = (
                        f"The {overloaded_piece_color} {piece_name} on {chess.square_name(square)} is overloaded. "
                        f"It is the sole defender of {target_descs[0]} and {target_descs[1]}.")
                motifs.append({"tactic": "Overloading", "details": details, "value": 250})
        return motifs

    def _detect_clearance_sacrifices(self, board_before: chess.Board, move: chess.Move) -> List[Dict[str, str]]:
        motifs = []
        moving_piece = board_before.piece_at(move.from_square)
        if not moving_piece: return motifs
        if self.see(board_before, move) < 0:
            board_after = board_before.copy(); board_after.push(move)
            board_after.turn = board_before.turn
            for next_move in board_after.legal_moves:
                if board_after.gives_check(next_move):
                    board_after_check = board_after.copy(); board_after_check.push(next_move)
                    if board_after_check.is_checkmate():
                        sacrificing_color = "White" if moving_piece.color == chess.WHITE else "Black"
                        details = f"The move {board_before.san(move)} is a brilliant clearance sacrifice by the {sacrificing_color} {chess.piece_name(moving_piece.piece_type)}, leading to a forced checkmate."
                        motifs.append({"tactic": "Clearance Sacrifice", "details": details, "value": 1000})
        return motifs

    def _detect_decoy_and_attraction(self, board: chess.Board, side: chess.Color) -> List[Dict[str, str]]:
        motifs = []
        for move in board.legal_moves:
            if board.turn != side: continue
            if not board.gives_check(move) and not board.is_capture(move): continue
            moving_piece = board.piece_at(move.from_square)
            if not moving_piece: continue
            temp_board = board.copy(); temp_board.push(move)
            for opp_move in temp_board.legal_moves:
                board_after_reply = temp_board.copy(); board_after_reply.push(opp_move)
                board_after_reply.turn = side
                forks = self._find_forks_on_board(board_after_reply, side)
                if forks:
                    decoy_color = "White" if moving_piece.color == chess.WHITE else "Black"
                    decoy_piece_name = chess.piece_name(moving_piece.piece_type)
                    details = (f"The move {board.san(move)} by the {decoy_color} {decoy_piece_name} is a decoy. It forces the opponent to respond, "
                               f"setting up a deadly tactic: {forks[0]['details']}.")
                    motifs.append({"tactic": "Decoy", "details": details, "value": forks[0]['value']})
        return motifs

    # ========================================================================================
    # HELPER METHODS (ORIGINAL AND NEW)
    # ========================================================================================

    def _find_actual_pinning_piece(self, board: chess.Board, pinned_square: int, 
                                   king_square: int, our_side: chess.Color) -> Optional[Tuple[chess.Piece, int]]:
        rank_diff = chess.square_rank(pinned_square) - chess.square_rank(king_square)
        file_diff = chess.square_file(pinned_square) - chess.square_file(king_square)
        rank_step = 0 if rank_diff == 0 else (1 if rank_diff > 0 else -1)
        file_step = 0 if file_diff == 0 else (1 if file_diff > 0 else -1)
        current_rank, current_file = chess.square_rank(pinned_square) + rank_step, chess.square_file(pinned_square) + file_step
        while 0 <= current_rank <= 7 and 0 <= current_file <= 7:
            square = chess.square(current_file, current_rank)
            piece = board.piece_at(square)
            if piece is not None:
                if piece.color == our_side and self._can_piece_pin(piece, square, king_square):
                    return piece, square
                break
            current_rank += rank_step; current_file += file_step
        return None

    def _can_piece_pin(self, piece: chess.Piece, piece_square: int, king_square: int) -> bool:
        rank_diff = chess.square_rank(king_square) - chess.square_rank(piece_square)
        file_diff = chess.square_file(king_square) - chess.square_file(piece_square)
        if piece.piece_type == chess.ROOK: return rank_diff == 0 or file_diff == 0
        elif piece.piece_type == chess.BISHOP: return abs(rank_diff) == abs(file_diff)
        elif piece.piece_type == chess.QUEEN: return (rank_diff == 0 or file_diff == 0 or abs(rank_diff) == abs(file_diff))
        return False

    def _get_pieces_behind(self, board: chess.Board, attacker_square: int, target_square: int) -> List[Tuple[int, chess.Piece]]:
        pieces_behind = []
        attacker_rank, attacker_file = chess.square_rank(attacker_square), chess.square_file(attacker_square)
        target_rank, target_file = chess.square_rank(target_square), chess.square_file(target_square)
        rank_diff, file_diff = target_rank - attacker_rank, target_file - attacker_file
        if rank_diff != 0 and file_diff != 0 and abs(rank_diff) != abs(file_diff): return pieces_behind
        if rank_diff == 0 and file_diff == 0: return pieces_behind
        step_rank = 0 if rank_diff == 0 else (1 if rank_diff > 0 else -1)
        step_file = 0 if file_diff == 0 else (1 if file_diff > 0 else -1)
        current_rank, current_file = target_rank + step_rank, target_file + step_file
        while 0 <= current_rank <= 7 and 0 <= current_file <= 7:
            square = chess.square(current_file, current_rank)
            piece = board.piece_at(square)
            if piece is not None:
                pieces_behind.append((square, piece)); break
            current_rank += step_rank; current_file += step_file
        return pieces_behind
    
    def _get_pieces_between(self, board: chess.Board, sq1: int, sq2: int) -> List[chess.Piece]:
        pieces = []
        for sq in chess.SquareSet(chess.between(sq1, sq2)):
            piece = board.piece_at(sq)
            if piece: pieces.append(piece)
        return pieces

    def _are_on_same_line(self, sq1: int, sq2: int) -> bool:
        return (
                chess.square_rank(sq1) == chess.square_rank(sq2) or
                chess.square_file(sq1) == chess.square_file(sq2) or
                abs(chess.square_rank(sq1) - chess.square_rank(sq2)) == abs(chess.square_file(sq1) - chess.square_file(sq2))
        )

    def _is_between(self, sq_mid: int, sq1: int, sq2: int) -> bool:
        return sq_mid in chess.SquareSet(chess.between(sq1, sq2))

    def _get_squares_on_line(self, sq1: int, sq2: int) -> List[int]:
        squares = []
        rank1, file1 = chess.square_rank(sq1), chess.square_file(sq1)
        rank2, file2 = chess.square_rank(sq2), chess.square_file(sq2)
        rank_step = rank2 - rank1; file_step = file2 - file1
        if rank_step != 0: rank_step //= abs(rank_step)
        if file_step != 0: file_step //= abs(file_step)
        curr_rank, curr_file = rank2 + rank_step, file2 + file_step
        while 0 <= curr_rank <= 7 and 0 <= curr_file <= 7:
            squares.append(chess.square(curr_file, curr_rank))
            curr_rank += rank_step; curr_file += file_step
        return squares

    def _clean_motifs(self, motifs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not motifs: return motifs
        
        def get_key(m):
            # Use the stable canonical_key if it exists, otherwise fall back to the old method.
            return m.get('canonical_key', (m.get('tactic', ''), m.get('details', '')))

        seen = set()
        unique_motifs = []
        for motif in motifs:
            key = get_key(motif)
            if key not in seen:
                seen.add(key)
                unique_motifs.append(motif)

        priority = {
            'Clearance Sacrifice': 110,
            'Decoy': 105,
            'Double Attack': 102,
            'Discovered Check': 100,
            'Discovered Attack': 95,
            'Fork': 90,
            'Skewer': 85,
            'Pin': 80,
            'Lesser of Two Evils': 79,
            'Trapped Piece': 78,
            'Hanging Piece': 75,
            'Overloading': 70,
            'Interference': 65,
            'Profitable Trade': 50,
            'Battery': 40
        }
        unique_motifs.sort(key=lambda x: (
            priority.get(x.get('tactic', ''), 0),
            x.get('value', 0)
        ), reverse=True)
        return unique_motifs

