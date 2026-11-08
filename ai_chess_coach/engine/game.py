"""Turn a game into per-ply analysis records (models.PlyAnalysis), and describe engine lines.

Most of the time goes to the tactical analyzer (pure Python), not to Stockfish, so this module avoids
repeating tactics work: every engine position is analysed up front through the pool, motif searches
are reused when the same move is examined twice in the same tactics state, and line descriptions
are computed once per line, in worker processes when there are several.
"""

import copy
import logging
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from typing import Callable, Iterable, Optional, Sequence

import chess
import chess.polyglot
import chess.syzygy

from ai_chess_coach.config import AnalysisConfig, Constants
from ai_chess_coach.engine.classify import MATE_SCORE, classify_move
from ai_chess_coach.engine.positional import PositionalAnalyzer
from ai_chess_coach.models import LineAnalysis, PlyAnalysis
from ai_chess_coach.tactics.analyzer import TacticalAnalyzer

log = logging.getLogger(__name__)

OPENING_LAST_MOVE = 12            # moves 1-12 are the opening
ENDGAME_LAST_PLIES = 30           # the last 30 plies of a game are the endgame
DESCRIBE_DEPTH = 3                # moves of a line described with tactics and positional notes
# Worker processes for line descriptions, never more than the machine's cores. Measured on 16 cores
# (15-ply Opera commentary input, 2026-11-08): 4 workers 9.73 s, 8 workers 9.98 s, 16 workers 11.48 s,
# start-up included.
DESCRIBE_WORKERS = min(4, os.cpu_count() or 1)

# Extra engine checks on the played move (depth, time limit in s)
ZWISCHENZUG_CHECK = (14, 0.5)
ZWISCHENZUG_MIN_CP_GAP = 120      # the engine's in-between move must be this much better than the recapture
OPPONENT_CHOICE_CHECK = (14, 0.7)
OPPONENT_CHOICE_MIN_CP = 50       # both captures must be at least this good for the opponent

# Key moments: biggest win-probability losses, plus a bonus for great and brilliant moves
KEY_MOMENT_BONUS = {Constants.BRILLIANT: 50, Constants.GREAT_MOVE: 20}


# ---- small helpers ---------------------------------------------------------------------------

def san_line(moves: Iterable[chess.Move], board: chess.Board) -> list[str]:
    """SAN of `moves` played from `board`, stopping at the first illegal move."""
    board, sans = board.copy(), []
    for move in moves:
        if move not in board.legal_moves:
            break
        sans.append(board.san(move))
        board.push(move)
    return sans


def game_phase(ply_index: int, total_plies: int) -> str:
    if ply_index // 2 + 1 <= OPENING_LAST_MOVE:
        return "opening"
    if ply_index > total_plies - ENDGAME_LAST_PLIES:
        return "endgame"
    return "middlegame"


def find_key_moments(plies: Sequence[PlyAnalysis], count: int = 5) -> list[PlyAnalysis]:
    """The `count` plies with the most impact, in game order."""
    ranked = sorted(plies, key=lambda p: p.points_loss * 100 + KEY_MOMENT_BONUS.get(p.classification, 0), reverse=True)
    return sorted(ranked[:count], key=lambda p: p.ply)


def _white_cp(line: dict) -> Optional[int]:
    return line["score"].white().score(mate_score=MATE_SCORE) if line.get("score") else None


def _first_white_cp(lines: list) -> int:
    try:
        return lines[0]["score"].white().score(mate_score=MATE_SCORE) if lines else 0
    except (KeyError, AttributeError):
        return 0


# ---- line descriptions -----------------------------------------------------------------------

def _describe(tactics: TacticalAnalyzer, positional: PositionalAnalyzer, fen: str, uci: Sequence[str]) -> dict:
    """SAN of the line plus tactical motifs and notable positional features after each of its first moves."""
    board = chess.Board(fen)
    sans = san_line((chess.Move.from_uci(u) for u in uci), board)
    if not sans:
        return {}
    result: dict = {"san": " ".join(sans[:DESCRIBE_DEPTH])}
    moves = []
    for u in uci[:DESCRIBE_DEPTH]:
        move = chess.Move.from_uci(u)
        if move not in board.legal_moves:
            break
        before = board.copy()
        board.push(move)
        entry: dict = {"move_san": before.san(move)}
        motifs = [m for m in tactics.find_tactical_motifs(before, board, move) if m.get("tactic") != "None"]
        if motifs:
            entry["tactical_motifs"] = motifs
        notes = {
            k: v for k, v in asdict(positional.analyze_position(board)).items()
            if not (isinstance(v, (list, dict)) and not v)
            and not (isinstance(v, float) and v == 0.0)
            and not (k == "king_safety" and isinstance(v, dict) and v.get("status") == "safe")
        }
        if notes:
            entry["positional_analysis"] = notes
        moves.append(entry)
    if moves:
        result["moves_analysis"] = moves
    return result


_worker_analyzers = None


def _start_worker() -> None:
    global _worker_analyzers
    _worker_analyzers = (TacticalAnalyzer(), PositionalAnalyzer(AnalysisConfig()))


def _describe_in_worker(key: tuple) -> tuple:
    fen, uci = key
    return key, _describe(*_worker_analyzers, fen, uci)


def _stop_workers(executor: ProcessPoolExecutor) -> None:
    """Cancel queued work and end the worker processes now, even busy ones."""
    processes = list((executor._processes or {}).values())  # private: the only handle on busy workers
    executor.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5)


class LineDescriber:
    """describe(fen, uci) for engine lines, as used in the coach's prompts.

    Every description starts from a fresh tactics state, so the same line always gets the same
    description (the old code filtered motifs by whichever game move was analysed last).
    Results are kept, and prefetch() computes many lines at once in worker processes.
    """

    def __init__(self, workers: int = DESCRIBE_WORKERS):
        self._workers = workers
        self._tactics = TacticalAnalyzer()  # never given a ply index, so its state stays empty
        self._positional = PositionalAnalyzer(AnalysisConfig())
        self._done: dict = {}
        self._lock = threading.Lock()
        self._executor: Optional[ProcessPoolExecutor] = None
        self._closed = False

    @staticmethod
    def _key(fen: str, uci: Sequence[str]) -> tuple:
        return fen, tuple(uci[:DESCRIBE_DEPTH])

    def __call__(self, fen: str, uci: Sequence[str]) -> dict:
        key = self._key(fen, uci)
        with self._lock:
            known = self._done.get(key)
        if known is None:
            known = _describe(self._tactics, self._positional, *key)
            with self._lock:
                self._done[key] = known
        return copy.deepcopy(known)

    def prefetch(self, requests: Iterable[tuple]) -> None:
        """Describe every (fen, uci) not seen yet, spread over worker processes. Does nothing after close()."""
        with self._lock:
            if self._closed:
                return
            missing = [k for k in dict.fromkeys(self._key(f, u) for f, u in requests) if k not in self._done]
        if len(missing) < 2 or self._workers < 2:
            for fen, uci in missing:
                self(fen, uci)
            return
        results = self._describe_in_workers(missing)
        if results is None:  # the workers could not start or failed: describe the lines here
            if self._closed:
                return
            results = {k: _describe(self._tactics, self._positional, *k) for k in missing}
        with self._lock:
            self._done.update(results)

    def _describe_in_workers(self, keys: list) -> Optional[dict]:
        try:
            executor = ProcessPoolExecutor(max_workers=min(self._workers, len(keys)), initializer=_start_worker)
        except Exception as e:
            log.warning("Could not start worker processes (%s); describing lines in this process", e)
            return None
        with self._lock:
            if self._closed:
                _stop_workers(executor)
                return None
            self._executor = executor
        try:
            return dict(executor.map(_describe_in_worker, keys))
        except Exception as e:  # a worker failed to start or crashed, or close() stopped them
            if not self._closed:
                log.warning("Worker processes failed (%s); describing lines in this process", e)
            return None
        finally:
            with self._lock:
                self._executor = None
            _stop_workers(executor)

    def close(self) -> None:
        """Stop worker processes still running (window closed); later prefetch() calls do nothing."""
        with self._lock:
            self._closed = True
            executor = self._executor
        if executor:
            _stop_workers(executor)


# ---- game analysis ---------------------------------------------------------------------------

class GameAnalyzer:
    """Engine pool plus the tactical, positional and classification layers, for one app session."""

    def __init__(self, pool, config: Optional[AnalysisConfig] = None, opening_book_path: Optional[str] = None,
                 syzygy_path: Optional[str] = None, describer: Optional[LineDescriber] = None):
        self.pool = pool
        self.config = config or AnalysisConfig()
        self.tactics = TacticalAnalyzer()
        self.positional = PositionalAnalyzer(self.config)
        self.describe_line = describer or LineDescriber()
        self.opening_book = self.tablebases = None
        if opening_book_path and os.path.exists(opening_book_path):
            try:
                self.opening_book = chess.polyglot.open_reader(opening_book_path)
            except Exception as e:
                log.warning("Could not load opening book: %s", e)
        if syzygy_path and os.path.exists(syzygy_path):
            try:
                self.tablebases = chess.syzygy.open_tablebase(syzygy_path)
            except Exception as e:
                log.warning("Could not load Syzygy tablebases: %s", e)
        self._motif_memo: dict = {}

    def analyse(self, board: chess.Board, depth: int, multipv: int = 1, time_limit: Optional[float] = None) -> list:
        return self.pool.analyse(board, depth, multipv, time_limit)

    def close(self) -> None:
        self.describe_line.close()
        self.pool.close()
        if self.opening_book:
            self.opening_book.close()
        if self.tablebases:
            self.tablebases.close()

    def probe_wdl(self, board: chess.Board) -> Optional[int]:
        if not self.tablebases or len(board.piece_map()) > self.tablebases.max_pieces:
            return None
        try:
            return self.tablebases.probe_wdl(board)
        except (IndexError, KeyError):
            return None

    # -- tactics with reuse --

    def _motifs(self, before: chess.Board, after: chess.Board, move: chess.Move, previous: Optional[chess.Move],
                ply_index: Optional[int] = None) -> list:
        if ply_index is not None:  # a game move: updates the tactics state, never reused
            return self.tactics.find_tactical_motifs(before.copy(), after.copy(), move, ply_index=ply_index, previous_move=previous)
        key = (before.fen(), move.uci(), previous.uci() if previous else None, frozenset(self.tactics.previous_motifs))
        if key not in self._motif_memo:
            self._motif_memo[key] = self.tactics.find_tactical_motifs(before.copy(), after.copy(), move, previous_move=previous)
        return copy.deepcopy(self._motif_memo[key])

    def _line(self, board: chess.Board, engine_line: dict, previous: Optional[chess.Move]) -> LineAnalysis:
        pv = engine_line.get("pv") or []
        if not pv or pv[0] not in board.legal_moves:
            return LineAnalysis()
        after = board.copy()
        after.push(pv[0])
        cp = _white_cp(engine_line)
        return LineAnalysis(
            san=" ".join(san_line(pv, board)), uci=[m.uci() for m in pv], score_cp=cp or 0,
            tactical_motifs=self._motifs(board, after, pv[0], previous),
            positional_analysis=self.positional.analyze_position(after),
        )

    # -- extra checks on the played move --

    def _missed_zwischenzug(self, board: chess.Board, move: chess.Move, lines_before: list) -> Optional[dict]:
        """A capture was played where the engine had a strong in-between move instead."""
        if not lines_before or not board.is_capture(move) or not lines_before[0].get("pv"):
            return None
        best = lines_before[0]["pv"][0]
        if best == move:
            return None
        best_cp = lines_before[0]["score"].pov(board.turn).score(mate_score=MATE_SCORE)
        after = board.copy()
        after.push(move)
        reply = self.analyse(after, ZWISCHENZUG_CHECK[0], 1, ZWISCHENZUG_CHECK[1])
        reply_cp = reply[0]["score"].pov(after.turn).score(mate_score=MATE_SCORE) if reply else None
        if best_cp is None or reply_cp is None:
            return None
        played_cp = -reply_cp
        gap = best_cp - played_cp
        if gap <= ZWISCHENZUG_MIN_CP_GAP:
            return None
        after_best = board.copy()
        after_best.push(best)
        motifs = self._motifs(board, after_best, best, None)
        if not (after_best.is_check() or any(m["tactic"] not in ("None", "Hanging Piece") for m in motifs)):
            return None
        details = (f"A powerful intermediate move was missed. Instead of the simple recapture {board.san(move)} "
                   f"(eval: {played_cp / 100.0:+.2f}), the move {board.san(best)} (eval: {best_cp / 100.0:+.2f}) "
                   "creates a significant tactical advantage.")
        return {"tactic": "Missed Zwischenzug", "details": details, "value": gap}

    def _opponent_choice(self, board: chess.Board) -> Optional[dict]:
        """After the move, the opponent can choose between two good captures."""
        captures = [m for m in board.legal_moves if board.is_capture(m)]
        if len(captures) < 2:
            return None
        gains = []
        for m in captures:
            try:
                gain = int(self.tactics.see(board, m, with_sequence=False))
            except Exception:
                gain = 0
            if gain > 0:
                gains.append((m, gain))
        if len(gains) < 2:
            return None
        gains.sort(key=lambda mg: mg[1], reverse=True)
        (a, _), (b, _) = gains[:2]
        them = board.turn
        scores = []
        for m in (a, b):
            after = board.copy()
            after.push(m)
            lines = self.analyse(after, OPPONENT_CHOICE_CHECK[0], 1, OPPONENT_CHOICE_CHECK[1])
            scores.append(lines[0]["score"].pov(them).score(mate_score=MATE_SCORE) if lines else None)
        if any(s is None or s < OPPONENT_CHOICE_MIN_CP for s in scores):
            return None
        sa, sb = scores
        details = (f"Your move gives the opponent a choice between two good captures: {board.san(a)} "
                   f"(eval for them: {sa / 100.0:+.2f}) and {board.san(b)} (eval for them: {sb / 100.0:+.2f}).")
        return {"tactic": "Opponent Choice", "details": details, "value": abs(sa - sb)}

    # -- one ply, one game --

    def analyze_ply(self, board: chess.Board, move: chess.Move, ply_index: int, total_plies: int,
                    opponent_class: Optional[str], opening_name: str, previous: Optional[chess.Move],
                    fast_pass: bool = True) -> PlyAnalysis:
        c = self.config
        depth, multipv, time_limit = (c.FAST_PASS_DEPTH, c.FAST_PASS_MULTIPV, c.FAST_PASS_TIME) if fast_pass \
            else (c.FULL_DEPTH, c.FULL_MULTIPV, c.FULL_TIME)
        if ply_index == 0:
            self.tactics.reset_state()
        after = board.copy()
        after.push(move)
        lines_before = self.analyse(board, depth, multipv, time_limit)
        lines_after = self.analyse(after, depth, multipv, time_limit)

        engine_lines = [self._line(board, line, previous) for line in lines_before]
        positional_before = self.positional.analyze_position(board)
        player_line = LineAnalysis(
            san=board.san(move), uci=[move.uci()], score_cp=_first_white_cp(lines_after),
            tactical_motifs=self._motifs(board, after, move, previous, ply_index),
            positional_analysis=self.positional.analyze_position(after),
        )
        for extra in (self._missed_zwischenzug(board, move, lines_before), self._opponent_choice(after)):
            if extra:
                player_line.tactical_motifs.insert(0, extra)
        follow_ups = [self._line(after, line, move) for line in lines_after]

        if lines_before and lines_after:
            cls, loss, cp_loss = classify_move(board, move, lines_before, lines_after[0], opponent_class,
                                               opening_book=self.opening_book, tablebases=self.tablebases)
        else:
            cls, loss, cp_loss = Constants.BLUNDER, 1.0, MATE_SCORE
        return PlyAnalysis(
            ply=ply_index + 1, move_number_str=str(ply_index // 2 + 1),
            player="White" if board.turn == chess.WHITE else "Black", move_san=board.san(move),
            evaluation_symbol=Constants.EVALUATION_SYMBOLS.get(cls, ""), fen_before=board.fen(),
            classification=cls, points_loss=loss, cp_loss=cp_loss, player_line=player_line,
            best_engine_line=engine_lines[0] if engine_lines else LineAnalysis(), best_engine_lines=engine_lines,
            best_follow_up=follow_ups[0] if follow_ups else LineAnalysis(), best_follow_up_lines=follow_ups,
            positional_analysis_before=positional_before, positional_analysis_after=player_line.positional_analysis,
            syzygy_wdl=self.probe_wdl(board),
            raw_engine_lines_for_display=[{"line": [m.uci() for m in line.get("pv", [])], "score": str(line["score"].pov(board.turn))}
                                          for line in lines_before],
            game_phase=game_phase(ply_index, total_plies), opening_name=opening_name,
            score_cp_white_before=_first_white_cp(lines_before), score_cp_white_after=_first_white_cp(lines_after),
        )

    def analyze_game(self, moves: Sequence[chess.Move], opening_name: str = "Unknown Opening",
                     on_progress: Optional[Callable[[int, int], None]] = None, total_plies: Optional[int] = None,
                     start: Optional[chess.Board] = None) -> list[PlyAnalysis]:
        """PlyAnalysis for each move, in order. on_progress(done, total) is called after each ply.
        total_plies defaults to len(moves); pass the full game length when analysing only its start."""
        total = total_plies or len(moves)
        board = start.copy() if start else chess.Board()
        positions = [board.copy()]
        for move in moves:
            board.push(move)
            positions.append(board.copy())
        c = self.config
        self.pool.analyse_many(positions, c.FAST_PASS_DEPTH, c.FAST_PASS_MULTIPV, c.FAST_PASS_TIME)  # fills the cache in parallel

        self._motif_memo.clear()
        plies: list[PlyAnalysis] = []
        opponent_class, previous = None, None
        for i, move in enumerate(moves):
            ply = self.analyze_ply(positions[i], move, i, total, opponent_class, opening_name, previous)
            plies.append(ply)
            opponent_class, previous = ply.classification, move
            if on_progress:
                on_progress(i + 1, len(moves))
        return plies
