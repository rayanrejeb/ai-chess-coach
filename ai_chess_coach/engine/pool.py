"""A small pool of Stockfish processes behind an evaluation cache.

Callers on different threads (game analysis, the score bar, the coach's engine tool) each borrow an
idle process instead of queueing behind one engine. Results are cached by (FEN, depth, multipv).
"""

import logging
import queue
import threading
from collections import OrderedDict
from concurrent.futures import CancelledError, ThreadPoolExecutor
from typing import Optional, Sequence

import chess
import chess.engine

log = logging.getLogger(__name__)

# python-chess raises either of these when the engine process dies, depending on timing
_ENGINE_DIED = (chess.engine.EngineTerminatedError, CancelledError)

POOL_SIZE = 2
MAX_POOL_SIZE = 4
CACHE_SIZE = 4096  # analyses kept, least recently used dropped first


class EnginePool:
    """Starting the pool launches the processes: call it off the UI thread. Always close() it."""

    def __init__(self, path: str, size: int = POOL_SIZE, cache_size: int = CACHE_SIZE):
        if not 1 <= size <= MAX_POOL_SIZE:
            raise ValueError(f"pool size must be 1..{MAX_POOL_SIZE}, got {size}")
        self._path = path
        self._engines: list[chess.engine.SimpleEngine] = []
        self._idle: "queue.Queue[chess.engine.SimpleEngine]" = queue.Queue()
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()
        self._closed = False
        self.hits = self.misses = 0
        try:
            for _ in range(size):
                self._add_engine()
        except Exception:
            self.close()
            raise
        self.name = self._engines[0].id.get("name", "UCI engine")

    @property
    def size(self) -> int:
        return len(self._engines)

    def _add_engine(self) -> chess.engine.SimpleEngine:
        engine = chess.engine.SimpleEngine.popen_uci(self._path)
        self._engines.append(engine)
        self._idle.put(engine)
        return engine

    def analyse(self, board: chess.Board, depth: int, multipv: int = 1, time_limit: Optional[float] = None) -> list:
        """Engine lines (python-chess InfoDicts, best first), or [] if the engine failed.
        The returned list may be shared with other callers: do not modify it."""
        if self._closed:
            return []
        key = (board.fen(), depth, multipv)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self.hits += 1
                return self._cache[key]
            self.misses += 1
        lines = self._run(board, chess.engine.Limit(depth=depth, time=time_limit), multipv)
        if lines:
            with self._lock:
                self._cache[key] = lines
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
        return lines

    def analyse_many(self, boards: Sequence[chess.Board], depth: int, multipv: int = 1, time_limit: Optional[float] = None) -> list:
        """analyse() for several positions, spread over the pool's processes."""
        with ThreadPoolExecutor(max_workers=self.size) as executor:
            return list(executor.map(lambda b: self.analyse(b, depth, multipv, time_limit), boards))

    def _run(self, board: chess.Board, limit: chess.engine.Limit, multipv: int) -> list:
        engine = self._idle.get()
        try:
            try:
                return engine.analyse(board, limit, multipv=multipv)
            except _ENGINE_DIED:
                log.error("Stockfish exited unexpectedly; restarting it")
                engine = self._restart(engine)
                return engine.analyse(board, limit, multipv=multipv)
        except Exception as e:
            log.error("Engine analysis failed: %s", e)
            return []
        finally:
            self._idle.put(engine)  # a dead engine goes back too, so the next caller retries the restart

    def _restart(self, dead: chess.engine.SimpleEngine) -> chess.engine.SimpleEngine:
        try:
            dead.quit()
        except Exception:
            pass
        engine = chess.engine.SimpleEngine.popen_uci(self._path)
        with self._lock:
            self._engines[self._engines.index(dead)] = engine
        return engine

    def close(self) -> None:
        self._closed = True
        for engine in self._engines:
            try:
                engine.quit()
            except Exception:
                pass
