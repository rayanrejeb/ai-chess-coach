import os
import threading

import chess
import pytest

from ai_chess_coach.engine.pool import EnginePool
from golden_cases import stockfish_path

pytestmark = pytest.mark.skipif(not stockfish_path(), reason="Stockfish not found (set STOCKFISH_PATH)")


@pytest.fixture
def pool():
    p = EnginePool(stockfish_path(), size=2)
    yield p
    p.close()


def test_rejects_bad_sizes():
    with pytest.raises(ValueError):
        EnginePool(stockfish_path(), size=5)


def test_cache_returns_the_same_analysis(pool):
    first = pool.analyse(chess.Board(), depth=8)
    assert first and first[0]["pv"]
    assert pool.analyse(chess.Board(), depth=8) is first
    assert (pool.hits, pool.misses) == (1, 1)
    pool.analyse(chess.Board(), depth=9)  # another depth is another entry
    assert pool.misses == 2


def test_concurrent_callers_share_the_pool(pool):
    boards = []
    b = chess.Board()
    for san in "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6".split():
        b.push_san(san)
        boards.append(b.copy())
    results = [None] * len(boards)

    def work(i):
        results[i] = pool.analyse(boards[i], depth=8)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(len(boards))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(r and r[0]["pv"] for r in results)
    assert pool.analyse_many(boards, depth=8) == results  # all cached now


def test_restarts_a_crashed_engine():
    pool = EnginePool(stockfish_path(), size=1)
    try:
        os.kill(pool._engines[0].transport.get_pid(), 9)
        assert pool.analyse(chess.Board(), depth=6)
    finally:
        pool.close()


def test_close_stops_every_process():
    pool = EnginePool(stockfish_path(), size=2)
    engines = list(pool._engines)
    pool.close()
    assert all(e.transport.get_returncode() is not None for e in engines)
    assert pool.analyse(chess.Board(), depth=6) == []
