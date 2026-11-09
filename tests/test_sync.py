"""sync/ on a fake HTTP session: incremental fetching, 429 handling, errors, old games files."""

import json
from datetime import datetime, timezone

import pytest

from ai_chess_coach import sync as sync_pkg
from ai_chess_coach.sync import Http, SyncError, chesscom, store, sync

USER = "SomeUser"
ARCHIVES = "https://api.chess.com/pub/player/someuser/games/archives"
LICHESS = "https://lichess.org/api/games/user/SomeUser"


def month_url(key):
    return f"https://api.chess.com/pub/player/someuser/games/{key}"


class FakeResponse:
    def __init__(self, status=200, data=None, lines=()):
        self.status_code, self.ok, self.data, self.lines = status, status < 400, data, lines

    def json(self):
        return self.data

    def iter_lines(self):
        return iter(self.lines)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


class FakeSession:
    """routes: url -> list of responses, served in order (the last one repeats)."""

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, headers=None, params=None, timeout=None, stream=False):
        self.calls.append({"url": url, "headers": headers, "params": params})
        queue = self.routes[url]
        return queue.pop(0) if len(queue) > 1 else queue[0]


def cc_game(n, end_time):
    return {"url": f"https://www.chess.com/game/live/{n}", "pgn": "1. e4 *", "end_time": end_time,
            "white": {"username": "a"}, "black": {"username": "b"}, "time_control": "600"}


def ts(year, month, day=15):
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp())


def li_line(game_id, created, status="mate", speed="blitz"):
    game = {"id": game_id, "createdAt": created, "status": status, "speed": speed, "pgn": "1. e4 *",
            "players": {"white": {"user": {"name": "a"}}, "black": {"user": {"name": "b"}}},
            "clock": {"initial": 600, "increment": 0}}
    return json.dumps(game).encode()


def run(session, tmp_path, platform="chess.com", **kwargs):
    http = Http(session, sleep=lambda s: None)
    return sync(USER, platform, str(tmp_path), http=http, **kwargs)


NOW = datetime(2026, 11, 8, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def fixed_now(monkeypatch):
    real = chesscom.iter_new_games
    monkeypatch.setattr(chesscom, "iter_new_games", lambda http, user, state: real(http, user, state, now=NOW))


def test_chesscom_second_sync_fetches_only_the_archive_list_and_current_month(tmp_path):
    routes = {
        ARCHIVES: [FakeResponse(data={"archives": [month_url("2026/11"), month_url("2026/09"), month_url("2026/10")]})],
        month_url("2026/09"): [FakeResponse(data={"games": [cc_game(1, ts(2026, 9))]})],
        month_url("2026/10"): [FakeResponse(data={"games": [cc_game(2, ts(2026, 10))]})],
        month_url("2026/11"): [FakeResponse(data={"games": [cc_game(3, ts(2026, 11, 2))]}),
                               FakeResponse(data={"games": [cc_game(3, ts(2026, 11, 2)), cc_game(4, ts(2026, 11, 7))]})],
    }
    session = FakeSession(routes)
    first = run(session, tmp_path)
    assert (first.new, first.total, first.requests) == (3, 3, 4)
    assert all(c["headers"]["User-Agent"] == sync_pkg.USER_AGENT for c in session.calls)

    session.calls.clear()
    second = run(session, tmp_path)
    assert (second.new, second.total, second.requests) == (1, 4, 2)
    assert [c["url"] for c in session.calls] == [ARCHIVES, month_url("2026/11")]
    saved = store.load_games(str(tmp_path), USER, "chess.com")
    assert [g["url"][-1] for g in saved] == ["1", "2", "3", "4"]


def test_chesscom_error_keeps_the_months_already_saved(tmp_path):
    routes = {
        ARCHIVES: [FakeResponse(data={"archives": [month_url("2026/09"), month_url("2026/10")]})],
        month_url("2026/09"): [FakeResponse(data={"games": [cc_game(1, ts(2026, 9))]})],
        month_url("2026/10"): [FakeResponse(status=500), FakeResponse(data={"games": [cc_game(2, ts(2026, 10))]})],
    }
    session = FakeSession(routes)
    with pytest.raises(SyncError) as error:
        run(session, tmp_path)
    assert USER.lower() not in str(error.value).lower()
    assert len(store.load_games(str(tmp_path), USER, "chess.com")) == 1
    assert store.load_state(str(tmp_path), USER, "chess.com") == {"complete_through": "2026/09"}

    session.calls.clear()
    assert run(session, tmp_path).total == 2
    assert [c["url"] for c in session.calls] == [ARCHIVES, month_url("2026/10")]


def test_rate_limit_waits_at_least_60_s_and_retries_once(tmp_path):
    waits = []
    session = FakeSession({ARCHIVES: [FakeResponse(status=429), FakeResponse(data={"archives": []})]})
    assert Http(session, sleep=waits.append).get(ARCHIVES).ok
    assert waits == [sync_pkg.RATE_LIMIT_WAIT_S] and sync_pkg.RATE_LIMIT_WAIT_S >= 60

    session = FakeSession({LICHESS: [FakeResponse(status=429)]})
    http = Http(session, sleep=waits.append)
    with pytest.raises(SyncError, match="HTTP 429"):
        http.get(LICHESS)
    assert http.count == 2


def test_lichess_skips_ongoing_games_and_keeps_the_cursor_on_them(tmp_path):
    first_lines = [li_line("g1", 1000), li_line("g2", 2000, "started"), li_line("g3", 3000), li_line("g4", 4000, "aborted")]
    second_lines = [li_line("g2", 2000, "resign"), li_line("g3", 3000), li_line("g5", 5000)]
    session = FakeSession({LICHESS: [FakeResponse(lines=first_lines), FakeResponse(lines=second_lines)]})

    first = run(session, tmp_path, "lichess")
    assert (first.new, first.total, first.requests) == (2, 2, 1)
    assert session.calls[0]["params"] == {"since": 0, "sort": "dateAsc", "ongoing": "true", "pgnInJson": "true"}
    assert "Authorization" not in session.calls[0]["headers"]
    assert store.load_state(str(tmp_path), USER, "lichess") == {"since": 2000}

    second = run(session, tmp_path, "lichess", token="lip_secret")
    assert (second.new, second.total, second.requests) == (2, 4, 1)
    assert session.calls[1]["params"]["since"] == 2000
    assert session.calls[1]["headers"]["Authorization"] == "Bearer lip_secret"
    assert store.load_state(str(tmp_path), USER, "lichess") == {"since": 5001}
    saved = store.load_games(str(tmp_path), USER, "lichess")
    assert [g["url"] for g in saved] == [f"https://lichess.org/g{i}" for i in (1, 2, 3, 5)]
    assert saved[0] == {"url": "https://lichess.org/g1", "pgn": "1. e4 *", "createdAt": 1000,
                        "white": {"username": "a"}, "black": {"username": "b"}, "time_control": "600+0"}


def test_ongoing_correspondence_game_does_not_hold_the_lichess_cursor(tmp_path):
    lines = [li_line("g1", 1000), li_line("corr", 2000, "started", "correspondence"), li_line("g3", 3000)]
    result = run(FakeSession({LICHESS: [FakeResponse(lines=lines)]}), tmp_path, "lichess")
    assert (result.new, result.total) == (2, 2)
    assert store.load_state(str(tmp_path), USER, "lichess") == {"since": 3001}


def test_lichess_saves_every_batch_before_a_stream_error(tmp_path, monkeypatch):
    monkeypatch.setattr(sync_pkg.lichess, "BATCH_SIZE", 2)

    class Broken(FakeResponse):
        def iter_lines(self):
            yield from [li_line("g1", 1000), li_line("g2", 2000), li_line("g3", 3000)]
            raise sync_pkg.requests.exceptions.ChunkedEncodingError("connection broken")

    with pytest.raises(SyncError, match="network error"):
        run(FakeSession({LICHESS: [Broken()]}), tmp_path, "lichess")
    assert len(store.load_games(str(tmp_path), USER, "lichess")) == 2
    assert store.load_state(str(tmp_path), USER, "lichess") == {"since": 2001}


def test_old_games_files_without_sync_state_are_continued(tmp_path):
    """Games files written before sync/ existed: same format, and only newer games are fetched."""
    old_cc = [cc_game(1, ts(2026, 9)), cc_game(2, ts(2026, 10, 20))]
    old_li = [{"url": "https://lichess.org/g1", "pgn": "1. e4 *", "createdAt": 1000,
               "white": {"username": "a"}, "black": {"username": "b"}, "time_control": "600+0"}]
    (tmp_path / "someuser_chess.com_games.json").write_text(json.dumps(old_cc), encoding="utf-8")
    (tmp_path / "someuser_lichess_games.json").write_text(json.dumps(old_li), encoding="utf-8")

    routes = {
        ARCHIVES: [FakeResponse(data={"archives": [month_url("2026/09"), month_url("2026/10"), month_url("2026/11")]})],
        month_url("2026/10"): [FakeResponse(data={"games": [cc_game(2, ts(2026, 10, 20)), cc_game(3, ts(2026, 10, 30))]})],
        month_url("2026/11"): [FakeResponse(data={"games": [cc_game(4, ts(2026, 11, 2))]})],
        LICHESS: [FakeResponse(lines=[li_line("g2", 2000)])],
    }
    session = FakeSession(routes)
    assert run(session, tmp_path).total == 4
    assert month_url("2026/09") not in [c["url"] for c in session.calls]
    assert run(session, tmp_path, "lichess").total == 2
    assert session.calls[-1]["params"]["since"] == 1001
    assert store.load_games(str(tmp_path), USER, "chess.com")[:2] == old_cc


def test_corrupt_games_file_is_not_overwritten(tmp_path):
    path = tmp_path / "someuser_lichess_games.json"
    path.write_text("[{broken", encoding="utf-8")
    with pytest.raises(ValueError):
        run(FakeSession({}), tmp_path, "lichess")
    assert path.read_text(encoding="utf-8") == "[{broken"
