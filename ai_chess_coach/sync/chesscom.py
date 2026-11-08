"""Chess.com: monthly archives. A past month is fetched once more after it ends, then never again.

State: {"complete_through": "YYYY/MM"}, the last month of the contiguous run of complete months.
"""

from datetime import datetime, timedelta, timezone

ARCHIVES_URL = "https://api.chess.com/pub/player/{}/games/archives"
ARCHIVE_SETTLE = timedelta(days=1)  # margin after a month ends before its archive counts as final


def _month_key(day):
    return f"{day.year:04d}/{day.month:02d}"


def bootstrap_state(games):
    """State for a games file saved before sync state existed: months before the newest game are complete."""
    if not games:
        return {}
    newest = datetime.fromtimestamp(max(g.get("end_time", 0) for g in games), timezone.utc)
    return {"complete_through": _month_key(newest.replace(day=1) - timedelta(days=1))}


def iter_new_games(http, username, state, now=None):
    """Yield the games of every month not marked complete, oldest first, one list per month.

    `state` is updated before each yield, so saving it then is consistent with the games yielded."""
    now = now or datetime.now(timezone.utc)
    archives = http.get(ARCHIVES_URL.format(username.lower())).json().get("archives", [])
    all_complete_so_far = True
    for url in sorted(archives):
        year, month = map(int, url.rstrip("/").split("/")[-2:])
        key = f"{year:04d}/{month:02d}"
        if key <= state.get("complete_through", ""):
            continue
        games = http.get(url).json().get("games", [])
        month_end = datetime(year + month // 12, month % 12 + 1, 1, tzinfo=timezone.utc)
        all_complete_so_far = all_complete_so_far and now >= month_end + ARCHIVE_SETTLE
        if all_complete_so_far:
            state["complete_through"] = key
        yield games
