"""Lichess: one streamed request for every game created since the cursor, oldest first.

State: {"since": ms}. Games still in progress are not saved; the cursor stays at the earliest real-time
one, so the next sync fetches it again once it is over (games already saved are deduplicated by URL).
Ongoing correspondence games do not hold the cursor: they can last for months or never end.
# LIMITATION: a correspondence game that finishes after the cursor has moved past it is never saved.
"""

import json

GAMES_URL = "https://lichess.org/api/games/user/{}"
BATCH_SIZE = 100  # games per save
ONGOING = {"created", "started"}
NOT_PLAYED = {"aborted", "noStart"}


def bootstrap_state(games):
    """State for a games file saved before sync state existed: continue after the newest game."""
    return {"since": max(g.get("createdAt", 0) for g in games) + 1} if games else {}


def _entry(game):
    """The saved format (same keys as before sync state existed)."""
    players, clock = game.get("players", {}), game.get("clock", {})

    def name(color):
        return players.get(color, {}).get("user", {}).get("name", color.title())

    return {
        "url": f"https://lichess.org/{game['id']}",
        "pgn": game.get("pgn", ""),
        "createdAt": game.get("createdAt", 0),
        "white": {"username": name("white")},
        "black": {"username": name("black")},
        "time_control": f"{clock.get('initial', '?')}+{clock.get('increment', '?')}",
    }


def iter_new_games(http, username, state, token=None):
    """Yield finished games in batches of BATCH_SIZE. `state` is updated before each yield."""
    headers = {"Accept": "application/x-ndjson"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    params = {"since": state.get("since", 0), "sort": "dateAsc", "ongoing": "true", "pgnInJson": "true"}
    earliest_ongoing = None
    batch = []
    with http.get(GAMES_URL.format(username), headers=headers, params=params, stream=True) as response:
        for line in response.iter_lines():
            if not line:
                continue
            game = json.loads(line)
            created = game.get("createdAt", 0)
            if game.get("status") in ONGOING:
                if earliest_ongoing is None and game.get("speed") != "correspondence":
                    earliest_ongoing = created
            elif game.get("status") not in NOT_PLAYED:
                batch.append(_entry(game))
            state["since"] = created + 1 if earliest_ongoing is None else earliest_ongoing
            if len(batch) >= BATCH_SIZE:
                yield batch
                batch = []
    yield batch
