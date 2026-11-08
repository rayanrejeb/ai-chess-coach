"""Incremental game sync for Chess.com and Lichess.

sync() loads the saved games, asks the platform module for new ones batch by batch, and saves the
games and the sync state after every batch, so an error keeps everything fetched so far.
"""

import logging
import threading
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

import requests

from . import chesscom, lichess, store

log = logging.getLogger(__name__)
logging.getLogger("urllib3").setLevel(logging.WARNING)  # its DEBUG lines contain URLs with the username

USER_AGENT = "AI Chess Coach (+https://github.com/rayanrejeb/ai-chess-coach)"
TIMEOUT_S = 30
RATE_LIMIT_WAIT_S = 60  # Lichess asks clients to wait a full minute after a 429; used for both sites
PLATFORMS = {"chess.com": chesscom, "lichess": lichess}

_lock = threading.Lock()  # one request at a time, even if two syncs are started


class SyncError(Exception):
    """A sync failed. The message never contains the username or the token."""


class Http:
    """Counts requests, sends the User-Agent, waits and retries once after an HTTP 429."""

    def __init__(self, session=None, sleep=time.sleep):
        self.session = session or requests.Session()
        self.sleep = sleep
        self.count = 0

    def get(self, url, headers=None, **kwargs):
        headers = {"User-Agent": USER_AGENT, **(headers or {})}
        host = urlsplit(url).hostname
        for attempt in range(2):
            self.count += 1
            response = self.session.get(url, headers=headers, timeout=TIMEOUT_S, **kwargs)
            if response.status_code != 429 or attempt:
                break
            response.close()
            log.warning("%s rate limit (HTTP 429), waiting %d s before retrying", host, RATE_LIMIT_WAIT_S)
            self.sleep(RATE_LIMIT_WAIT_S)
        if not response.ok:
            response.close()
            raise SyncError(f"{host} answered HTTP {response.status_code}")
        return response


@dataclass
class SyncResult:
    new: int
    total: int
    requests: int


def _sort_key(game):
    return game.get("end_time", game.get("createdAt", 0))


def sync(username, platform, games_dir, token=None, status=lambda message: None, http=None):
    """Fetch the games of `username` on `platform` ("chess.com" or "lichess") that are not saved yet.

    `token` (Lichess only) is optional and sent when set. Raises SyncError; saved data is kept."""
    source = PLATFORMS[platform]
    http = http or Http()
    with _lock:
        games = store.load_games(games_dir, username, platform)
        state = store.load_state(games_dir, username, platform)
        if state is None:
            state = source.bootstrap_state(games)
        urls = {g.get("url") for g in games}
        new = 0
        batches = lichess.iter_new_games(http, username, state, token) if source is lichess else chesscom.iter_new_games(http, username, state)
        try:
            for batch in batches:
                fresh = [g for g in batch if g.get("url") not in urls]
                urls.update(g.get("url") for g in fresh)
                games += fresh
                new += len(fresh)
                games.sort(key=_sort_key)
                store.save(games_dir, username, platform, games, state)
                status(f"{platform}: {len(games)} games saved ({new} new)")
        except requests.RequestException as e:  # its message would contain the URL, so the username
            raise SyncError(f"{platform}: network error ({type(e).__name__})") from None
    log.info("%s sync: %d new games, %d total, %d requests", platform, new, len(games), http.count)
    return SyncResult(new, len(games), http.count)
