"""python -m ai_chess_coach.sync [run|count]

run: sync both platforms. count: count the saved games per platform (no network).
Usernames: CHESSCOM_USERNAME / LICHESS_USERNAME, else DEFAULT_USERNAME. Lichess token: LICHESS_TOKEN (optional).
Usernames and the token are never printed.
"""

import argparse
import logging
import os
import time

from dotenv import load_dotenv

from ai_chess_coach.config import Constants
from ai_chess_coach.sync import PLATFORMS, Http, SyncError, store, sync

USERNAME_VARS = {"chess.com": "CHESSCOM_USERNAME", "lichess": "LICHESS_USERNAME"}


def main(argv=None):
    load_dotenv()
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="python -m ai_chess_coach.sync")
    parser.add_argument("command", choices=["run", "count"])
    command = parser.parse_args(argv).command
    total = 0
    for platform in PLATFORMS:
        username = os.getenv(USERNAME_VARS[platform]) or os.getenv("DEFAULT_USERNAME")
        if not username:
            print(f"{platform}: no username set")
            continue
        if command == "count":
            n = len(store.load_games(Constants.GAMES_DIR, username, platform))
            print(f"{platform}: {n} games")
        else:
            http, start = Http(), time.perf_counter()
            try:
                token = os.getenv("LICHESS_TOKEN") if platform == "lichess" else None
                result = sync(username, platform, Constants.GAMES_DIR, token=token, http=http)
                n = result.total
                print(f"{platform}: {result.new} new, {n} total, {http.count} requests, {time.perf_counter() - start:.2f} s")
            except SyncError as e:
                n = len(store.load_games(Constants.GAMES_DIR, username, platform))
                print(f"{platform}: FAILED ({e}), {n} saved, {http.count} requests, {time.perf_counter() - start:.2f} s")
        total += n
    print(f"total: {total} games")


if __name__ == "__main__":
    main()
