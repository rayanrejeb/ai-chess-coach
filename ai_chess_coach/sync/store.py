"""Saved games: {user}_{platform}_games.json (unchanged format) and {user}_{platform}_sync.json (state)."""

import json
import os


def _path(games_dir, username, platform, kind):
    return os.path.join(games_dir, f"{username.lower()}_{platform}_{kind}.json")


def _read(path, default):
    """A missing file gives `default`; a corrupt one raises, so it is never overwritten with less data."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def _write(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_games(games_dir, username, platform):
    return _read(_path(games_dir, username, platform, "games"), [])


def load_state(games_dir, username, platform):
    """None when no sync state was saved yet."""
    return _read(_path(games_dir, username, platform, "sync"), None)


def save(games_dir, username, platform, games, state):
    """Games first: a crash in between leaves the old state, which only refetches games already saved."""
    os.makedirs(games_dir, exist_ok=True)
    _write(_path(games_dir, username, platform, "games"), games)
    _write(_path(games_dir, username, platform, "sync"), state)
