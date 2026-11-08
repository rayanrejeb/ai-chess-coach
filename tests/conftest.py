"""Load .env when present, so local runs find STOCKFISH_PATH. Without it (CI) the Stockfish tests skip."""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
