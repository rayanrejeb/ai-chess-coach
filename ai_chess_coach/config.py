# Adapted from Chess Analyzer by yusufelgen07 (MIT License).
# See THIRD_PARTY_NOTICES.md.
import logging
import os
import sys
from dataclasses import dataclass

log = logging.getLogger(__name__)


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS  # type: ignore[attr-defined]
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


@dataclass
class AnalysisConfig:
    """Configuration for chess analysis parameters."""
    KING_SAFETY_THRESHOLD: int = 2
    BAD_BISHOP_PAWN_RATIO: float = 0.6
    SPACE_ADVANTAGE_RATIO: float = 1.2
    MATE_SCORE: int = 10000
    FAST_PASS_DEPTH: int = 16
    FULL_DEPTH: int = 10
    FAST_PASS_MULTIPV: int = 1  # Analyze only the single best line
    FULL_MULTIPV: int = 1       # Analyze only the single best line
    FAST_PASS_TIME: float = 1.0
    FULL_TIME: float = 5.0


class Constants:
    """A namespace for application-wide constants to improve maintainability."""
    # --- File Paths ---
    CONFIG_FILE = "config.json"
    ANALYSIS_DIR = "analyzed_games"
    GAMES_DIR = "fetched_games"

    # --- API & Network ---
    LICHESS_API_URL = "https://lichess.org/api"
    CHESSDOTCOM_USER_AGENT = "AI Chess Coach"

    # --- Move Classifications ---
    BRILLIANT = "Brilliant"
    GREAT_MOVE = "Great Move"
    BEST = "Best"
    EXCELLENT = "Excellent"
    GOOD = "Good"
    BOOK_MOVE = "Book Move"
    INACCURACY = "Inaccuracy"
    MISTAKE = "Mistake"
    BLUNDER = "Blunder"
    MISS = "Miss"
    CLASSIFICATION_ORDER = [
        BRILLIANT, GREAT_MOVE, BEST, EXCELLENT, GOOD,
        BOOK_MOVE, INACCURACY, MISTAKE, BLUNDER, MISS
    ]

    # --- Evaluation Tags ---
    EVALUATION_SYMBOLS = {
        BRILLIANT: "!!",
        GREAT_MOVE: "!",
        BEST: "!",
        EXCELLENT: "!",
        GOOD: "",
        BOOK_MOVE: "⩗",
        INACCURACY: "?!",
        MISTAKE: "?",
        BLUNDER: "??",
        MISS: "??"
    }

    # --- UI Colors ---
    ARROW_PLAYER_MOVE = "#FBFB2B"
    ARROW_BEST_MOVE = "#27C539"
    ARROW_FOLLOW_UP = "#FF6347"
    RTL_LANGUAGES = ["Arabic", "Hebrew", "Persian", "Urdu"]

