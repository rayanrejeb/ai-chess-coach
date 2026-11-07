# Adapted from Chess Analyzer by yusufelgen07 (MIT License).
# See THIRD_PARTY_NOTICES.md.
import logging
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, Any

log = logging.getLogger(__name__)


@dataclass
class PositionalFeatures:
    """Structured positional analysis results."""
    material_balance: float = 0.0
    king_safety: Dict[str, Any] = field(default_factory=dict)
    pawn_structure: Dict[str, List[str]] = field(default_factory=dict)
    rook_activity: Dict[str, List[str]] = field(default_factory=dict)
    knight_outposts: List[str] = field(default_factory=list)
    bad_bishops: List[str] = field(default_factory=list)
    has_bishop_pair: Dict[str, bool] = field(default_factory=dict)
    advantages: List[str] = field(default_factory=list)


Classification = str


@dataclass
class LineAnalysis:
    """A structured container for all data related to a single engine line (PV)."""
    san: str = "N/A"
    uci: List[str] = field(default_factory=list)
    score_cp: int = 0
    tactical_motifs: List[Dict[str, str]] = field(default_factory=list)
    positional_analysis: PositionalFeatures = field(default_factory=PositionalFeatures)


@dataclass
class PlyAnalysis:
    """A structured container for all data related to a single ply (a half-move)."""
    ply: int = 0
    move_number_str: str = ""
    player: str = ""
    move_san: str = ""
    evaluation_symbol: str = ""
    fen_before: str = ""
    classification: Classification = ""
    points_loss: float = 0.0
    cp_loss: int = 0
    player_line: LineAnalysis = field(default_factory=LineAnalysis)
    best_engine_line: LineAnalysis = field(default_factory=LineAnalysis)
    best_follow_up: LineAnalysis = field(default_factory=LineAnalysis)
    best_engine_lines: List[LineAnalysis] = field(default_factory=list)
    best_follow_up_lines: List[LineAnalysis] = field(default_factory=list)
    positional_analysis_before: PositionalFeatures = field(default_factory=PositionalFeatures)
    positional_analysis_after: PositionalFeatures = field(default_factory=PositionalFeatures)
    syzygy_wdl: Optional[int] = None
    raw_engine_lines_for_display: List[Dict[str, Any]] = field(default_factory=list)
    game_phase: str = "middlegame"
    opening_name: str = "Unknown Opening"
    score_cp_white_before: int = 0
    score_cp_white_after: int = 0


@dataclass
class GeminiCommentary:
    """Holds structured commentary from Gemini to allow for safe BiDi text rendering."""
    ply: int = 0
    move_san: str = ""
    move_commentary: str = "No AI comment available for this move."
    missed_opportunities: List[str] = field(default_factory=list)
    best_replies: List[str] = field(default_factory=list)
    illegal_moves: List[str] = field(default_factory=list)      # set by the grounding check
    unverified_moves: List[str] = field(default_factory=list)   # set by the grounding check

    @classmethod
    def from_dict(cls, data: dict):
        """Factory method to create an instance, ignoring unexpected keys."""
        known_keys = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in known_keys}
        return cls(**filtered_data)

