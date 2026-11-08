import chess
import pytest

from ai_chess_coach.llm.grounding import References, check, extract_moves, normalize_san, references_for_ply
from fakes import plies_from_game

OPERA = "e4 e5 Nf3 d6 d4 Bg4 dxe5 Bxf3 Qxf3 dxe5 Bc4 Nf6 Qb3 Qe7 Nc3 c6 Bg5 b5 Nxb5 cxb5 Bxb5+ Nbd7 O-O-O Rd8".split()


@pytest.mark.parametrize("text, expected", [
    ("13.Rxd7! Rxd7 14.Rd1", ["Rxd7", "Rd1"]),
    ("after 13...Rxd7 and 13… Qe6", ["Rxd7", "Qe6"]),
    ("16.Qb8+!! Nxb8 17.Rd8#", ["Qb8", "Nxb8", "Rd8"]),
    ("Bxb5?! was risky, Nf3?? lost, exd5!? is interesting", ["Bxb5", "Nf3", "exd5"]),
    ("castle with O-O-O or 0-0", ["O-O-O", "O-O"]),
    ("promote with e8=Q+ or exd8Q", ["e8=Q", "exd8=Q"]),
    ("Nbd7 and R1e2 are fine", ["Nbd7", "R1e2"]),
    ("play e4, then d4", ["e4", "d4"]),
    ("Rxd7 then Rxd7 again", ["Rxd7"]),
])
def test_extract_moves(text, expected):
    assert extract_moves(text) == expected


@pytest.mark.parametrize("text", [
    "the knight on f3 is strong",
    "control of the d5 square",
    "the bishop goes to c4",
    "Cf3 et Fxb5",          # French piece letters are not SAN
    "e2e4 in UCI",          # UCI is not SAN
    "Be careful, Queen side",
])
def test_extract_ignores_non_moves(text):
    assert extract_moves(text) == []


def test_normalize_san():
    assert normalize_san("Qb8+!!") == "Qb8"
    assert normalize_san("0-0-0") == "O-O-O"
    assert normalize_san("e8Q") == "e8=Q"


def _opera_refs():
    plies = plies_from_game(OPERA + ["Rxd7"], alternatives={25: ["d1d7", "d8d7", "h1d1"]}, replies={25: ["d8d7", "h1d1"]})
    game_moves = [chess.Move.from_uci(p.player_line.uci[0]) for p in plies]
    return references_for_ply(plies[-1], game_moves)


def test_check_separates_verified_unverified_illegal():
    report = check("Rxd7 is best, Rxd7 Rd1 follows. Bxf6 also exists. Qh5 wins.", _opera_refs())
    assert report.moves == ["Rxd7", "Rd1", "Bxf6", "Qh5"]
    assert report.unverified == ["Bxf6"]     # legal before the move, but no engine line has it
    assert report.illegal == ["Qh5"]         # the queen on b3 cannot reach h5
    assert not report.ok


def test_moves_played_in_the_game_are_verified():
    # 12.O-O-O and 6.Bxf3 are not legal now, but they were played in the game.
    report = check("Your 12.O-O-O was strong and 4...Bxf3 gave up the bishop.", _opera_refs())
    assert report.ok, report


@pytest.mark.parametrize("text, illegal, unverified", [
    ("Earlier, 7.Qb3 was active.", [], []),          # right number: the game move
    ("Now 13.Qb3 is strong.", ["Qb3"], []),          # wrong number: the queen is already on b3
    ("White should play Qb3.", ["Qb3"], []),         # no number: normal check, illegal here
    ("Black answered 7...Qb3.", ["Qb3"], []),        # Black's 7th move was Qe7
    ("Bc4 was the plan.", [], ["Bc4"]),              # game move 6.Bc4, bare: legal now (Bb5-c4) but not engine-backed
])
def test_game_moves_need_the_right_move_number(text, illegal, unverified):
    report = check(text, _opera_refs())
    assert (report.illegal, report.unverified) == (illegal, unverified)


def test_extra_references_count_as_verified():
    tool_refs = References()
    tool_refs.add_line(_opera_refs().positions[0].fen(), ["b3a4"])
    report = check("What about Qa4?", _opera_refs(), tool_refs)
    assert report.ok
