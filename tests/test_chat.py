import chess
import chess.engine
import pytest

from ai_chess_coach.llm.chat import Coach, CoachChat, EngineTool
from ai_chess_coach.llm.client import GeminiClient
from ai_chess_coach.llm.grounding import references_for_ply
from fakes import FakeSDK, api_error, fake_analyse, fake_describe, plies_from_game, response
from golden_cases import load_position, stockfish_path
from test_grounding import OPERA

OPERA_FEN = "3rkb1r/p2nqppp/5n2/1B2p1B1/4P3/1Q6/PPP2PPP/2KR3R w k - 3 13"


def opera_ply_and_moves():
    plies = plies_from_game(OPERA + ["Rxd7"], alternatives={25: ["d1d7", "d8d7"]}, replies={25: ["d8d7"]})
    return plies[-1], [chess.Move.from_uci(p.player_line.uci[0]) for p in plies]


def test_tool_rejects_illegal_move_with_examples():
    result = EngineTool(fake_analyse).evaluate(OPERA_FEN, "Qh5")
    assert "Qh5 is not a legal move" in result["error"] and "Bxf6" in result["error"]


def test_tool_rejects_bad_fen():
    assert "Invalid FEN" in EngineTool(fake_analyse).evaluate("not a fen")["error"]


def test_tool_plays_hypothetical_move_and_records_it():
    tool = EngineTool(fake_analyse, fake_describe)
    result = tool.evaluate(OPERA_FEN, "Bxf6")
    assert result["side_to_move"] == "Black"
    assert result["eval_cp_side_to_move"] == 30 and result["eval_cp_white"] == -30
    assert "best_line_details" in result
    assert "Bxf6" in tool.evaluated.verified


@pytest.mark.skipif(not stockfish_path(), reason="Stockfish not found (set STOCKFISH_PATH)")
def test_tool_with_real_stockfish():
    board, move = load_position("opera_game_1858.pgn", 24, "Rxd7")
    with chess.engine.SimpleEngine.popen_uci(stockfish_path()) as engine:
        tool = EngineTool(lambda b: engine.analyse(b, chess.engine.Limit(depth=12)))
        result = tool.as_function()(board.fen(), "Rxd7")
    assert result["resulting_fen"].startswith("3rkb1r/p2Rqppp")
    assert result["best_line_san"].startswith("13...")
    assert isinstance(result["eval_cp_white"], int)


def test_chat_registers_tool_and_tool_calls_verify_moves():
    """The model calls the tool on Bxf6 (as automatic function calling would), then mentions it."""
    def model(message, config):
        evaluate_position = config.tools[0]
        assert evaluate_position.__name__ == "evaluate_position"
        evaluate_position(OPERA_FEN, "Bxf6")
        return response(text="Bxf6 also works, but Rxd7 is stronger.")

    ply, moves = opera_ply_and_moves()
    sdk = FakeSDK([model])
    answer = CoachChat(GeminiClient("k", "m", sdk_client=sdk), fake_analyse).answer("What about Bxf6?", {"fen_before": OPERA_FEN}, "", references_for_ply(ply, moves))
    assert answer.final_report.ok and not answer.retried
    assert "fen_before" in sdk.calls[0]["message"] and "What about Bxf6?" in sdk.calls[0]["message"]


def test_chat_retries_once_with_correction_then_succeeds():
    ply, moves = opera_ply_and_moves()
    sdk = FakeSDK([response(text="Play Qh5 or Bxf6!"), response(text="Rxd7 is the move.")])
    answer = CoachChat(GeminiClient("k", "m", sdk_client=sdk), fake_analyse).answer("Best move?", {}, "", references_for_ply(ply, moves))
    assert answer.retried and answer.first_report.illegal == ["Qh5"] and answer.first_report.unverified == ["Bxf6"]
    assert answer.final_report.ok and answer.text == "Rxd7 is the move."
    retry = sdk.calls[1]
    assert "Qh5" in retry["message"] and "Bxf6" in retry["message"]
    assert [c.role for c in retry["history"]] == ["user", "model"]
    assert retry["history"][1].parts[0].text == "Play Qh5 or Bxf6!"


def test_chat_flags_moves_still_ungrounded_after_retry():
    ply, moves = opera_ply_and_moves()
    sdk = FakeSDK([response(text="Qh5!"), response(text="I still like Qh5.")])
    answer = CoachChat(GeminiClient("k", "m", sdk_client=sdk), fake_analyse).answer("?", {}, "", references_for_ply(ply, moves))
    assert answer.final_report.illegal == ["Qh5"]
    assert answer.text.endswith("[Not verified by the engine: Qh5]")


def test_coach_turns_errors_into_messages():
    ply, moves = opera_ply_and_moves()
    per_day = {"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel"}]}
    coach = Coach("k", "m", fake_analyse, fake_describe, client=GeminiClient("k", "m", sdk_client=FakeSDK([api_error(429, "q", [per_day])])))
    assert "daily quota" in coach.answer("?", {}, "", ply, moves)
    coach = Coach("k", "m", fake_analyse, fake_describe, client=GeminiClient("k", "old-model", sdk_client=FakeSDK([api_error(404, "not found")])))
    assert "GEMINI_MODEL" in coach.answer("?", {}, "", ply, moves)
    slow = FakeSDK([api_error(504, "Deadline expired")] * 2)
    coach = Coach("k", "m", fake_analyse, fake_describe, client=GeminiClient("k", "slow-model", sdk_client=slow, sleep=lambda s: None))
    assert "too slow" in coach.answer("?", {}, "", ply, moves)


def test_moves_the_tool_rejected_are_not_flagged():
    """The coach correctly says Qb3 is illegal (the queen is already on b3): no flag, no retry."""
    def model(message, config):
        assert "error" in config.tools[0](OPERA_FEN, "Qb3")
        return response(text="13. Qb3 is not legal: the queen is already there. 13. Rxd7 was best.")

    ply, moves = opera_ply_and_moves()
    answer = CoachChat(GeminiClient("k", "m", sdk_client=FakeSDK([model])), fake_analyse).answer("What about Qb3?", {}, "", references_for_ply(ply, moves))
    assert answer.final_report.ok and not answer.retried


def test_chat_follows_the_language_setting():
    ply, moves = opera_ply_and_moves()
    sdk = FakeSDK([response(text="13. Rxd7 était le meilleur coup.")])
    coach = Coach("k", "m", fake_analyse, fake_describe, client=GeminiClient("k", "m", sdk_client=sdk))
    assert coach.answer("Quel était le meilleur coup ?", {}, "", ply, moves, "French").startswith("13. Rxd7")
    system = sdk.calls[0]["config"].system_instruction
    assert "Answer in French." in system and "English Standard Algebraic Notation" in system


@pytest.mark.parametrize("number, plies_played, next_move", [
    ("10.", 18, "Nxb5"),    # before White's 10th move, the sacrifice
    ("10...", 19, "cxb5"),  # before Black's 10th move
    ("13.", 24, "Rxd7"),    # the current position
])
def test_tool_finds_past_game_positions(number, plies_played, next_move):
    _, moves = opera_ply_and_moves()
    board = EngineTool(fake_analyse, game_moves=moves).game_position(number)
    assert board.san(moves[plies_played]) == next_move
    expected = chess.Board()
    for m in moves[:plies_played]:
        expected.push(m)
    assert board.fen() == expected.fen()


@pytest.mark.parametrize("number", ["14.", "99...", "ten", "0."])
def test_tool_rejects_move_numbers_outside_the_game(number):
    _, moves = opera_ply_and_moves()
    assert "No position for move number" in EngineTool(fake_analyse, game_moves=moves[:24]).evaluate(move_number=number)["error"]


def test_model_checks_a_past_sacrifice():
    """Asked about 10.Nxb5, the model evaluates the position before White's 10th move, as AFC would call it."""
    tool_results = []

    def model(message, config):
        tool_results.append(config.tools[0](move_number="10.", move_san="Nxb5"))
        return response(text="10.Nxb5 was sound: after 10...cxb5 11.Bxb5+ White's attack is worth the piece.")

    ply, moves = opera_ply_and_moves()
    sdk = FakeSDK([model])
    coach = Coach("k", "m", fake_analyse, fake_describe, client=GeminiClient("k", "m", sdk_client=sdk))
    text = coach.answer("Was 10.Nxb5 a sound sacrifice?", {"fen_before": OPERA_FEN}, "", ply, moves)
    assert "error" not in tool_results[0] and tool_results[0]["side_to_move"] == "Black"
    assert "Not verified" not in text and len(sdk.calls) == 1
    assert "move_number" in sdk.calls[0]["config"].system_instruction


def test_rated_game_moves_gives_the_apps_rating_for_moves_in_the_question():
    from ai_chess_coach.llm.chat import SYSTEM_PROMPT, rated_game_moves

    plies = plies_from_game("e4 e5 Nf3 Nc6 Bc4 Bc5 c3".split())
    plies[2].classification, plies[2].points_loss, plies[2].cp_loss = "Excellent", -0.008, -8   # 2.Nf3: noise, shown as 0
    plies[5].classification, plies[5].points_loss, plies[5].cp_loss = "Mistake", 0.2652, 272   # 3...Bc5
    rated = rated_game_moves("Was 2.Nf3 a good move? And 3...Bc5?", plies)
    assert rated == [
        {"move": "2. Nf3", "app_rating": "Excellent", "win_chance_lost_pct": 0.0, "centipawns_lost": 0},
        {"move": "3... Bc5", "app_rating": "Mistake", "win_chance_lost_pct": 26.5, "centipawns_lost": 272},
    ]
    assert [r["move"] for r in rated_game_moves("Why c3?", plies)] == ["4. c3"]   # bare move, one match in the game
    assert rated_game_moves("Was 3.Nf3 good?", plies) == []                     # wrong move number: not a game move
    assert "Stay consistent with that rating" in SYSTEM_PROMPT
