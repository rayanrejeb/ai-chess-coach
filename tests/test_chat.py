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
