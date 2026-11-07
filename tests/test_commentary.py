import json
import logging

import chess

from ai_chess_coach.llm.client import GeminiClient
from ai_chess_coach.llm.commentary import (
    SYSTEM_PROMPT, CommentaryBatch, CommentaryContext, GameSummary, MoveComment,
    build_ply_payload, chunk_payloads, generate_game_commentary,
)
from fakes import FakeSDK, api_error, fake_describe, plies_from_game, response

GAME = "e4 e5 Nf3 Nc6 Bc4 Bc5".split()
CTX = CommentaryContext(language="French", username="student", player_color="white")


def test_prompt_requires_english_san_in_any_language():
    prompt = SYSTEM_PROMPT.format(language="French")
    assert "Write explanations in French" in prompt
    assert "English Standard Algebraic Notation" in prompt and "Never translate piece letters" in prompt


def test_chunk_payloads_respects_ply_and_size_limits():
    payloads = [{"ply": i, "pad": "x" * 100} for i in range(10)]
    assert [len(c) for c in chunk_payloads(payloads, max_plies=4, max_chars=10_000)] == [4, 4, 2]
    one_size = len(json.dumps(payloads[0]))
    assert [len(c) for c in chunk_payloads(payloads, max_plies=99, max_chars=one_size * 3)] == [3, 3, 3, 1]


def test_payload_drops_repeated_positional_notes():
    plies = plies_from_game(GAME, alternatives={3: ["b1c3"]})
    reported = set()
    payload = build_ply_payload(plies[2], fake_describe, reported)
    assert payload["move"] == "Nf3" and payload["engine_alternatives"] == [{"san": "2. Nc3"}]
    plies[2].positional_analysis_after.advantages = ["space advantage"]
    first = build_ply_payload(plies[2], fake_describe, set())
    seen = {"space advantage"}
    second = build_ply_payload(plies[2], fake_describe, seen)
    assert first["positional_notes"] == ["space advantage"] and second["positional_notes"] == []


def _batch_for(contents, *, extra_text=""):
    """Fake model: comment on every ply in the prompt; mention the played move (+ extra text)."""
    data = json.loads(contents.split("Moves to comment on:\n", 1)[1])
    last = "This is the last part" in contents
    return response(parsed=CommentaryBatch(
        comments=[MoveComment(ply=p["ply"], move_san=p["move"], move_commentary=f"{p['move']} develops. {extra_text}") for p in data],
        summary=GameSummary(white_takeaway="ok") if last else None,
    ))


def test_commentary_chunks_and_summary_from_last_part(monkeypatch):
    monkeypatch.setattr("ai_chess_coach.llm.commentary.MAX_PLIES_PER_CHUNK", 4)
    plies = plies_from_game(GAME)
    sdk = FakeSDK([lambda c, cfg: _batch_for(c), lambda c, cfg: _batch_for(c)])
    statuses = []
    comments, summary = generate_game_commentary(GeminiClient("k", "m", sdk_client=sdk), plies, CTX, fake_describe, on_status=statuses.append)
    assert sorted(comments, key=int) == ["1", "2", "3", "4", "5", "6"]
    assert summary == {"white_takeaway": "ok", "black_takeaway": "", "white_pattern": "", "black_pattern": ""}
    assert statuses == ["AI Part 1/2...", "AI Part 2/2..."]
    assert all(not c.illegal_moves and not c.unverified_moves for c in comments.values())


def test_commentary_flags_and_logs_ungrounded_moves(caplog):
    plies = plies_from_game(GAME)
    game_moves = [chess.Move.from_uci(p.player_line.uci[0]) for p in plies]
    sdk = FakeSDK([lambda c, cfg: _batch_for(c, extra_text="Qh5 was possible, and so was d3.")])
    with caplog.at_level(logging.INFO, logger="ai_chess_coach.llm.commentary"):
        comments, _ = generate_game_commentary(GeminiClient("k", "m", sdk_client=sdk), plies, CTX, fake_describe, game_moves)
    first = comments["1"]   # before 1.e4, Qh5 is illegal and d3 is legal but unverified
    assert first.illegal_moves == ["Qh5"] and first.unverified_moves == ["d3"]
    assert "grounding commentary ply=1 illegal=['Qh5'] unverified=['d3']" in caplog.text


def test_commentary_stops_on_daily_quota(monkeypatch):
    monkeypatch.setattr("ai_chess_coach.llm.commentary.MAX_PLIES_PER_CHUNK", 2)
    per_day = {"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel"}]}
    sdk = FakeSDK([lambda c, cfg: _batch_for(c), api_error(429, "quota", [per_day])])
    comments, summary = generate_game_commentary(GeminiClient("k", "m", sdk_client=sdk, sleep=lambda s: None), plies_from_game(GAME), CTX, fake_describe)
    assert sorted(comments) == ["1", "2"] and summary is None
    assert len(sdk.calls) == 2   # the third part was never requested
