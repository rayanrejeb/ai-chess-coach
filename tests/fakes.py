"""Stand-ins for the google-genai SDK and the engine, so LLM tests run without a key or Stockfish."""

from types import SimpleNamespace

import chess
import chess.engine
from google.genai import errors

from ai_chess_coach.models import LineAnalysis, PlyAnalysis


def response(text=None, parsed=None, finish_reason="STOP"):
    return SimpleNamespace(text=text, parsed=parsed, prompt_feedback=None, candidates=[SimpleNamespace(finish_reason=finish_reason)])


def api_error(code, message="error", details=()):
    body = {"error": {"code": code, "message": message, "details": list(details)}}
    return errors.ServerError(code, body) if code >= 500 else errors.ClientError(code, body)


class FakeSDK:
    """Plays back scripted outcomes: a response object, an exception, or a callable(message, config) -> response."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.models = SimpleNamespace(generate_content=self._generate)
        self.chats = SimpleNamespace(create=self._create_chat)

    def _next(self, *args):
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome(*args) if callable(outcome) else outcome

    def _generate(self, model, contents, config):
        self.calls.append({"kind": "generate", "model": model, "contents": contents, "config": config})
        return self._next(contents, config)

    def _create_chat(self, model, config, history):
        call = {"kind": "chat", "model": model, "config": config, "history": history}
        self.calls.append(call)

        def send_message(message):
            call["message"] = message
            return self._next(message, config)

        return SimpleNamespace(send_message=send_message)


def fake_analyse(board):
    """Deterministic 'engine': first legal moves in UCI order as the line, +0.30 for the side to move."""
    pv, b = [], board.copy()
    for _ in range(3):
        moves = sorted(b.legal_moves, key=lambda m: m.uci())
        if not moves:
            break
        pv.append(moves[0])
        b.push(moves[0])
    return {"score": chess.engine.PovScore(chess.engine.Cp(30), board.turn), "pv": pv}


def fake_describe(fen, uci):
    board = chess.Board(fen)
    return {"san": board.variation_san([chess.Move.from_uci(u) for u in uci])}


def plies_from_game(game_san, alternatives=None, replies=None):
    """PlyAnalysis list for a SAN move list. alternatives/replies: {ply: [uci, ...]} engine lines."""
    board, plies = chess.Board(), []
    for i, san in enumerate(game_san):
        move = board.parse_san(san)
        after = board.copy()
        after.push(move)
        ply = i + 1
        best = (alternatives or {}).get(ply, [move.uci()])
        follow = (replies or {}).get(ply, [])
        plies.append(PlyAnalysis(
            ply=ply, player="White" if board.turn else "Black", move_san=san, fen_before=board.fen(),
            classification="Good", player_line=LineAnalysis(san=san, uci=[move.uci()]),
            best_engine_line=LineAnalysis(uci=best), best_engine_lines=[LineAnalysis(uci=best)],
            best_follow_up=LineAnalysis(uci=follow), best_follow_up_lines=[LineAnalysis(uci=follow)] if follow else [],
        ))
        board.push(move)
    return plies
