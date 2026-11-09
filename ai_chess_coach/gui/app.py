# Adapted from Chess Analyzer by yusufelgen07 (MIT License).
# See THIRD_PARTY_NOTICES.md.
import chess
import chess.engine
import chess.pgn
import chess.polyglot
import chess.svg
import chess.syzygy
import customtkinter as ctk
import io
import json
import logging
import os
import re
import threading
import tkinter as tk
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import asdict
from tkinter import filedialog, font as tkfont, messagebox
from typing import Dict, List, Optional
from PIL import Image

from ai_chess_coach.config import Constants, resource_path
from ai_chess_coach.engine.classify import MATE_SCORE, win_probability
from ai_chess_coach.engine.game import GameAnalyzer, find_key_moments, san_line
from ai_chess_coach.engine.pool import EnginePool
from ai_chess_coach.llm.chat import Coach, move_rating, rated_game_moves
from ai_chess_coach.llm.client import DEFAULT_MODEL
from ai_chess_coach.models import GeminiCommentary, LineAnalysis, PlyAnalysis, PositionalFeatures
from ai_chess_coach.sync import SyncError, store, sync

log = logging.getLogger(__name__)


class ChessGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.analyzer: Optional[GameAnalyzer] = None
        self.coach: Optional[Coach] = None
        self._engine_generation = 0  # bumped when the engine is restarted or the app closes
        self._engine_future: Optional[Future] = None
        self.stockfish_path: Optional[str] = None
        self.opening_book_path: Optional[str] = None
        self.syzygy_path: Optional[str] = None
        self.lichess_token: Optional[str] = None
        self.gemini_model, self.gemini_language = DEFAULT_MODEL, "English"
        self.moves: List[chess.Move] = []
        self.analysis_data: List[PlyAnalysis] = []
        self.games: List[dict] = []
        self.gemini_data: Dict[str, GeminiCommentary] = {}
        self.game_summary: Optional[dict] = None
        self.current_ply = 0
        self.player_stats = {"white": {}, "black": {}}
        self.player_accuracy = {"white": 0.0, "black": 0.0}
        self.analysis_mode = "game"
        self.engine_line_board: Optional[chess.Board] = None
        self.engine_line_moves: List[str] = []
        self.current_engine_line_ply = 0
        self._resize_job = None
        self._last_board_cache = {}
        self._max_board_cache = 32
        self._ui_executor = ThreadPoolExecutor(max_workers=2)
        self._nav_generation = 0

        if not os.path.exists(Constants.ANALYSIS_DIR): os.makedirs(Constants.ANALYSIS_DIR)
        if not os.path.exists(Constants.GAMES_DIR): os.makedirs(Constants.GAMES_DIR)

        self.setup_ui()
        self.load_config_and_initialize()
        self.bind("<Configure>", self._on_resize)

    def setup_ui(self):
        self.title("AI Chess Coach")
        self.geometry("1400x950")
        try:
            self.iconphoto(True, tk.PhotoImage(file=resource_path('Chess.ico')))
        except tk.TclError:
            log.warning("Could not load application icon.")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.score_bar_frame = ctk.CTkFrame(self, width=50)
        self.score_bar_frame.grid(row=0, column=0, padx=(10, 0), pady=10, sticky="ns")
        self.board_frame = ctk.CTkFrame(self)
        self.board_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.right_panel = ctk.CTkFrame(self, width=450)
        self.right_panel.grid(row=0, column=2, padx=(0, 10), pady=10, sticky="ns")
        
        self.score_label = ctk.CTkLabel(self.score_bar_frame, text="", font=("Arial", 16, "bold"))
        self.score_label.pack(pady=10)
        self.score_bar = ctk.CTkProgressBar(self.score_bar_frame, orientation="vertical", progress_color="white")
        self.score_bar.pack(fill="y", expand=True, padx=5, pady=5)
        
        self.board_label = ctk.CTkLabel(self.board_frame, text="")
        self.board_label.pack(expand=True)

        self.right_panel.grid_columnconfigure(0, weight=1)
        self.right_panel.grid_rowconfigure(4, weight=1)

        input_frame = ctk.CTkFrame(self.right_panel)
        input_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        input_frame.grid_columnconfigure(0, weight=1)
        self.platform_var = tk.StringVar(value="Chess.com")
        ctk.CTkLabel(input_frame, text="Platform:").grid(row=0, column=0, sticky="w")
        platform_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        platform_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 5))
        ctk.CTkRadioButton(platform_frame, text="Chess.com", variable=self.platform_var, value="Chess.com").pack(side="left")
        ctk.CTkRadioButton(platform_frame, text="Lichess", variable=self.platform_var, value="Lichess").pack(side="left", padx=10)
        self.username_entry = ctk.CTkEntry(input_frame, placeholder_text="Username")
        self.username_entry.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 5))
        self.fetch_btn = ctk.CTkButton(input_frame, text="Fetch Games", command=self.fetch_games)
        self.fetch_btn.grid(row=3, column=0, sticky="ew", padx=(0, 5))
        self.settings_btn = ctk.CTkButton(input_frame, text="Settings", command=self.open_settings_window)
        self.settings_btn.grid(row=3, column=1, sticky="ew", padx=(5, 0))

        self.game_combo = ctk.CTkComboBox(self.right_panel, values=["Fetch games to see list"], state="disabled", command=self.on_game_select)
        self.game_combo.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        self.game_combo._entry.bind("<KeyRelease>", self._filter_game_list)

        analysis_btns_frame = ctk.CTkFrame(self.right_panel)
        analysis_btns_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        analysis_btns_frame.grid_columnconfigure((0, 1), weight=1)
        self.analyze_btn = ctk.CTkButton(analysis_btns_frame, text="Analyze Selected Game", command=self.analyze_game, state="disabled")
        self.analyze_btn.grid(row=0, column=0, sticky="ew", padx=(0, 2))
        self.reanalyze_btn = ctk.CTkButton(analysis_btns_frame, text="Re-analyze", command=lambda: self.analyze_game(force_reanalyze=True), state="disabled")
        self.reanalyze_btn.grid(row=0, column=1, sticky="ew", padx=(2, 0))

        self.key_moments_frame = ctk.CTkFrame(self.right_panel, fg_color=("gray80", "gray25"), height=60)
        self.key_moments_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=2)
        self.key_moments_frame.pack_propagate(False)
        ctk.CTkLabel(self.key_moments_frame, text="Key Moments", font=("Arial", 12, "bold")).pack(pady=(3, 0))
        self.key_moments_container = ctk.CTkFrame(self.key_moments_frame, fg_color="transparent")
        self.key_moments_container.pack(fill="x", padx=5, pady=(0, 3))

        self.main_analysis_frame = ctk.CTkScrollableFrame(self.right_panel)
        self.main_analysis_frame.grid(row=4, column=0, sticky="nsew", padx=10, pady=(0, 5))
        self.main_analysis_frame.grid_columnconfigure(0, weight=1)
        self.summary_frame = ctk.CTkFrame(self.main_analysis_frame, fg_color=("gray85", "gray20"))
        ctk.CTkLabel(self.summary_frame, text="Game Summary & Takeaways", font=("Arial", 14, "bold")).pack(pady=5, padx=10)
        self.white_summary_label = ctk.CTkLabel(self.summary_frame, text="", justify="left", wraplength=380, anchor="w")
        self.white_summary_label.pack(fill="x", padx=10, pady=(0, 5))
        self.black_summary_label = ctk.CTkLabel(self.summary_frame, text="", justify="left", wraplength=380, anchor="w")
        self.black_summary_label.pack(fill="x", padx=10, pady=(0, 10))
        self.analysis_title_label = ctk.CTkLabel(self.main_analysis_frame, text="", font=("Arial", 16, "bold"), anchor="w")
        self.gemini_textbox = ctk.CTkTextbox(self.main_analysis_frame, wrap="word", font=("Arial", 16), state="disabled", height=250)
        self.stats_title = ctk.CTkLabel(self.main_analysis_frame, text="Move Classification", font=("Arial", 14, "bold"))
        self.stats_frame = ctk.CTkFrame(self.main_analysis_frame, fg_color="transparent")
        self.stats_frame.grid_columnconfigure((0, 1), weight=1)
        self.white_stats_label = ctk.CTkLabel(self.stats_frame, text="White: -", font=("Courier", 12), justify="left", anchor="nw")
        self.white_stats_label.grid(row=0, column=0, sticky="nsew", padx=5)
        self.black_stats_label = ctk.CTkLabel(self.stats_frame, text="Black: -", font=("Courier", 12), justify="left", anchor="nw")
        self.black_stats_label.grid(row=0, column=1, sticky="nsew", padx=5)
        self.lines_title = ctk.CTkLabel(self.main_analysis_frame, text="Top Engine Lines", font=("Arial", 14, "bold"))
        self.engine_lines_frame = ctk.CTkFrame(self.main_analysis_frame, fg_color="transparent")
        self.explored_line_label = ctk.CTkLabel(self.main_analysis_frame, text="", font=("Arial", 12, "italic"), wraplength=400)

        nav_frame = ctk.CTkFrame(self.right_panel)
        nav_frame.grid(row=5, column=0, sticky="ew", padx=10, pady=5)
        nav_frame.grid_columnconfigure(list(range(3)), weight=1)
        self.prev_btn = ctk.CTkButton(nav_frame, text="< Prev", command=self.prev_move, state="disabled")
        self.prev_btn.grid(row=0, column=0, sticky="ew")
        self.show_follow_up_btn = ctk.CTkButton(nav_frame, text="Follow-up", command=self.show_follow_up, state="disabled")
        self.show_follow_up_btn.grid(row=0, column=1, sticky="ew")
        self.next_btn = ctk.CTkButton(nav_frame, text="Next >", command=self.next_move, state="disabled")
        self.next_btn.grid(row=0, column=2, sticky="ew")
        self.toggle_engine_line_btn = ctk.CTkButton(nav_frame, text="Explore Top Line", command=self._toggle_engine_line_mode, state="disabled")
        self.toggle_engine_line_btn.grid(row=1, column=0, columnspan=3, sticky="ew")

        chat_frame = ctk.CTkFrame(self.right_panel)
        chat_frame.grid(row=6, column=0, sticky="ew", padx=10, pady=(10, 10))
        chat_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(chat_frame, text="Ask about the game:", font=("Arial", 14, "bold")).grid(row=0, column=0, sticky="w", padx=10, pady=(5,0))
        self.chat_history = ctk.CTkTextbox(chat_frame, wrap="word", font=("Arial", 14), height=120, state="disabled")
        self.chat_history.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        # Same family and (already DPI-scaled) size as the textbox, in bold. CTkTextbox.tag_config refuses fonts.
        self._chat_bold_font = tkfont.Font(self, font=self.chat_history._textbox.cget("font"))
        self._chat_bold_font.configure(weight="bold")  # Font() ignores options when given font=
        self.chat_history._textbox.tag_configure("bold", font=self._chat_bold_font)
        chat_input_frame = ctk.CTkFrame(chat_frame, fg_color="transparent")
        chat_input_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 5))
        chat_input_frame.grid_columnconfigure(0, weight=1)
        self.chat_entry = ctk.CTkEntry(chat_input_frame, placeholder_text="E.g., What if I played Ng5 instead?")
        self.chat_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.chat_entry.bind("<Return>", lambda event: self.send_chat_message())
        self.send_chat_btn = ctk.CTkButton(chat_input_frame, text="Send", width=60, command=self.send_chat_message)
        self.send_chat_btn.grid(row=0, column=1, sticky="e")
        self.update_display()

    def send_chat_message(self, event=None):
        question = self.chat_entry.get()
        if not question or not self.analyzer or not self.analysis_data: return
        self.chat_entry.delete(0, "end")
        self._display_chat_message("You", question)
        self.send_chat_btn.configure(state="disabled", text="...")
        analysis = self.analysis_data[self.current_ply - 1] if self.current_ply > 0 else self.analysis_data[0]
        board_after_player_move = chess.Board(analysis.fen_before)
        if self.current_ply > 0:
            player_move_uci = analysis.player_line.uci[0]
            player_move_obj = chess.Move.from_uci(player_move_uci)
            board_after_player_move.push(player_move_obj)
        fen_after_player_move = board_after_player_move.fen()
        simplified_player_line = self.analyzer.describe_line(analysis.fen_before, analysis.player_line.uci)
        simplified_best_alternative_line = self.analyzer.describe_line(analysis.fen_before, analysis.best_engine_line.uci)
        simplified_best_opponent_follow_up = self.analyzer.describe_line(fen_after_player_move, analysis.best_follow_up.uci)
        game_context = {
            "ply": self.current_ply,
            "fen_before": analysis.fen_before,
            "move_info": {
                "move": f"{analysis.move_number_str}{'.' if analysis.player == 'White' else '...'} {analysis.move_san}",
                "classification": analysis.classification, "commentary": self._get_gemini_commentary_for_ply(self.current_ply),
                "analysis": simplified_player_line, **move_rating(analysis)
            },
            "rated_game_moves": rated_game_moves(question, self.analysis_data),
            "best_alternative_line": simplified_best_alternative_line,
            "best_opponent_follow_up": simplified_best_opponent_follow_up
        }
        log.debug("Chat context: %s", json.dumps(game_context, indent=2))
        chat_history = self.chat_history.get("1.0", "end").strip()
        future = self._ui_executor.submit(self._get_and_display_chat_response, question, game_context, chat_history, analysis)
        def chat_done_callback(fut: Future):
            try: fut.result()
            except Exception as e:
                log.exception(f"Chat thread failed: {e}")
                self.after(0, self._display_chat_message, "System Error", f"An error occurred in the chat task: {e}")
                self.after(0, lambda: self.send_chat_btn.configure(state="normal", text="Send"))
        future.add_done_callback(chat_done_callback)

    def _get_and_display_chat_response(self, question, game_context, chat_history, analysis):
        if not self.coach:
            response = "The AI chat model is not configured. Please check your Gemini API key in the settings."
        else:
            response = self.coach.answer(question, game_context, chat_history, analysis, self.moves, self.gemini_language)
        self.after(0, self._display_chat_message, "Coach", response)
        self.after(0, lambda: self.send_chat_btn.configure(state="normal", text="Send"))

    def _display_chat_message(self, sender: str, message: str):
        self.chat_history.configure(state="normal")
        self.chat_history.insert("end", f"{sender}: ")
        # Gemini marks emphasis as **bold**: the odd parts of the split are the bold ones.
        for i, part in enumerate(re.split(r"\*\*(.+?)\*\*", message.strip())):
            self.chat_history.insert("end", part, "bold" if i % 2 else None)
        self.chat_history.insert("end", "\n\n")
        self.chat_history.see("end")
        self.chat_history.configure(state="disabled")

    def go_to_ply(self, ply_number: int):
        if 0 <= ply_number <= len(self.moves):
            self.current_ply = ply_number
            self.update_display()

    def _show_moments_menu(self, event, moments):
        menu = tk.Menu(self, tearoff=0)
        for moment in moments:
            text = f"{moment.move_number_str}{'...' if moment.player == 'Black' else '.'} {moment.move_san} ({moment.classification})"
            menu.add_command(label=text, command=lambda p=moment.ply: self.go_to_ply(p))
        try: menu.tk_popup(event.x_root, event.y_root)
        finally: menu.grab_release()

    def _update_key_moments_display(self, key_moments: List[PlyAnalysis]):
        for widget in self.key_moments_container.winfo_children(): widget.destroy()
        if not key_moments:
            label = ctk.CTkLabel(self.key_moments_container, text="Key Moments: (No game loaded)", font=("Arial", 10), anchor="w", text_color="gray")
            label.pack(fill="x", padx=5)
        else:
            moments_texts = [f"{m.move_number_str}{'...' if m.player == 'Black' else '.'} {m.move_san}" for m in key_moments]
            label = ctk.CTkLabel(self.key_moments_container, text="Key Moments: " + "  |  ".join(moments_texts), font=("Arial", 10), justify="left", anchor="w", text_color=("#3B8ED0", "#2FA572"), cursor="hand2")
            label.pack(fill="x", padx=5)
            label.bind("<Button-1>", lambda e, m=key_moments: self._show_moments_menu(e, m))

    def _analyze_thread(self, pgn_data: str, game_url: str, username: str):
        try:
            game = chess.pgn.read_game(io.StringIO(pgn_data))
            if game is None: raise ValueError("Could not parse PGN data.")
            
            player_color = "white" if username.lower() == game.headers.get("White", "").lower() else "black"
            opening_name = game.headers.get("Opening", "Unknown Opening")
            # --- FIX: Use a local variable to prevent UI state desync ---
            local_moves = list(game.mainline_moves())

            self.after(0, lambda: self.analyze_btn.configure(text=f"Analyzing 0/{len(local_moves)}..."))
            def progress(done, total):
                self.after(0, lambda: self.analyze_btn.configure(text=f"Analyzing {done}/{total}..."))
            full_analysis: List[PlyAnalysis] = self.analyzer.analyze_game(local_moves, opening_name, on_progress=progress, start=game.board())
            player_stats = {"white": defaultdict(int), "black": defaultdict(int)}
            for ply_analysis in full_analysis:
                player_stats[ply_analysis.player.lower()][ply_analysis.classification] += 1

            gemini_comments, game_summary = {}, None

            if self.coach:
                self.after(0, lambda: self._set_navigation_enabled(False))
                status_callback = lambda text: self.after(0, lambda: self.analyze_btn.configure(text=text))

                gemini_comments, game_summary = self.coach.comment_game(full_analysis, self.gemini_language, username, player_color, local_moves, status_callback)
                
                # --- FIX 3: Display a popup if the AI completely fails ---
                if not gemini_comments:
                    self.after(0, lambda: messagebox.showwarning(
                        "AI Commentary Failed", 
                        "The engine finished analyzing, but no AI commentary was generated.\n\nThis usually happens because your Google Cloud API key ran out of daily free requests, the model returned empty output due to limits, or GEMINI_MODEL in .env names a model that is not available (see the log)."
                    ))
                # ---------------------------------------------------------
                
                self.after(0, lambda: self._set_navigation_enabled(True))
            
            key_moments = find_key_moments(full_analysis, 3)
            
            # --- FIX: Pass local_moves into finalize_analysis. 
            # This safely updates the UI's self.moves on the main thread only when done. ---
            self.after(0, self._finalize_analysis, local_moves, full_analysis, gemini_comments, game_summary, player_stats, game_url, key_moments)
            
        except Exception as e:
            log.exception("Game analysis failed")
            self.after(0, lambda e=e: self.display_message("Analysis Error", f"An unexpected error occurred: {e}"))
            self.after(0, self._reset_ui_after_error)

    def _on_resize(self, event=None):
        # The binding on the window also fires for every child widget, and update_display rebuilds some of them,
        # so reacting to those would redraw forever.
        if event is not None and event.widget is not self: return
        if self._resize_job: self.after_cancel(self._resize_job)
        self._resize_job = self.after(100, self.update_display)

    def _get_title_for_analysis(self, analysis: PlyAnalysis):
        next_player = "Black" if analysis.player == "White" else "White"
        move_notation = f"{analysis.move_number_str}{'...' if analysis.player == 'Black' else '.'} {analysis.move_san} {analysis.evaluation_symbol}"
        return f"{move_notation} ({analysis.classification}) - {next_player} to Move"

    def update_display(self, *args):
        if not self.analyzer:
            self.display_message("Welcome!", "Please go to Settings to configure your Stockfish engine and Gemini API key.")
            self.update_board(chess.Board())
            self.update_score_bar(None)
            return
        self.update_summary_display()
        if self.analysis_mode == "game": self._update_game_display()
        else: self._update_engine_line_display()

    def _update_game_display(self):
        arrows = []
        if self.current_ply == 0:
            board = chess.Board()
            title, commentary, engine_lines, board_fen, score = "Initial Position", "Click 'Next' to begin.", [], board.fen(), 0
            if self.analysis_data:
                engine_lines = self.analysis_data[0].raw_engine_lines_for_display
                for line in engine_lines[:3]:
                    if (line_uci_list := line.get('line')) and line_uci_list:
                        move = chess.Move.from_uci(line_uci_list[0])
                        arrows.append(chess.svg.Arrow(move.from_square, move.to_square, color=Constants.ARROW_BEST_MOVE))
            self._update_analysis_display(title=title, commentary=commentary, engine_lines=engine_lines, board_fen=board_fen, show_sections=True)
            self.update_score_bar(score)
            self.update_board(board, arrows)
        else:
            analysis = self.analysis_data[self.current_ply - 1]
            board_before_move = chess.Board(analysis.fen_before)
            player_move = self.moves[self.current_ply - 1]
            if player_move not in board_before_move.legal_moves:
                log.error(f"Illegal move detected in display. Move: {player_move.uci()}, FEN: {board_before_move.fen()}")
                return 
            arrows.append(chess.svg.Arrow(player_move.from_square, player_move.to_square, color=Constants.ARROW_PLAYER_MOVE))
            if analysis.raw_engine_lines_for_display:
                for line_data in analysis.raw_engine_lines_for_display[:3]:
                    if (line_uci_list := line_data.get('line')) and line_uci_list:
                        best_move = chess.Move.from_uci(line_uci_list[0])
                        if best_move != player_move:
                            arrows.append(chess.svg.Arrow(best_move.from_square, best_move.to_square, color=Constants.ARROW_BEST_MOVE))
            board_after_move = board_before_move.copy()
            board_after_move.push(player_move)
            self._update_analysis_display(title=self._get_title_for_analysis(analysis), commentary=self._get_gemini_commentary_for_ply(analysis.ply), engine_lines=analysis.raw_engine_lines_for_display, board_fen=analysis.fen_before, show_sections=True)
            self.update_score_bar(analysis.score_cp_white_after)
            self.update_board(board_after_move, arrows)
            self._refine_selected_ply(analysis, board_before_move, player_move)
        self.prev_btn.configure(state="normal" if self.current_ply > 0 else "disabled")
        self.next_btn.configure(state="normal" if self.current_ply < len(self.moves) else "disabled")
        self.show_follow_up_btn.configure(state="normal" if self.current_ply > 0 else "disabled")
        self.toggle_engine_line_btn.configure(state="normal" if self.analysis_data else "disabled")

    def _refine_selected_ply(self, analysis: PlyAnalysis, board_before_move: chess.Board, player_move: chess.Move):
        gen = self._nav_generation = self._nav_generation + 1
        def work():
            try:
                before = self.analyzer.analyse(board_before_move, self.analyzer.config.FULL_DEPTH, self.analyzer.config.FULL_MULTIPV, self.analyzer.config.FULL_TIME)
                after_board = board_before_move.copy(); after_board.push(player_move)
                after = self.analyzer.analyse(after_board, self.analyzer.config.FULL_DEPTH, self.analyzer.config.FULL_MULTIPV, self.analyzer.config.FULL_TIME)
                return before, after
            except Exception: return None, None
        def done(fut: Future):
            if gen != self._nav_generation: return
            try:
                before, after = fut.result()
                if not before or not after: return
                lines = [{"line": [m.uci() for m in l.get('pv', [])], "score": str(l['score'].pov(board_before_move.turn))} for l in before]
                # Redrawing refines again; with cached results that would loop forever, so redraw only on a change.
                if lines == analysis.raw_engine_lines_for_display: return
                analysis.raw_engine_lines_for_display = lines
                self.after(0, self.update_display)
            except Exception: pass
        fut = self._ui_executor.submit(work)
        fut.add_done_callback(done)

    def _get_gemini_commentary_for_ply(self, ply: int) -> str:
        gemini_move_data = self.gemini_data.get(str(ply))
        return gemini_move_data.move_commentary if gemini_move_data else "No AI comment available for this move."

    def _update_engine_line_display(self):
        if not self.engine_line_board: return
        board = self.engine_line_board.copy()
        for i in range(self.current_engine_line_ply):
            try: board.push(chess.Move.from_uci(self.engine_line_moves[i]))
            except: break
        analysis = self.analysis_data[self.current_ply - 1]
        self._update_analysis_display(title=self._get_title_for_analysis(analysis), commentary=self._get_gemini_commentary_for_ply(analysis.ply), engine_lines=analysis.raw_engine_lines_for_display, board_fen=analysis.fen_before)
        line_san = san_line([chess.Move.from_uci(m) for m in self.engine_line_moves], self.engine_line_board)
        self.explored_line_label.configure(text=f"Exploring: {' '.join(line_san)}")
        self.update_board(board)
        threading.Thread(target=self._get_and_update_score_for_board, args=(board.fen(),), daemon=True).start()
        self.prev_btn.configure(state="normal" if self.current_engine_line_ply > 0 else "disabled")
        self.next_btn.configure(state="normal" if self.current_engine_line_ply < len(self.engine_line_moves) else "disabled")
        self.show_follow_up_btn.configure(state="disabled")
        self.toggle_engine_line_btn.configure(state="normal")

    def _get_and_update_score_for_board(self, board_fen):
        try:
            board = chess.Board(board_fen)
            analysis = self.analyzer.analyse(board, 15, 1, 1.0)
            if analysis: self.after(0, self.update_score_bar, analysis[0]['score'].white().score(mate_score=MATE_SCORE))
        except Exception as e:
            log.warning(f"Error getting score for board: {e}")
            self.after(0, self.update_score_bar, None)

    def _update_analysis_display(self, title="", commentary="", engine_lines=None, board_fen=None, show_sections=True):
        self.analysis_title_label.pack(fill="x", padx=10, pady=(5, 0))
        self.gemini_textbox.pack(fill="both", expand=True, padx=10, pady=5)
        self.analysis_title_label.configure(text=title)
        self.gemini_textbox.configure(state="normal")
        self.gemini_textbox.delete("1.0", "end")
        self.gemini_textbox.insert("1.0", commentary)
        self.gemini_textbox.yview_moveto(0.0)
        justify_style = "right" if self.gemini_language in Constants.RTL_LANGUAGES else "left"
        self.gemini_textbox._textbox.tag_configure("justify_align", justify=justify_style)
        self.gemini_textbox._textbox.tag_add("justify_align", "1.0", "end")
        self.gemini_textbox.configure(state="disabled")
        for widget in [self.stats_title, self.stats_frame, self.lines_title, self.engine_lines_frame, self.explored_line_label]: widget.pack_forget()
        if show_sections and self.analysis_data:
            self.stats_title.pack(fill="x", padx=10, pady=(10, 5), after=self.gemini_textbox)
            self.stats_frame.pack(fill="x", padx=10, pady=(0, 5), after=self.stats_title)
            self.lines_title.pack(fill="x", padx=10, pady=(10, 5), after=self.stats_frame)
            self.engine_lines_frame.pack(fill="x", expand=True, padx=5, after=self.lines_title)
            self.explored_line_label.pack(fill="x", padx=10, pady=5, after=self.engine_lines_frame)
        for widget in self.engine_lines_frame.winfo_children(): widget.destroy()
        if engine_lines and board_fen:
            board = chess.Board(board_fen)
            for i, line_data in enumerate(engine_lines[:3]):
                if line_uci := line_data.get('line', []):
                    line_san = " ".join(san_line([chess.Move.from_uci(m) for m in line_uci], board))
                    btn = ctk.CTkButton(self.engine_lines_frame, text=f"{i+1}. {line_san}", fg_color="transparent", text_color=["#3B8ED0", "#2FA572"], hover_color=["#E5E5E5", "#1F1F1F"], anchor="w", command=lambda u=line_uci, b=board.copy(): self._explore_specific_line(u, b))
                    btn.pack(fill="x")
                    score_label = ctk.CTkLabel(self.engine_lines_frame, text=f"   Score: {line_data.get('score', 'N/A')}", font=("Courier", 11, "italic"))
                    score_label.pack(fill="x", padx=10)

    def update_summary_display(self):
        if isinstance(self.game_summary, dict) and self.analysis_data:
            self.summary_frame.pack(fill="x", padx=10, pady=(0, 10), before=self.analysis_title_label)
            self.white_summary_label.configure(text=f"⚪ Takeaway: {self.game_summary.get('white_takeaway', 'N/A')}\n⚪ Pattern: {self.game_summary.get('white_pattern', 'N/A')}")
            self.black_summary_label.configure(text=f"⚫ Takeaway: {self.game_summary.get('black_takeaway', 'N/A')}\n⚫ Pattern: {self.game_summary.get('black_pattern', 'N/A')}")
        else:
            self.summary_frame.pack_forget()

    def update_board(self, board: chess.Board, arrows=None):
        if not self.board_frame.winfo_exists(): return
        size = max(50, int(min(self.board_frame.winfo_width(), self.board_frame.winfo_height()) * 0.8))
        if size < 50: return
        arrows_sig = tuple((a.tail, a.head, getattr(a, "color", None)) for a in arrows) if arrows else None
        key = (board.fen(), size, arrows_sig)
        if key in self._last_board_cache:
            self.board_label.configure(image=self._last_board_cache[key], text="")
            return
        try:
            from cairosvg import svg2png
            svg = chess.svg.board(board=board, size=size, arrows=arrows or [])
            img_data = svg2png(bytestring=svg.encode('utf-8'))
            pil_img = Image.open(io.BytesIO(img_data))
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(size, size))
            self.board_label.configure(image=ctk_img, text="")
            self._last_board_cache[key] = ctk_img
            if len(self._last_board_cache) > self._max_board_cache:
                self._last_board_cache.pop(next(iter(self._last_board_cache)), None)
        except ImportError:
            self.board_label.configure(image=None, text="Board render error: Please install 'cairosvg' (`pip install cairosvg`)", wraplength=400)
        except Exception as e:
            self.board_label.configure(image=None, text=f"Board render error:\n{e}", wraplength=400)

    def update_score_bar(self, white_cp: Optional[int]):
        if white_cp is None:
            self.score_bar.set(0.5); self.score_label.configure(text="--")
            return
        try:
            mate_score = MATE_SCORE
            if abs(white_cp) > mate_score - 1000:
                mate_in = (mate_score + 1 - abs(white_cp)) // 2 + (mate_score + 1 - abs(white_cp)) % 2
                self.score_label.configure(text=f"M{mate_in}")
                score_val = 1.0 if white_cp > 0 else 0.0
            else:
                score_val = win_probability(white_cp)
                self.score_label.configure(text=f"{white_cp/100.0:+.2f}")
            self.score_bar.set(score_val)
        except: self.score_bar.set(0.5); self.score_label.configure(text="ERR")

    def fetch_games(self):
        username = self.username_entry.get().strip()
        if not username: return
        self.fetch_btn.configure(state="disabled", text="Fetching...")
        self.game_combo.configure(values=["Fetching..."], state="disabled")
        threading.Thread(target=self._fetch_games_thread, args=(username,), daemon=True).start()

    def _load_local_games(self, username, platform):
        try:
            return store.load_games(Constants.GAMES_DIR, username, platform.lower())
        except (OSError, ValueError) as e:
            log.warning(f"Could not read the saved games: {e}")
            return []

    def _fetch_games_thread(self, username):
        platform = self.platform_var.get()
        previous_count = len(self._load_local_games(username, platform))
        def status(message): self.after(0, lambda: self.fetch_btn.configure(text=message))
        try:
            sync(username, platform.lower(), Constants.GAMES_DIR, token=self.lichess_token, status=status)
        except (SyncError, OSError, ValueError) as e:  # games saved before the error are kept
            self.after(0, lambda err=e: self.display_message(f"{platform} Error", f"Error fetching games: {err}"))
        self.after(0, self._handle_fetched_games, self._load_local_games(username, platform), previous_count)

    def _handle_fetched_games(self, games, previous_game_count=None):
        self.fetch_btn.configure(state="normal", text="Fetch Games")
        self.games = list(reversed(games)) # Reverse to show most recent games first

        # --- NEW UI FEEDBACK LOGIC ---
        if previous_game_count is not None:
            new_games_count = len(self.games) - previous_game_count
            if new_games_count > 0:
                log.info(f"Added {new_games_count} new game(s).")
                messagebox.showinfo("Fetch Complete", f"Added {new_games_count} new game(s) to the list.")
            else:
                log.info("No new games found.")
                messagebox.showinfo("Fetch Complete", "No new games found. Your list is up to date.")
        # --- END OF NEW LOGIC ---

        if self.games:
            game_titles = [f"{g.get('white', {}).get('username', 'White')} vs {g.get('black', {}).get('username', 'Black')} ({g.get('time_control', 'N/A')})" for g in self.games]
            self.game_combo.configure(values=game_titles, state="normal")
            if game_titles: 
                self.game_combo.set(game_titles[0])
        else:
            self.game_combo.configure(values=[f"No games for {self.username_entry.get()}"], state="disabled")
            self.game_combo.set("") # Clear the text
        
        self.on_game_select()

    def _filter_game_list(self, event=None):
        search_term = self.game_combo.get().lower()
        if not self.games: return
        def title_for(g): return f"{g.get('white', {}).get('username', 'White')} vs {g.get('black', {}).get('username', 'Black')}"
        filtered_titles = [f"{title_for(g)} ({g.get('time_control', 'N/A')})" for g in self.games if search_term in title_for(g).lower()]
        if filtered_titles: self.game_combo.configure(values=filtered_titles)
        elif not search_term: self.game_combo.configure(values=[f"{title_for(g)} ({g.get('time_control', 'N/A')})" for g in self.games])
        else: self.game_combo.configure(values=["No results found"], state="disabled")

    def on_game_select(self, event=None):
        if not self.games or self.game_combo.get().startswith("No games"):
            self.analyze_btn.configure(state="disabled"); self.reanalyze_btn.configure(state="disabled")
            return
        game_url = self._get_selected_game_url()
        if not game_url: return
        filepath = self._get_analysis_filepath(game_url)
        self.analyze_btn.configure(state="normal" if self.analyzer else "disabled")
        if os.path.exists(filepath):
            self.analyze_btn.configure(text="Load Saved Analysis"); self.reanalyze_btn.configure(state="normal")
        else:
            self.analyze_btn.configure(text="Analyze Selected Game"); self.reanalyze_btn.configure(state="disabled")

    def analyze_game(self, force_reanalyze=False):
        if not self.analyzer:
            self.display_message("Analyzer Not Ready", "Cannot analyze: Please check settings.")
            return
        game_url = self._get_selected_game_url()
        if not game_url: return
        filepath = self._get_analysis_filepath(game_url)
        if os.path.exists(filepath) and not force_reanalyze:
            threading.Thread(target=self._load_analysis_from_file, args=(filepath,), daemon=True).start()
            return
        pgn_data = self._get_selected_game_pgn()
        if not pgn_data: return
        self.analyze_btn.configure(state="disabled", text="Analyzing...")
        self.reanalyze_btn.configure(state="disabled")
        self.fetch_btn.configure(state="disabled")
        threading.Thread(target=self._analyze_thread, args=(pgn_data, game_url, self.username_entry.get().strip()), daemon=True).start()

    def _calculate_accuracy(self, player_color: str):
        player_moves = [p for p in self.analysis_data if p.player.lower() == player_color.lower() and p.classification != Constants.BOOK_MOVE]
        if not player_moves: return 0.0
        return max(0, min(100, 100 * (1 - sum(p.points_loss for p in player_moves) / len(player_moves))))

    def _finalize_analysis(self, moves, analysis_data, gemini_data, game_summary, player_stats, game_url, key_moments):
        self.moves, self.analysis_data, self.gemini_data, self.game_summary = moves, analysis_data, gemini_data, game_summary
        self.player_stats = {k: dict(v) for k, v in player_stats.items()}
        self.player_accuracy["white"] = self._calculate_accuracy("white")
        self.player_accuracy["black"] = self._calculate_accuracy("black")
        self.current_ply = 0
        self._save_analysis_to_file(game_url)
        self.fetch_btn.configure(state="normal")
        self.on_game_select()
        self._update_key_moments_display(key_moments)
        self.update_stats()
        self.update_display()

    def _reset_ui_after_error(self):
        self.fetch_btn.configure(state="normal", text="Fetch Games")
        self.analyze_btn.configure(state="normal", text="Analyze Selected Game")
        self.reanalyze_btn.configure(state="disabled")
        self.on_game_select()

    def update_stats(self):
        white_stats, black_stats = self.player_stats.get('white', {}), self.player_stats.get('black', {})
        white_acc, black_acc = self.player_accuracy.get('white', 0.0), self.player_accuracy.get('black', 0.0)
        def format_stats(stats):
            lines = [f"{t:<12}: {stats.get(t, 0)}" for t in Constants.CLASSIFICATION_ORDER if stats.get(t, 0) > 0]
            return "\n".join(lines) if lines else "-"
        self.white_stats_label.configure(text=f"White (Acc: {white_acc:.1f}%)\n" + format_stats(white_stats))
        self.black_stats_label.configure(text=f"Black (Acc: {black_acc:.1f}%)\n" + format_stats(black_stats))

    def next_move(self):
        self.explored_line_label.configure(text="")
        if self.analysis_mode == "game":
            if self.current_ply < len(self.analysis_data): self.current_ply += 1
        elif self.current_engine_line_ply < len(self.engine_line_moves): self.current_engine_line_ply += 1
        self.update_display()

    def prev_move(self):
        self.explored_line_label.configure(text="")
        if self.analysis_mode == "game":
            if self.current_ply > 0: self.current_ply -= 1
        elif self.current_engine_line_ply > 0: self.current_engine_line_ply -= 1
        self.update_display()

    def show_follow_up(self):
        if self.current_ply > 0:
            analysis = self.analysis_data[self.current_ply - 1]
            if analysis.best_follow_up.uci:
                board_after_move = chess.Board(analysis.fen_before)
                try: board_after_move.push(self.moves[self.current_ply - 1])
                except: pass
                self._explore_specific_line(analysis.best_follow_up.uci, board_after_move)
            else: self.explored_line_label.configure(text="No follow-up line available.")

    def _toggle_engine_line_mode(self):
        if self.analysis_mode == "game":
            if self.current_ply > 0 and self.analysis_data:
                analysis = self.analysis_data[self.current_ply - 1]
                if analysis.best_engine_line.uci:
                    self._explore_specific_line(analysis.best_engine_line.uci, chess.Board(analysis.fen_before))
        else:
            self.analysis_mode = "game"
            self.toggle_engine_line_btn.configure(text="Explore Top Line")
            self.update_display()

    def _explore_specific_line(self, moves_uci_list, board_state):
        self.analysis_mode = "engine_line"
        self.toggle_engine_line_btn.configure(text="Return to Game")
        self.engine_line_board, self.engine_line_moves, self.current_engine_line_ply = board_state, moves_uci_list, 1
        self.update_display()

    def open_settings_window(self):
        settings_win = ctk.CTkToplevel(self)
        settings_win.title("Settings")
        settings_win.transient(self)
        settings_win.grab_set()

        def browse_file(e, t):
            p = filedialog.askopenfilename(title=t)
            if p: e.delete(0, "end"); e.insert(0, p)
        def browse_dir(e, t):
            p = filedialog.askdirectory(title=t)
            if p: e.delete(0, "end"); e.insert(0, p)

        ctk.CTkLabel(settings_win, text="Google Gemini API Key:").pack(pady=(10, 0), padx=20, anchor="w")
        key_entry = ctk.CTkEntry(settings_win, width=400, show="*"); key_entry.pack(pady=5, padx=20, fill="x")
        ctk.CTkLabel(settings_win, text="Gemini Model:").pack(pady=(10, 0), padx=20, anchor="w")
        model_menu = ctk.CTkOptionMenu(settings_win, values=[DEFAULT_MODEL]) ; model_menu.pack(pady=5, padx=20, fill="x")
        ctk.CTkLabel(settings_win, text="Commentary Language:").pack(pady=(10, 0), padx=20, anchor="w")
        lang_menu = ctk.CTkOptionMenu(settings_win, values=["English", "Spanish", "French", "German"]); lang_menu.pack(pady=5, padx=20, fill="x")
        ctk.CTkLabel(settings_win, text="Lichess API Token:").pack(pady=(10, 0), padx=20, anchor="w")
        lichess_entry = ctk.CTkEntry(settings_win, width=400, show="*"); lichess_entry.pack(pady=5, padx=20, fill="x")

        def create_browse_frame(parent, label_text, command):
            ctk.CTkLabel(parent, text=label_text).pack(pady=(10, 0), padx=20, anchor="w")
            frame = ctk.CTkFrame(parent, fg_color="transparent"); frame.pack(pady=5, padx=20, fill="x")
            entry = ctk.CTkEntry(frame); entry.pack(side="left", fill="x", expand=True)
            ctk.CTkButton(frame, text="Browse", width=80, command=lambda: command(entry, label_text)).pack(side="left", padx=(10, 0))
            return entry
        sf_entry = create_browse_frame(settings_win, "Stockfish Path:", lambda e, t: browse_file(e, "Select Stockfish"))
        book_entry = create_browse_frame(settings_win, "Opening Book Path (.bin):", lambda e, t: browse_file(e, "Select Polyglot Book"))
        syzygy_entry = create_browse_frame(settings_win, "Syzygy Tablebases Path (Folder):", lambda e, t: browse_dir(e, "Select Syzygy Folder"))

        try:
            with open(Constants.CONFIG_FILE, 'r', encoding='utf-8') as f: config = json.load(f)
            key_entry.insert(0, config.get("gemini_api_key", ""))
            model_menu.set(config.get("gemini_model", self.gemini_model))
            lang_menu.set(config.get("gemini_language", self.gemini_language))
            lichess_entry.insert(0, config.get("lichess_token", ""))
            sf_entry.insert(0, config.get("stockfish_path", ""))
            book_entry.insert(0, config.get("opening_book_path", ""))
            syzygy_entry.insert(0, config.get("syzygy_path", ""))
        except (FileNotFoundError, json.JSONDecodeError): pass

        def save_and_reinit(close=False):
            self.save_config(gemini_api_key=key_entry.get().strip(), gemini_model=model_menu.get(), gemini_language=lang_menu.get(), lichess_token=lichess_entry.get().strip(), stockfish_path=sf_entry.get().strip(), opening_book_path=book_entry.get().strip(), syzygy_path=syzygy_entry.get().strip())
            self.load_config_and_initialize()
            if close: settings_win.destroy()
            self.on_game_select(); self.update_display()
        btn_frame = ctk.CTkFrame(settings_win, fg_color="transparent"); btn_frame.pack(pady=20)
        ctk.CTkButton(btn_frame, text="Save", command=lambda: save_and_reinit(close=False)).pack(side="left", padx=10)
        ctk.CTkButton(btn_frame, text="Save and Close", command=lambda: save_and_reinit(close=True)).pack(side="left", padx=10)

    def load_config_and_initialize(self):
        try:
            try:
                with open(Constants.CONFIG_FILE, 'r', encoding='utf-8') as f: config = json.load(f)
            except FileNotFoundError:
                config = {}
            # Environment variables (.env) override config.json.
            env_overrides = {"gemini_api_key": "GEMINI_API_KEY", "gemini_model": "GEMINI_MODEL", "lichess_token": "LICHESS_TOKEN", "stockfish_path": "STOCKFISH_PATH"}
            config.update({key: os.environ[var] for key, var in env_overrides.items() if os.environ.get(var)})

            # Load configuration settings
            last_username = config.get("last_username") or os.environ.get("DEFAULT_USERNAME", "")
            last_platform = config.get("last_platform", "Chess.com")
            
            self.username_entry.delete(0, "end")
            self.username_entry.insert(0, last_username)
            self.platform_var.set(last_platform)
            
            self.gemini_model = config.get("gemini_model", DEFAULT_MODEL)
            self.gemini_language = config.get("gemini_language", "English")
            self.lichess_token = config.get("lichess_token")
            self.stockfish_path = config.get("stockfish_path")
            self.opening_book_path = config.get("opening_book_path")
            self.syzygy_path = config.get("syzygy_path")
            
            self.initialize_analyzer(config.get("gemini_api_key"))

            # --- THIS IS THE CRITICAL NEW LOGIC ---
            # After loading the last username, immediately load their games.
            log.info("Loading the saved games of the last user")
            initial_games = self._load_local_games(last_username, last_platform)
            if initial_games:
                log.info(f"Found {len(initial_games)} saved games. Populating list.")
                # Use the existing handler to populate the UI with these games.
                self._handle_fetched_games(games=initial_games)
            else:
                log.info("No saved games found for last user.")
            # --- END OF NEW LOGIC ---

        except FileNotFoundError:
            self.analyzer = None
            self.display_message("Welcome!", "Stockfish not found. Set STOCKFISH_PATH in .env or the Stockfish path in Settings.")
        except (TypeError, json.JSONDecodeError) as e:
            self.analyzer = None
            self.display_message("Initialization Error", f"Could not initialize analyzer: {e}. Please check your settings.")
        
        # This final call is still needed to render the initial empty board state
        # before the auto-load finishes populating the game list.
        self.update_display()


    def initialize_analyzer(self, gemini_key):
        """Starts the engine pool on a worker thread; Analyze stays disabled until it is ready."""
        old, self.analyzer, self.coach = self.analyzer, None, None
        self._engine_generation += 1
        if old: self._ui_executor.submit(old.close)
        if not self.stockfish_path or not os.path.exists(self.stockfish_path):
            raise FileNotFoundError("Stockfish path is not configured or is invalid.")
        generation = self._engine_generation
        paths = (self.stockfish_path, self.opening_book_path, self.syzygy_path)

        def start():
            analyzer = GameAnalyzer(EnginePool(paths[0]), opening_book_path=paths[1], syzygy_path=paths[2])
            if generation != self._engine_generation:  # restarted or closed meanwhile
                analyzer.close()
                return None
            return analyzer

        # The UI thread polls the start-up: Tk must not be called from the worker thread, and the
        # start may finish before mainloop() runs.
        self._engine_future = self._ui_executor.submit(start)
        self.after(100, self._on_engine_started, self._engine_future, generation, gemini_key)

    def _on_engine_started(self, future: Future, generation: int, gemini_key):
        if not future.done():
            self.after(100, self._on_engine_started, future, generation, gemini_key)
            return
        try:
            analyzer = future.result()
        except Exception as e:
            log.exception("Stockfish failed to start")
            self.display_message("Engine Error", f"Could not start Stockfish: {e}")
            return
        if analyzer is None: return
        if generation != self._engine_generation:
            self._ui_executor.submit(analyzer.close)
            return
        self.analyzer = analyzer
        log.info("Engine ready: %s x%d", analyzer.pool.name, analyzer.pool.size)
        if gemini_key:
            c = analyzer.config
            self.coach = Coach(gemini_key, self.gemini_model,
                               analyse=lambda board: next(iter(analyzer.analyse(board, c.FULL_DEPTH, 1, c.FULL_TIME)), None),
                               describe_line=analyzer.describe_line)
        self.on_game_select()

    def save_config(self, **kwargs):
        try:
            with open(Constants.CONFIG_FILE, 'r', encoding='utf-8') as f: config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError): config = {}
        config["last_username"], config["last_platform"] = self.username_entry.get().strip(), self.platform_var.get()
        config.update(kwargs)
        tmp = Constants.CONFIG_FILE + ".tmp"
        with open(tmp, 'w', encoding='utf-8') as f: json.dump(config, f, indent=4)
        os.replace(tmp, Constants.CONFIG_FILE)

    def display_message(self, title, text):
        self.game_summary = None
        self._update_analysis_display(title=title, commentary=text, show_sections=False)
        self.update_board(chess.Board())

    def _get_selected_game_url(self):
        selected_title = self.game_combo.get()
        for game in self.games:
            if f"{game.get('white', {}).get('username', 'White')} vs {game.get('black', {}).get('username', 'Black')} ({game.get('time_control', 'N/A')})" == selected_title:
                return game.get('url')
        return None

    def _get_selected_game_pgn(self):
        selected_title = self.game_combo.get()
        for game in self.games:
            if f"{game.get('white', {}).get('username', 'White')} vs {game.get('black', {}).get('username', 'Black')} ({game.get('time_control', 'N/A')})" == selected_title:
                return game.get('pgn')
        return None

    def _get_analysis_filepath(self, game_url):
        game_id = re.sub(r'[\\/*?:\"<>|]', "", (game_url or "").split('/')[-1])
        return os.path.join(Constants.ANALYSIS_DIR, f"{game_id}.json")

    def _save_analysis_to_file(self, game_url):
        filepath = self._get_analysis_filepath(game_url)
        data_to_save = {
            "version": "5.1", "moves": [m.uci() for m in self.moves], "analysis_data": [asdict(ply) for ply in self.analysis_data],
            "gemini_data": {ply: asdict(comment) for ply, comment in self.gemini_data.items()}, "game_summary": self.game_summary,
            "player_stats": self.player_stats, "player_accuracy": self.player_accuracy
        }
        try:
            tmp = filepath + ".tmp"
            with open(tmp, 'w', encoding='utf-8') as f: json.dump(data_to_save, f, ensure_ascii=False, indent=2)
            os.replace(tmp, filepath)
        except Exception as e: log.warning(f"Failed to save analysis: {e}")

    def _load_analysis_from_file(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f: data = json.load(f)
            if data.get("version") not in {"5.0", "5.1"}:
                self.after(0, lambda: self.display_message("Old Analysis File", "This analysis file is from an older version. Please re-analyze."))
                return
            moves = [chess.Move.from_uci(m) for m in data.get("moves", [])]
            def pos_feat_from(d): return PositionalFeatures(**d) if isinstance(d, dict) else PositionalFeatures()
            analysis_data: List[PlyAnalysis] = []
            for ply_data in data.get("analysis_data", []):
                ply_data["positional_analysis_before"] = pos_feat_from(ply_data.get("positional_analysis_before", {}))
                ply_data["positional_analysis_after"] = pos_feat_from(ply_data.get("positional_analysis_after", {}))
                def reconstruct_line(line_data):
                    if not isinstance(line_data, dict): return LineAnalysis()
                    return LineAnalysis(san=line_data.get("san", "N/A"), uci=line_data.get("uci", []), score_cp=line_data.get("score_cp", 0), tactical_motifs=line_data.get("tactical_motifs", []), positional_analysis=pos_feat_from(line_data.get("positional_analysis", {})))
                ply_obj = PlyAnalysis(
                    ply=ply_data.get("ply", 0), move_number_str=ply_data.get("move_number_str", ""), player=ply_data.get("player", ""), move_san=ply_data.get("move_san", ""),
                    evaluation_symbol=ply_data.get("evaluation_symbol", ""), fen_before=ply_data.get("fen_before", ""), classification=ply_data.get("classification", ""),
                    points_loss=ply_data.get("points_loss", 0.0), cp_loss=ply_data.get("cp_loss", 0), player_line=reconstruct_line(ply_data.get("player_line", {})),
                    best_engine_line=reconstruct_line(ply_data.get("best_engine_line", {})), best_follow_up=reconstruct_line(ply_data.get("best_follow_up", {})),
                    best_engine_lines=[reconstruct_line(l) for l in ply_data.get("best_engine_lines", [])], best_follow_up_lines=[reconstruct_line(l) for l in ply_data.get("best_follow_up_lines", [])],
                    positional_analysis_before=ply_data["positional_analysis_before"], positional_analysis_after=ply_data["positional_analysis_after"], syzygy_wdl=ply_data.get("syzygy_wdl"),
                    raw_engine_lines_for_display=ply_data.get("raw_engine_lines_for_display", []), game_phase=ply_data.get("game_phase", "middlegame"), opening_name=ply_data.get("opening_name", "Unknown Opening")
                )
                analysis_data.append(ply_obj)
            gemini_data = {ply: GeminiCommentary.from_dict(comment) for ply, comment in data.get("gemini_data", {}).items()}
            self.after(0, self._finalize_loading, moves, analysis_data, gemini_data, data.get("game_summary"), data.get("player_stats", {}), data.get("player_accuracy", {}))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            log.exception("Loading saved analysis failed")
            self.after(0, lambda e=e: self.display_message("Load Error", f"Error loading analysis: {e}. Please Re-analyze."))

    def _finalize_loading(self, moves, analysis_data, gemini_data, game_summary, player_stats, player_accuracy):
        self.moves, self.analysis_data, self.gemini_data = moves, analysis_data, gemini_data
        self.game_summary, self.player_stats, self.player_accuracy = game_summary, player_stats, player_accuracy
        self.current_ply = 0
        self.on_game_select()
        if self.analysis_data:
            key_moments = find_key_moments(self.analysis_data, 3)
            self._update_key_moments_display(key_moments)
        self.update_stats()
        self.update_display()
        self.analyze_btn.configure(text="Load Saved Analysis")

    def _set_navigation_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        for btn in [self.prev_btn, self.next_btn, self.show_follow_up_btn, self.toggle_engine_line_btn]:
            btn.configure(state=state)

    def on_closing(self):
        try:
            self._engine_generation += 1  # an engine still starting closes itself
            self._nav_generation += 1  # a refine still running drops its result instead of calling Tk
            if self.analyzer: self.analyzer.close()
            elif self._engine_future:  # started but not picked up yet: close it when it is there
                self._engine_future.add_done_callback(lambda f: f.exception() is None and f.result() and f.result().close())
        finally:
            try:
                if self.username_entry.get(): self.save_config()
            except: pass
            self.destroy()

