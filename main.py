"""Entry point: python main.py"""

import logging
import os
import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk
from dotenv import load_dotenv

from ai_chess_coach.gui.app import ChessGUI

log = logging.getLogger("ai_chess_coach")


def main():
    load_dotenv()
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        log.info("AI Chess Coach starting")
        app = ChessGUI()
        app.protocol("WM_DELETE_WINDOW", app.on_closing)
        app.mainloop()
    except Exception as e:
        log.exception(f"Fatal error in main: {e}")
        try:
            messagebox.showerror("Fatal Error", f"A fatal error occurred on startup:\n{e}\n\nPlease see the console for details.")
        except tk.TclError:
            pass


if __name__ == "__main__":
    main()
