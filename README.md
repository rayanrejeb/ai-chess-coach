<<<<<<< HEAD

# ♟️ Chess Analyzer — *Stockfish + Gemini API*


> **A chess analysis tool using Stockfish and Google’s Gemini Free API.**  
> Built to analyze chess games, identify tactical motifs, and generate insights.  
> This project is **open for community continuation** — I’m not maintaining it further.

---

## ⚙️ Setup Instructions

### 1️⃣ Install Python and Dependencies
Make sure you have **Python 3.10 or higher** installed, then install the requirements:

```bash
pip install -r requirements.txt
````

### 2️⃣ Get Your API Keys and Engines

* **Gemini API Key:** [Generate one here](https://aistudio.google.com/app/apikey).
* **Stockfish Engine:** [Download from the official website](https://stockfishchess.org/download/).
* **Lichess Token:** [Create one here](https://lichess.org/account/oauth/token).

### 3️⃣ Configure Settings

1. Run the program.
2. Open the **Settings** panel.
3. Add:

   * ✅ Path to your **Stockfish** binary.
   * ✅ Your **Lichess API token**.
   * ✅ Set **Depth = 10** for analysis.
   * ✅ Choose **Gemini Flash Lite** for testing.
   * ✅ Click Save and Close
   * ✅ add your username in chess.com or lichess ->> click fetch games ->> choose your game ->> click analyze Selected game 

---

## 🚀 What You Can Improve or Add

* 🎨 **Enhanced GUI** — redesign for a smoother and more modern experience.
* 📱 **Cross-Platform Apps** — build Android, iOS, or Windows installer versions.
* 🧩 **“Difficult Moves” Mode** — like Chessable: practice tricky positions interactively.
* 🧠 **Deeper Analysis Logic** — refine tactical/positional understanding and Gemini prompts.
* ⚡ **Stockfish** -synchronization because analysis is too slow.
* 📘 **Learning Mode** — auto-extract puzzles or lessons from books (e.g. *1001 Chess Exercises for Beginners*).
* 📊 **More Statistics** — performance graphs, accuracy over time, motif frequency, etc.
* 🧰 **Installer Creation** — make it user-friendly for offline use.
* 🤖 **Automatic Game Analyzer** — run batch analysis based on available Gemini API calls.
* 📚 **Opening Book Integration** — recognize and suggest openings dynamically.

---

## 🐞 Known Issues

|  #  | Issue                     | Description                                                          |
| :-: | ------------------------- | -------------------------------------------------------------------- |
|  1  | Repeated motifs           | Same tactical idea appears across multiple moves.                    |
|  2  | Pin detection (self)      | Potential pins by the player sometimes not detected.                 |
|  3  | Missing break logic       | No logic for identifying when motifs (like skewers/pins) are broken. |
|  4  | Redundant motifs          | Same pin suggested multiple times by different pieces.               |
|  5  | “Removal of Guard (Self)” | Needs logical correction.                                            |
|  6  | Potential fork flaws      | Doesn’t consider opponent’s immediate response.                      |

---

## 💡 Contribution Guide

Contributions are welcome!
If you’d like to improve or complete the project:

1. **Fork** this repository.
2. **Implement** your changes or fixes.
3. **Submit** a pull request.
4. Or open an **issue** for bugs, improvements, or questions.

### 🧾 When Reporting Issues

Include:

* The move or FEN position where it occurred.
* What you **expected** vs. what actually happened.
* Any relevant **logs, screenshots, or analysis outputs**.

---

## 🧭 Overview

This project merges **Stockfish’s tactical precision** with **Gemini’s natural-language reasoning** to analyze chess games, detect motifs, and provide human-like insights.

While incomplete, it serves as a **strong foundation** for an intelligent chess-training and analysis platform.

---

## 🧰 Tech Stack

| Component        | Description                                |
| ---------------- | ------------------------------------------ |
| **Language**     | Python                                     |
| **Engine**       | Stockfish                                  |
| **AI Model**     | Google Gemini Free API                     |
| **Game Data**    | Lichess API                                |
| **UI Framework** | Custom GUI (Tkinter — adaptable to others) |

---

## 🧑‍💻 Suggested Enhancements for Developers

* Integrate a **parallel Stockfish analyzer** for multi-move depth analysis.
* Implement **motif confidence scoring** to avoid repeated detections.
* Add **structured JSON responses** for Gemini outputs.
* Build a **course creation module** that converts books/puzzles into learning paths.

---

## 💬 License & Credits

* **License:** MIT
* **Engine:** Stockfish (GPL-compatible)
* **AI Model:** Gemini Free API (by Google)
* **Data Source:** Lichess API

> This project is provided as-is for educational and developmental purposes.

---

## 🏁 Final Note

This is an **open, unfinished project** — free for others to learn from, improve, or expand.
Good luck, and may your code always find the best move! ♜♞♝



=======
# Chess-Coach-Trainer
>>>>>>> origin/improved-base-app
