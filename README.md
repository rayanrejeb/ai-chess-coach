# ♟️ AI Chess Coach — *Stockfish + Gemini*

> **A chess analysis tool using Stockfish and Google's Gemini API.**
> Built to analyze chess games, identify tactical motifs, and generate insights.

---

## ⚙️ Setup Instructions

### 1️⃣ Install Python and Dependencies
Make sure you have **Python 3.10 or higher** installed, then install the requirements:

```bash
pip install -r requirements.txt
```

### 2️⃣ Install Stockfish
Stockfish is not included in this repository. [Download it from the official website](https://stockfishchess.org/download/) (or `apt install stockfish` / `brew install stockfish`) and note the path to the binary.

### 3️⃣ Configure `.env`

```bash
cp .env.example .env
```

Then edit `.env`:

* `GEMINI_API_KEY` — [generate one here](https://aistudio.google.com/app/apikey).
* `STOCKFISH_PATH` — full path to the Stockfish binary.
* `LICHESS_TOKEN` — only needed to sync Lichess games. [Create one here](https://lichess.org/account/oauth/token).
* `GEMINI_MODEL`, `DEFAULT_USERNAME`, `LOG_LEVEL` — optional.

Values in `.env` override the ones saved from the in-app **Settings** window (stored in the git-ignored `config.json`).

### 4️⃣ Run

```bash
python chess_analyzer.py
```

Enter your Chess.com or Lichess username → **Fetch Games** → pick a game → **Analyze Selected Game**.

Sample games (public-domain 19th-century classics) are in [`sample_data/`](sample_data/).

---

## 🚀 What You Can Improve or Add

* 🎨 **Enhanced GUI** — redesign for a smoother and more modern experience.
* 📱 **Cross-Platform Apps** — build Android, iOS, or Windows installer versions.
* 🧩 **"Difficult Moves" Mode** — practice tricky positions interactively.
* 🧠 **Deeper Analysis Logic** — refine tactical/positional understanding and Gemini prompts.
* ⚡ **Stockfish** — parallel analysis, because analysis is slow.
* 📘 **Learning Mode** — auto-extract puzzles or lessons from books.
* 📊 **More Statistics** — performance graphs, accuracy over time, motif frequency, etc.
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
|  5  | "Removal of Guard (Self)" | Needs logical correction.                                            |
|  6  | Potential fork flaws      | Doesn't consider opponent's immediate response.                      |

---

## 🧰 Tech Stack

| Component        | Description                       |
| ---------------- | --------------------------------- |
| **Language**     | Python                            |
| **Engine**       | Stockfish                         |
| **AI Model**     | Google Gemini API                 |
| **Game Data**    | Chess.com and Lichess APIs        |
| **UI Framework** | CustomTkinter                     |

---

## 💬 License & Credits

* **License:** [GPL-3.0](LICENSE) (required by python-chess, which is GPL-3.0)
* **Engine:** [Stockfish](https://stockfishchess.org/) (GPL-3.0)
* **Chess library:** [python-chess](https://github.com/niklasf/python-chess) (GPL-3.0)
* **AI Model:** Google Gemini API
* **Data Sources:** Chess.com and Lichess APIs
