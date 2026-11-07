# AI Chess Coach

Desktop app that reviews your Chess.com and Lichess games with Stockfish and explains them in plain language using Gemini.

## Setup

1. Install Python 3.10+ and the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Install Stockfish. It is not included in this repository: [download it](https://stockfishchess.org/download/), or run `apt install stockfish` / `brew install stockfish`.

3. Create your `.env`:

   ```bash
   cp .env.example .env
   ```

   | Variable | Required | Purpose |
   | --- | --- | --- |
   | `GEMINI_API_KEY` | yes | [Google AI Studio key](https://aistudio.google.com/app/apikey) |
   | `STOCKFISH_PATH` | yes | Full path to the Stockfish binary |
   | `LICHESS_TOKEN` | for Lichess sync | [Personal token](https://lichess.org/account/oauth/token) |
   | `GEMINI_MODEL`, `DEFAULT_USERNAME`, `LOG_LEVEL` | no | Optional settings |

   Values in `.env` override the ones saved from the in-app Settings window (stored in the git-ignored `config.json`).

4. Run:

   ```bash
   python main.py
   ```

   Enter a Chess.com or Lichess username, click **Fetch Games**, pick a game, then click **Analyze Selected Game**.

Sample games (public-domain 19th-century classics) are in [`sample_data/`](sample_data/).

## License

[GPL-3.0](LICENSE). Third-party code and its license are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Credits

Started from [Chess Analyzer](https://github.com/yusufelgen07/chess-analyzer-vibe-coding-) by yusufelgen07 (MIT License). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
