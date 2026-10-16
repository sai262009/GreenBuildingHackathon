# Smart Campus – Green Building Digital Twin

Admin panel + 3D twin served by a small Python (Flask) backend.
The backend keeps your Gemini key private and builds the PDF reports.

## Run it in VS Code
1. Install Python 3.10+ and the VS Code **Python** extension.
2. `File → Open Folder…` and pick this folder.
3. Open a terminal (`Ctrl+` `) and run:
   ```
   python -m venv .venv
   .venv\Scripts\activate        # Windows      (Mac/Linux: source .venv/bin/activate)
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and put a NEW free key from https://aistudio.google.com in `GEMINI_API_KEY=`.
5. Run `python app.py` (or press F5) and open http://127.0.0.1:5000

## Stored data and the Ask AI chat
- While the panel runs it posts snapshots (parameters, totals, rooms, alerts, events) to `/api/snapshot`; they are stored in `campus.db` (SQLite).
- **Ask AI** sends the whole conversation plus the live state, stored history, parameter changes, recent events and earlier chats to Gemini on every message.
- `/api/export` (Settings → Export stored data) downloads everything as JSON.

## What goes where
- `app.py` – backend: `/api/gemini` (calls Gemini), `/api/pdf` (builds the PDF), serves `static/`
- `static/index.html` – admin panel · `static/twin.html` – 3D twin (button in the header)
- Without the backend the panel still works: it falls back to the browser-only Gemini call and browser PDF.

Never commit `.env`; it is already in `.gitignore`.
