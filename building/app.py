import base64, io, json, os, re, sqlite3, time
from xml.sax.saxutils import escape

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_file, send_from_directory
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

load_dotenv()
app = Flask(__name__, static_folder="static", static_url_path="")
KEY = os.getenv("GEMINI_API_KEY") or ""
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "campus.db")
ORANGE, BLACK = colors.HexColor("#f97316"), colors.HexColor("#111111")

SYSTEM = (
    "You are the AI operations assistant of a Green Building Digital Twin for a college academic block "
    "(8 classrooms on 2 floors, college hours 09:00-16:00 with four 45-minute periods and an afternoon lab). "
    "You are talking to the campus admin. Use only the data provided below: current state, stored history, "
    "parameters, recent events and earlier chats. The data comes from a simulation, so say so if asked. "
    "Quote numbers with units (kWh, kW, degrees C). If something is missing, say what is missing instead of guessing. "
    "Answer in concise Markdown with concrete next steps. You cannot operate devices yourself: point the admin "
    "to the Control tab, the Alerts tab or the AI handoff switch.\n\n"
)


def db():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with db() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS snapshots(id INTEGER PRIMARY KEY, ts REAL, day INTEGER, sim_min INTEGER, payload TEXT);
            CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, ts REAL, day INTEGER, sim_time TEXT, kind TEXT, text TEXT);
            CREATE TABLE IF NOT EXISTS param_log(id INTEGER PRIMARY KEY, ts REAL, day INTEGER, sim_time TEXT, key TEXT, value TEXT);
            CREATE TABLE IF NOT EXISTS chats(id INTEGER PRIMARY KEY, title TEXT, created REAL);
            CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY, chat_id INTEGER, role TEXT, text TEXT, ts REAL);
            """
        )


init_db()


def gemini_call(contents, model=None, system=None, max_tokens=4096):
    body = {"contents": contents, "generationConfig": {"temperature": 0.4, "maxOutputTokens": max_tokens}}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    try:
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model or DEFAULT_MODEL}:generateContent",
            headers={"x-goog-api-key": KEY},
            json=body,
            timeout=90,
        )
        j = r.json()
        if not r.ok:
            return None, j.get("error", {}).get("message", f"HTTP {r.status_code}")
        return "".join(p.get("text", "") for p in j["candidates"][0]["content"]["parts"]), None
    except Exception as e:
        return None, str(e)


@app.get("/")
def home():
    return send_from_directory("static", "index.html")


@app.get("/api/health")
def health():
    with db() as c:
        n = lambda t: c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        return jsonify(ok=True, gemini_key_set=bool(KEY), snapshots=n("snapshots"), events=n("events"), messages=n("messages"))


@app.post("/api/gemini")
def gemini():
    d = request.get_json(force=True)
    if not KEY:
        return jsonify(error="GEMINI_API_KEY is not set in .env")
    model = d.get("model") or DEFAULT_MODEL
    text, err = gemini_call([{"role": "user", "parts": [{"text": d["prompt"]}]}], model)
    return jsonify(error=err) if err else jsonify(text=text, model=model)


@app.post("/api/snapshot")
def snapshot():
    d = request.get_json(force=True)
    now = time.time()
    with db() as c:
        last = c.execute("SELECT payload FROM snapshots ORDER BY id DESC LIMIT 1").fetchone()
        old = json.loads(last["payload"]).get("params", {}) if last else {}
        for k, v in d.get("params", {}).items():
            if old.get(k) != v:
                c.execute(
                    "INSERT INTO param_log(ts,day,sim_time,key,value) VALUES(?,?,?,?,?)",
                    (now, d["day"], d["clock"], k, json.dumps(v)),
                )
        c.execute("INSERT INTO snapshots(ts,day,sim_min,payload) VALUES(?,?,?,?)", (now, d["day"], d["minute"], json.dumps(d)))
        for e in d.get("events", []):
            c.execute(
                "INSERT INTO events(ts,day,sim_time,kind,text) VALUES(?,?,?,?,?)",
                (now, e["day"], e["simTime"], e["kind"], e["text"]),
            )
    return jsonify(ok=True)


@app.get("/api/export")
def export():
    with db() as c:
        data = {
            "snapshots": [json.loads(r["payload"]) for r in c.execute("SELECT payload FROM snapshots ORDER BY id LIMIT 5000")],
            "events": [dict(r) for r in c.execute("SELECT day, sim_time, kind, text FROM events ORDER BY id")],
            "parameter_changes": [dict(r) for r in c.execute("SELECT day, sim_time, key, value FROM param_log ORDER BY id")],
            "chats": [
                {**dict(ch), "messages": [dict(m) for m in c.execute("SELECT role, text, ts FROM messages WHERE chat_id=? ORDER BY id", (ch["id"],))]}
                for ch in c.execute("SELECT id, title, created FROM chats ORDER BY id")
            ],
        }
    return jsonify(data)


def build_context(chat_id):
    with db() as c:
        last = c.execute("SELECT payload FROM snapshots ORDER BY id DESC LIMIT 1").fetchone()
        days = c.execute(
            "SELECT day, MAX(json_extract(payload,'$.totals.kwh')) kwh, MAX(json_extract(payload,'$.totals.waste')) wasted_kwh, "
            "MAX(json_extract(payload,'$.totals.acts')) ai_actions, COUNT(*) samples FROM snapshots GROUP BY day ORDER BY day DESC LIMIT 14"
        ).fetchall()
        events = c.execute("SELECT day, sim_time, kind, text FROM events ORDER BY id DESC LIMIT 40").fetchall()
        params = c.execute("SELECT day, sim_time, key, value FROM param_log ORDER BY id DESC LIMIT 25").fetchall()
        other = c.execute("SELECT chat_id, role, text FROM messages WHERE chat_id != ? ORDER BY id DESC LIMIT 12", (chat_id,)).fetchall()
    rows = lambda rs: json.dumps([dict(r) for r in rs])
    parts = [
        "CURRENT STATE (latest stored snapshot, includes parameters, totals, hourly energy, rooms, open alerts):\n"
        + (last["payload"] if last else "No simulation data stored yet."),
        "DAILY HISTORY (newest first; kwh and wasted_kwh are the day's final totals):\n" + rows(days),
        "PARAMETER CHANGES (newest first):\n" + rows(params),
        "RECENT EVENTS (newest first):\n" + rows(events),
        "EARLIER CHATS (newest first, truncated):\n" + "\n".join(f"chat {r['chat_id']} {r['role']}: {r['text'][:300]}" for r in other),
    ]
    return SYSTEM + "\n\n".join(parts)


@app.get("/api/chats")
def chats():
    with db() as c:
        rows = c.execute("SELECT id, title, created FROM chats ORDER BY id DESC").fetchall()
    return jsonify(chats=[dict(r) for r in rows])


@app.get("/api/chats/<int:cid>")
def chat_get(cid):
    with db() as c:
        rows = c.execute("SELECT role, text FROM messages WHERE chat_id=? ORDER BY id", (cid,)).fetchall()
    return jsonify(messages=[dict(r) for r in rows])


@app.delete("/api/chats/<int:cid>")
def chat_delete(cid):
    with db() as c:
        c.execute("DELETE FROM messages WHERE chat_id=?", (cid,))
        c.execute("DELETE FROM chats WHERE id=?", (cid,))
    return jsonify(ok=True)


@app.post("/api/chat")
def chat():
    d = request.get_json(force=True)
    msg = (d.get("message") or "").strip()
    if not msg:
        return jsonify(error="Empty message")
    if not KEY:
        return jsonify(error="GEMINI_API_KEY is not set in .env")
    cid = d.get("chat_id")
    with db() as c:
        if not cid:
            cid = c.execute("INSERT INTO chats(title, created) VALUES(?,?)", (msg[:48], time.time())).lastrowid
        mid = c.execute("INSERT INTO messages(chat_id, role, text, ts) VALUES(?,?,?,?)", (cid, "user", msg, time.time())).lastrowid
        history = c.execute("SELECT role, text FROM messages WHERE chat_id=? ORDER BY id", (cid,)).fetchall()
    contents = [{"role": "user" if h["role"] == "user" else "model", "parts": [{"text": h["text"]}]} for h in history]
    reply, err = gemini_call(contents, d.get("model"), build_context(cid), 2048)
    with db() as c:
        if err:
            c.execute("DELETE FROM messages WHERE id=?", (mid,))
            return jsonify(error=err)
        c.execute("INSERT INTO messages(chat_id, role, text, ts) VALUES(?,?,?,?)", (cid, "model", reply, time.time()))
    return jsonify(reply=reply, chat_id=cid)


def clean(t):
    t = str(t).replace("₂", "2").replace("→", "->").replace("≈", "~").replace("−", "-").replace("×", "x")
    return re.sub(r"[^\x20-\xFF\u2013\u2014\u2022]", "", t)


def inline(t):
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escape(clean(t))).replace("`", "")


def table(rows, head_color=ORANGE, size=8):
    cell = ParagraphStyle("c", fontSize=size, leading=size + 2)
    data = [[Paragraph(inline(c), cell) for c in r] for r in rows]
    t = Table(data, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), head_color), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("GRID", (0, 0), (-1, -1), .25, colors.HexColor("#d1d5db")), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return t


def markdown_flowables(text, st):
    out, tb = [], []

    def flush():
        nonlocal tb
        if tb:
            out.extend([table(tb), Spacer(1, 6)])
            tb = []

    for raw in text.split("\n"):
        l = raw.strip()
        if l.startswith("|"):
            if not re.fullmatch(r"[\s|:-]+", l):
                tb.append([c.strip() for c in l.strip("|").split("|")])
            continue
        flush()
        if not l:
            continue
        m = re.match(r"#{1,4} (.*)", l)
        if m:
            out.append(Paragraph(inline(m.group(1)), st["h2"]))
        elif re.match(r"([-*]|\d+\.) ", l):
            out.append(Paragraph("• " + inline(re.sub(r"^([-*]|\d+\.) ", "", l)), st["b"]))
        else:
            out.append(Paragraph(inline(l), st["Normal"]))
    flush()
    return out


@app.post("/api/pdf")
def pdf():
    d = request.get_json(force=True)
    ss = getSampleStyleSheet()
    st = {"Normal": ParagraphStyle("n", parent=ss["Normal"], fontSize=10, leading=14, spaceAfter=3),
          "b": ParagraphStyle("b", parent=ss["Normal"], fontSize=10, leading=14, leftIndent=12, spaceAfter=2),
          "h2": ParagraphStyle("h", parent=ss["Heading2"], fontSize=13, textColor=BLACK, spaceBefore=10, spaceAfter=4),
          "t": ParagraphStyle("t", parent=ss["Title"], fontSize=20, alignment=0, textColor=BLACK)}
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=15 * mm, bottomMargin=15 * mm,
                            title=clean(d.get("title", "Report")))
    story = [Paragraph("Smart Campus - " + inline(d.get("title", "Report")), st["t"]),
             Paragraph(inline(d.get("header", "")), ParagraphStyle("s", fontSize=8, textColor=colors.grey)), Spacer(1, 8)]
    if d.get("kpis"):
        story += [table([[k[0] for k in d["kpis"]], [str(k[1]) for k in d["kpis"]]], BLACK, 9), Spacer(1, 10)]
    figs, w = d.get("figs", []), (A4[0] - 30 * mm - 6) / 2
    cells = []
    for f in figs:
        img = Image(io.BytesIO(base64.b64decode(f["u"].split(",", 1)[1])), width=w - 6, height=(w - 6) * 260 / 640)
        cells.append([Paragraph("<b>" + inline(f["t"]) + "</b>", ParagraphStyle("f", fontSize=8, leading=10)), img])
    for i in range(0, len(cells), 2):
        row = cells[i:i + 2] + ([""] if len(cells[i:i + 2]) == 1 else [])
        story += [Table([row], colWidths=[w, w]), Spacer(1, 4)]
    story += markdown_flowables(d.get("text", ""), st)
    if d.get("suggestions"):
        story += [Paragraph("Smart suggestions", st["h2"]),
                  table([["Priority", "Suggestion", "Evidence", "Action", "Impact", "Effort"]] +
                        [[s["pri"], s["t"], s["why"], s["act"], f"~{s['kwh']} kWh/mo" if s["kwh"] else "-", s["eff"]]
                         for s in d["suggestions"]])]
    doc.build(story)
    buf.seek(0)
    return send_file(buf, mimetype="application/pdf", as_attachment=True, download_name="report.pdf")


if __name__ == "__main__":
    print("Admin panel: http://127.0.0.1:5000   (Gemini key set: %s)" % bool(KEY))
    app.run(host="127.0.0.1", port=5000, debug=True)
