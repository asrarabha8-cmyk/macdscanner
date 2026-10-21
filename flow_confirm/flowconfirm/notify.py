"""تليجرام + سجل الإشارات + منع التكرار."""
import csv
import datetime as dt
import json
import re
from pathlib import Path

import requests

from . import config as C

DATA = Path(__file__).resolve().parent.parent / "data"


def plain(text: str) -> str:
    return re.sub(r"</?b>", "", text).replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")


def send_telegram(text: str) -> bool:
    if not (C.TG_TOKEN and C.TG_CHAT):
        print("⚠️ TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID غير مضبوطين — طباعة فقط")
        return False
    ok = True
    for i in range(0, len(text), 3900):
        r = requests.post(f"https://api.telegram.org/bot{C.TG_TOKEN}/sendMessage",
                          data={"chat_id": C.TG_CHAT, "text": text[i:i + 3900],
                                "parse_mode": "HTML", "disable_web_page_preview": True},
                          timeout=20)
        ok &= r.ok
        if not r.ok:
            print("Telegram:", r.status_code, r.text[:200])
    return ok


def log_signal(row: dict):
    DATA.mkdir(exist_ok=True)
    path = DATA / "signals.csv"
    rows, fields = [], list(row)
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            old = rd.fieldnames or []
            if set(fields) <= set(old):
                fields = old  # نفس الأعمدة: إلحاق فقط
            else:
                rows = list(rd)  # أعمدة جديدة: نعيد كتابة الملف بالرأس الموسّع
                fields = old + [k for k in fields if k not in old]
        if not rows:
            with path.open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fields, extrasaction="ignore").writerow(row)
            return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows + [row])


def _state_path():
    return DATA / "alert_state.json"


def should_alert(key: str, mode: str, now: dt.datetime) -> bool:
    p = _state_path()
    state = json.loads(p.read_text()) if p.exists() else {}
    last = state.get(key)
    if last and now - dt.datetime.fromisoformat(last) < dt.timedelta(hours=C.ALERT_COOLDOWN_HOURS[mode]):
        return False
    return True


def mark_alert(key: str, now: dt.datetime):
    DATA.mkdir(exist_ok=True)
    p = _state_path()
    state = json.loads(p.read_text()) if p.exists() else {}
    state[key] = now.isoformat()
    p.write_text(json.dumps(state, indent=1))
