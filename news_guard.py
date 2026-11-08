"""news_guard — فحص أخبار السهم قبل التنبيه (فكرة مأخوذة من daily_stock_analysis).

يستخدم أخبار Polygon/Massive (/v2/reference/news) مع تقييم المشاعر لكل سهم
(insights)، ويضيف فحص كلمات للأخبار الخطيرة (طرح أسهم، تخفيض تصنيف، تحقيق...).

الاستخدام في أي ماسح:
    from news_guard import check, badge
    info = check("AAPL")          # الآن
    msg += badge(info)            # يضيف سطر تحذير للتنبيه إذا فيه خبر سلبي

لا يمنع التنبيه — يضيف تحذير فقط، والقرار لك.
"""
from __future__ import annotations

import datetime as dt
import os
import re

import requests

BASE = (os.getenv("POLYGON_BASE_URL") or "https://api.polygon.io").rstrip("/")
KEY = os.getenv("POLYGON_API_KEY", "")

# أخبار خطيرة غالباً تضغط السعر حتى لو كان التقييم محايد
RED_FLAGS = [
    r"\boffering\b", r"\bdilut", r"\bdowngrade", r"\bcuts? (guidance|outlook|forecast)",
    r"\blowers? (guidance|outlook|forecast)", r"\bmiss(es|ed)?\b", r"\bprobe\b",
    r"\binvestigation\b", r"\bsubpoena", r"\blawsuit\b", r"\bclass action\b", r"\bSEC\b",
    r"\bfraud\b", r"\brecall\b", r"\bbankrupt", r"\bchapter 11\b", r"\bdelist",
    r"\bresign", r"\bsteps down\b", r"\bshort seller\b", r"\bhalt(ed)?\b", r"\bplunge",
]
_RED = re.compile("|".join(RED_FLAGS), re.I)
# إعلانات مكاتب المحاماة (class action) تتكرر يومياً وما تحرك السعر
_SPAM = re.compile(r"investors? (to|who)|secure counsel|deadline|law firm|llc urges|rosen|pomerantz|"
                   r"bronstein|levi & korsinsky|faruqi|glancy|bragar|schall|kessler topaz", re.I)


def _iso(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch(ticker: str, end: dt.datetime, days: int = 3, limit: int = 50) -> list[dict]:
    start = end - dt.timedelta(days=days)
    r = requests.get(
        f"{BASE}/v2/reference/news",
        params={"ticker": ticker, "published_utc.gte": _iso(start),
                "published_utc.lte": _iso(end), "order": "desc",
                "limit": limit, "apiKey": KEY},
        timeout=20,
    )
    r.raise_for_status()
    return r.json().get("results", []) or []


def check(ticker: str, when: dt.datetime | None = None, days: int = 3) -> dict:
    """يرجّع ملخص أخبار السهم خلال `days` أيام قبل `when` (افتراضياً الآن)."""
    when = when or dt.datetime.now(dt.timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    out = {"ticker": ticker, "n": 0, "pos": 0, "neg": 0, "neu": 0,
           "red": 0, "red_titles": [], "neg_titles": [], "error": ""}
    try:
        items = fetch(ticker, when, days)
    except Exception as e:  # الشبكة أو المفتاح — لا نوقف الماسح
        out["error"] = str(e)[:120]
        return out
    for it in items:
        title = it.get("title") or ""
        out["n"] += 1
        sent = ""
        for ins in it.get("insights") or []:
            if (ins.get("ticker") or "").upper() == ticker.upper():
                sent = (ins.get("sentiment") or "").lower()
                break
        if sent == "positive":
            out["pos"] += 1
        elif sent == "negative":
            out["neg"] += 1
            out["neg_titles"].append(title)
        else:
            out["neu"] += 1
        # الخبر الخطير لازم يكون عن السهم نفسه: التقييم سلبي له + كلمة خطيرة، وليس إعلان محامين
        if sent == "negative" and _RED.search(title) and not _SPAM.search(title):
            out["red"] += 1
            out["red_titles"].append(title)
    out["score"] = out["pos"] - out["neg"]
    out["flag"] = level(out)
    return out


def level(info: dict) -> str:
    """negative / red / positive / quiet / mixed"""
    if info.get("red"):
        return "red"
    if info["neg"] > info["pos"]:
        return "negative"
    if info["pos"] > info["neg"]:
        return "positive"
    return "quiet" if info["n"] == 0 else "mixed"


def badge(info: dict) -> str:
    """سطر جاهز يُلصق في رسالة التليجرام."""
    f = info.get("flag")
    if info.get("error") or f is None:
        return ""
    if f == "red":
        t = info["red_titles"][0][:90]
        return f"\n🚨 خبر خطير: {t}"
    if f == "negative":
        return f"\n⚠️ أخبار سلبية ({info['neg']} سلبي / {info['pos']} إيجابي)"
    if f == "positive":
        return f"\n📰 أخبار إيجابية ({info['pos']})"
    return ""


if __name__ == "__main__":
    import json
    import sys
    for t in sys.argv[1:] or ["AAPL"]:
        print(json.dumps(check(t), ensure_ascii=False, indent=1))
