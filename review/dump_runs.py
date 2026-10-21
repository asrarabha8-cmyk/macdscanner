"""Dump scanner run logs + alert artifacts into review/raw so the alerts can be reviewed.

Runs inside GitHub Actions (needs GITHUB_TOKEN with actions:read).
"""
import io
import os
import re
import sys
import zipfile
from pathlib import Path

import requests

REPO = os.environ["GITHUB_REPOSITORY"]
TOKEN = os.environ["GITHUB_TOKEN"]
H = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json"}
API = f"https://api.github.com/repos/{REPO}"
OUT = Path("review/raw")
SINCE = os.environ.get("SINCE", "2026-09-01")

KEEP = {
    "MACD Cascade Scan", "MACD Slow Scan", "SMC Breakout Scanner", "SMC Daily Scanner",
    "MA 100/200 Breakout Scan", "EMA Cross Scan", "Bottom Hunter + Perp Filter",
    "SPY 0DTE ORB Alert", "OptiTrader Options Sim", "Flow Check", "Flow Scan",
    "Harmonic Scan (Carney)", "SA MA100-200 Scan (Tadawul)", "Stocks In Play Forward Test",
}
TS = re.compile(r"^\d{4}-\d\d-\d\dT[\d:.]+Z ")
NOISE = re.compile(r"(Downloading|Collecting|Requirement already|Installing|Successfully installed|"
                   r"Using cached|##\[group\]|##\[endgroup\]|Post job|git version|Cleaning up|"
                   r"Syncing repository|Temporarily overriding|Adding repository|hint:|"
                   r"Setting up auth|Fetching the repository|\[command\]|Determining the|Getting Git|"
                   r"Initializing the repository|Checking out|Disabling automatic|"
                   r"Waiting for a runner|Runner Image|Image:|Version:|Included Software|"
                   r"Operating System|Current runner|Secret source|Prepare workflow|"
                   r"Download action|Getting action|Complete job|Uploaded bytes|Artifact |"
                   r"Finished uploading|Beginning upload|With the provided path)")


def runs():
    page = 1
    while True:
        r = requests.get(f"{API}/actions/runs", headers=H,
                         params={"per_page": 100, "page": page, "created": f">={SINCE}"}, timeout=60)
        r.raise_for_status()
        items = r.json()["workflow_runs"]
        if not items:
            return
        yield from items
        page += 1


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for run in runs():
        if run["name"] not in KEEP:
            continue
        d = OUT / slug(run["name"])
        d.mkdir(exist_ok=True)
        f = d / f"{run['created_at'][:16].replace(':', '')}_{run['id']}.txt"
        if f.exists():
            continue
        r = requests.get(f"{API}/actions/runs/{run['id']}/logs", headers=H, timeout=120)
        if r.status_code != 200:
            f.write_text(f"# no logs ({r.status_code}) conclusion={run['conclusion']}\n")
            continue
        lines = []
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            # top-level files are one per job (full log); skip per-step duplicates in sub-dirs
            names = [x for x in z.namelist() if "/" not in x] or z.namelist()
            for name in names:
                for ln in z.read(name).decode("utf-8", "replace").splitlines():
                    ln = TS.sub("", ln)
                    if ln.strip() and not NOISE.search(ln):
                        lines.append(ln)
        f.write_text(f"# {run['name']} | {run['event']} | {run['conclusion']} | {run['created_at']}\n"
                     + "\n".join(lines) + "\n")
        n += 1
    print("dumped", n, "run logs")

    # alert-log artifacts (SMC breakout etc.)
    ad = OUT / "artifacts"
    ad.mkdir(exist_ok=True)
    r = requests.get(f"{API}/actions/artifacts", headers=H, params={"per_page": 100}, timeout=60)
    seen = set()
    for a in r.json().get("artifacts", []):
        if a["expired"] or a["name"] in seen or a["name"] in ("github-pages",) or "backtest" in a["name"] \
                or a["name"] == "kronos-results":
            continue
        if a["name"] != "alerts-log":
            seen.add(a["name"])  # newest of each name only (alerts-log: keep every run)
        z = requests.get(a["archive_download_url"], headers=H, timeout=120)
        if z.status_code != 200:
            continue
        with zipfile.ZipFile(io.BytesIO(z.content)) as zz:
            for name in zz.namelist():
                (ad / f"{a['name']}__{a['id']}__{Path(name).name}").write_bytes(zz.read(name))
    print("artifacts:", sorted(seen))


if __name__ == "__main__":
    sys.exit(main())
