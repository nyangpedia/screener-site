#!/usr/bin/env python3
"""매일 15:00 자동 실행용: 지난 후보 성적 기록 → 오늘 스크리닝 → 텔레그램 전송.

  python daily.py --dry-run     # 텔레그램 대신 화면에 출력
  python daily.py               # 환경변수 TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID 필요

성적 기록(reports/track.csv): reports/*.csv 의 조건 통과 상위 후보를
그날 확정 종가에 사서 다음 거래일 시가에 판 수익률(왕복 비용 차감)로 쌓는다.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import traceback
import urllib.request
from pathlib import Path

import pandas as pd

from screener import HERE, KST, Config, FDRProvider, fmt, run

TRACK_COLS = ["date", "Code", "Name", "close", "next_date", "next_open", "ret%"]


# ---------------------------------------------------------------- 성적 기록
def picks(out_dir: Path, top: int) -> pd.DataFrame:
    """지난 리포트 CSV에서 날짜별 조건 통과 상위 후보 (같은 날 여러 번 돌렸으면 합집합)."""
    rows = []
    for f in sorted(out_dir.glob("????-??-??_????.csv")):
        d = pd.read_csv(f, dtype={"Code": str})
        d = d[d["pass"].astype(str) == "True"].head(top)
        rows += [{"date": f.stem[:10], "Code": c.zfill(6), "Name": n} for c, n in zip(d["Code"], d["Name"])]
    return pd.DataFrame(rows, columns=["date", "Code", "Name"]).drop_duplicates(["date", "Code"])


def update_track(provider, cfg: Config, out_dir: Path) -> pd.DataFrame:
    path = out_dir / "track.csv"
    done = pd.read_csv(path, dtype={"Code": str}) if path.exists() else pd.DataFrame(columns=TRACK_COLS)
    seen = set(zip(done["date"], done["Code"]))
    new = []
    for r in picks(out_dir, cfg.top).itertuples():
        if (r.date, r.Code) in seen:
            continue
        try:
            h = provider.history(r.Code, r.date)
        except Exception:
            continue                                   # 다음 실행 때 다시 시도
        h = h[h.index >= r.date]
        # 휴장일 리포트(첫 봉 날짜 불일치)이거나 다음날 시가가 아직 없으면 건너뜀
        if len(h) < 2 or h.index[0].date().isoformat() != r.date:
            continue
        close, nxt = float(h["Close"].iloc[0]), float(h["Open"].iloc[1])
        new.append({"date": r.date, "Code": r.Code, "Name": r.Name, "close": close,
                    "next_date": h.index[1].date().isoformat(), "next_open": nxt,
                    "ret%": (nxt / close - 1 - cfg.round_trip_cost) * 100})
    if new:
        done = pd.concat([done, pd.DataFrame(new)], ignore_index=True).sort_values(["date", "Code"])
        done.to_csv(path, index=False, encoding="utf-8-sig")
    return done


def track_summary(t: pd.DataFrame, last_date: str | None = None) -> list[str]:
    if t.empty:
        return ["성적: 아직 결과 없음 (다음날 시가가 나와야 기록됨)"]
    r = t["ret%"].astype(float)
    L = [f"누적 {len(r)}건 ({t['date'].min()}~): 평균 {fmt(r.mean(),2,'%')}, 중앙값 {fmt(r.median(),2,'%')}, "
         f"승률 {fmt((r > 0).mean()*100,0,'%')}, 최악 {fmt(r.min(),2,'%')}"]
    if len(r) < 30:
        L.append(f"  ⚠ {len(r)}건은 통계로 읽기엔 적음")
    last = t[t["date"] == (last_date or t["date"].max())]
    L += [f"  {x['date']} {x['Name']}: {fmt(float(x['ret%']),2,'%')}" for x in last.to_dict("records")]
    return L


# ---------------------------------------------------------------- 메시지 & 전송
def is_trading_day(today: dt.date) -> bool:
    """KRX 휴장일 달력(holidays 패키지 XKRX) 기준. 휴장일엔 전 거래일 후보가 다시 나가는 것 방지.
    ponytail: 갑자기 정해진 임시공휴일은 패키지 업데이트 전까지 놓칠 수 있음."""
    import holidays
    return today.weekday() < 5 and today not in holidays.financial_holidays("XKRX", years=today.year)


def build_message(res: pd.DataFrame, cfg: Config, now: dt.datetime, track: pd.DataFrame) -> str:
    passed = res[res["pass"]].head(cfg.top) if not res.empty else res
    when = "장중 스냅샷" if (now.hour, now.minute) < (15, 31) else "장 마감 후"
    L = [f"[종가베팅 후보] {now:%Y-%m-%d %H:%M} KST {when}", f"조건 통과 {len(passed)}개"]
    if passed.empty:
        L.append("오늘은 조건 통과 종목 없음")
    for i, d in enumerate(passed.to_dict("records"), 1):
        warn = "" if d["sig_n"] >= cfg.min_samples else "⚠"
        L.append(f"{i}. {d['Name']}({d['Code']}) {fmt(d['Chg%'],1,'%')} {fmt(d['Amount억'],0)}억 "
                 f"거래량{fmt(d['VolRatio'],1,'x')} | 과거n={d['sig_n']}{warn} 평균{fmt(d['sig_mean%'],2,'%')} "
                 f"승률{fmt(d['sig_win%'],0,'%')}")
    L += ["", *track_summary(track)]
    L += ["", "뉴스·공시는 확인하지 않은 가격 조건 결과. 동시호가에서 가격이 바뀔 수 있음. 투자 자문 아님."]
    return "\n".join(L)


# ---------------------------------------------------------------- 웹 대시보드 (docs/ = 공개 폴더)
SITE = HERE / "docs"


def records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", force_ascii=False)) if not df.empty else []


def export_site(res: pd.DataFrame, cfg: Config, now: dt.datetime, track: pd.DataFrame, out_dir: Path):
    """docs/data.json 갱신. 페이지(index.html)는 이 파일만 읽는다."""
    passed = res[res["pass"]].head(cfg.top) if not res.empty else res
    cols = ["Code", "Name", "Market", "Close", "Chg%", "Amount억", "VolRatio", "ClosePos", "MA20Gap%",
            "Break20H", "sig_n", "sig_mean%", "sig_median%", "sig_win%", "sig_worst%", "last_bar"]
    seen = set(zip(track["date"], track["Code"]))
    pending = picks(out_dir, cfg.top)
    pending = pending[[k not in seen for k in zip(pending["date"], pending["Code"])]]
    r = track["ret%"].astype(float)
    data = {
        "updated": f"{now:%Y-%m-%d %H:%M}",
        "session": "장중 스냅샷" if (now.hour, now.minute) < (15, 31) else "장 마감 후",
        "cost_pct": round(cfg.round_trip_cost * 100, 3),
        "min_samples": cfg.min_samples,
        "candidates": records(passed[cols] if not passed.empty else passed),
        "stats": None if r.empty else {"n": len(r), "mean": r.mean(), "median": r.median(),
                                       "win": (r > 0).mean() * 100, "worst": r.min(), "since": track["date"].min()},
        "track": records(track.sort_values("date", ascending=False).head(60)),
        "pending": records(pending.sort_values("date", ascending=False)),
    }
    SITE.mkdir(exist_ok=True)
    (SITE / "data.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def publish_site():
    """저장소(GitHub Actions 실행 환경)면 docs/·reports/ 변경을 커밋·push. 아니면 아무것도 안 함."""
    if not (HERE / ".git").exists():
        return
    git = ["git", "-C", str(HERE)]
    subprocess.run(git + ["add", "docs", "reports"], check=True)
    if subprocess.run(git + ["diff", "--cached", "--quiet"]).returncode:
        subprocess.run(git + ["commit", "-m", f"daily {dt.date.today()}"], check=True)
        subprocess.run(git + ["push"], check=True, timeout=120)


def send(text: str):
    token, chat = os.environ["TELEGRAM_BOT_TOKEN"], os.environ["TELEGRAM_CHAT_ID"]
    body = json.dumps({"chat_id": chat, "text": text[:4000]}).encode()
    req = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage", body,
                                 {"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=30).read()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="텔레그램 대신 화면 출력")
    p.add_argument("--force", action="store_true", help="휴장일 확인 건너뛰기")
    a = p.parse_args()
    out = send if not a.dry_run else print
    now = dt.datetime.now(KST)
    cfg, provider, out_dir = Config(), FDRProvider(), HERE / "reports"
    try:
        if not a.force and not is_trading_day(now.date()):
            print("휴장일 — 건너뜀")
            return
        out_dir.mkdir(exist_ok=True)
        track = update_track(provider, cfg, out_dir)   # 오늘 스크리닝 전에: 어제 후보의 오늘 시가 결과 기록
        res, _, path = run(cfg, provider, now=now, verbose=False)
        export_site(res, cfg, now, track, out_dir)
        publish_site()
        link = os.environ.get("SITE_URL", "")
        out(build_message(res, cfg, now, track) + (f"\n\n{link}" if link else ""))
    except Exception:
        out(f"[종가베팅 스크리너 실패] {now:%Y-%m-%d %H:%M}\n{traceback.format_exc()[-1500:]}")
        raise


if __name__ == "__main__":
    main()
