#!/usr/bin/env python3
"""오버나이트(종가 매수 → 다음날 시가 매도) 후보 스크리너 — 한국 KOSPI/KOSDAQ.

원칙
  - 모든 숫자는 데이터 소스(FinanceDataReader: KRX 시세 + 네이버 일봉)에서 받아 코드로 계산한다.
  - 결과는 "조건에 맞는 후보와 근거"일 뿐, 매수/매도 지시가 아니다.
  - 과거 통계는 해당 종목의 과거 기록이며 미래를 보장하지 않는다. 표본(n)이 작으면 표시한다.

사용
  python3 screener.py                 # 기본 조건으로 실행 → reports/ 에 md, csv 저장
  python3 screener.py --top 15 --min-amount 200
  python3 screener.py --help

권장 실행 시각: 15:00~15:15 KST (장중 스냅샷 기준, 종가 동시호가 전에 후보 확인).
장 마감 후 실행하면 확정 종가 기준으로 같은 조건을 다시 확인할 수 있다.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

KST = dt.timezone(dt.timedelta(hours=9))
HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------- 설정
@dataclass
class Config:
    min_amount_eok: float = 100.0      # 당일 거래대금 하한 (억원)
    min_marcap_eok: float = 1000.0     # 시가총액 하한 (억원)
    min_price: float = 1000.0          # 주가 하한 (원)
    min_change: float = 2.0            # 당일 등락률 하한 (%)
    max_change: float = 20.0           # 당일 등락률 상한 (%) — 상한가 근처(갭 리스크·체결 불확실) 제외
    min_close_pos: float = 0.75        # 종가 위치 (종가-저가)/(고가-저가) 하한 — 고가 부근 마감
    max_upper_wick: float = 0.30       # 윗꼬리 비율 (고가-max(시가,종가))/(고가-저가) 상한
    min_vol_ratio: float = 1.5         # 당일 거래량 / 직전 20일 평균 거래량 하한
    require_trend: bool = True         # 종가 > MA5 > MA20
    max_ma20_gap: float = 25.0         # 종가가 MA20 대비 몇 % 이상 떨어져(올라) 있으면 과열로 제외
    lookback_days: int = 250           # 과거 신호 백테스트 기간 (거래일)
    min_samples: int = 5               # 과거 신호 표본이 이보다 적으면 "표본 부족"
    fee_pct: float = 0.015             # 매수·매도 각각 증권사 수수료 (%) — 본인 증권사 기준으로 수정
    sell_tax_pct: float = 0.20         # 매도 시 거래세 (%) — 연도별 세율 변동, 확인 필요
    slippage_pct: float = 0.05         # 매수·매도 각각 슬리피지 가정 (%)
    prefilter: int = 80                # 1차(스냅샷) 통과 종목 중 일봉을 받아볼 최대 개수 (거래대금 순)
    top: int = 10                      # 리포트에 올릴 후보 수

    @property
    def round_trip_cost(self) -> float:
        """왕복 비용 (소수, 예: 0.0033 = 0.33%)."""
        return (2 * self.fee_pct + self.sell_tax_pct + 2 * self.slippage_pct) / 100.0


# ---------------------------------------------------------------- 데이터 소스
class FDRProvider:
    """FinanceDataReader 래퍼. 테스트에서는 같은 인터페이스의 가짜 provider로 바꿔 끼운다."""

    name = "FinanceDataReader (KRX 상장종목 시세, 네이버 금융 일봉)"

    def __init__(self):
        import FinanceDataReader as fdr  # 지연 import: 테스트는 fdr 없이도 돈다
        self.fdr = fdr

    def snapshot(self) -> pd.DataFrame:
        df = self.fdr.StockListing("KRX")
        return df

    def admin_codes(self) -> set[str]:
        """관리종목 코드. 못 받으면 빈 집합 (리포트에 '관리종목 필터 미적용' 표시)."""
        try:
            df = self.fdr.StockListing("KRX-ADMIN")
            col = "Symbol" if "Symbol" in df.columns else "Code"
            return set(df[col].astype(str).str.zfill(6))
        except Exception:
            return set()

    def history(self, code: str, start: str) -> pd.DataFrame:
        return self.fdr.DataReader(code, start)

    def index_change(self, symbol: str) -> float | None:
        try:
            d = self.fdr.DataReader(symbol, (dt.date.today() - dt.timedelta(days=14)).isoformat())
            return float((d["Close"].iloc[-1] / d["Close"].iloc[-2] - 1) * 100)
        except Exception:
            return None


# ---------------------------------------------------------------- 스냅샷 정규화 & 1차 필터
def normalize_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ren = {"Symbol": "Code", "ChagesRatio": "ChangesRatio"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    df["Code"] = df["Code"].astype(str).str.zfill(6)
    for c in ["Close", "Open", "High", "Low", "Volume", "Amount", "Marcap", "ChangesRatio"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def prefilter(snap: pd.DataFrame, cfg: Config, admin: set[str]) -> tuple[pd.DataFrame, dict]:
    """스냅샷만으로 걸러낼 수 있는 조건. 단계별 남은 종목 수를 함께 돌려준다 (투명성)."""
    funnel = {"전체": len(snap)}
    s = snap[snap["Market"].isin(["KOSPI", "KOSDAQ", "KOSDAQ GLOBAL"])]
    funnel["KOSPI/KOSDAQ"] = len(s)
    s = s[s["Code"].str.endswith("0")]                       # 우선주(끝자리 5,7,9,K 등) 제외
    s = s[~s["Name"].str.contains("스팩|리츠", regex=True, na=False)]
    if admin:
        s = s[~s["Code"].isin(admin)]
    funnel["우선주·스팩·리츠·관리종목 제외"] = len(s)
    s = s[(s["Amount"] >= cfg.min_amount_eok * 1e8) & (s["Marcap"] >= cfg.min_marcap_eok * 1e8)
          & (s["Close"] >= cfg.min_price)]
    funnel[f"거래대금≥{cfg.min_amount_eok:g}억·시총≥{cfg.min_marcap_eok:g}억·주가≥{cfg.min_price:g}원"] = len(s)
    s = s[(s["ChangesRatio"] >= cfg.min_change) & (s["ChangesRatio"] <= cfg.max_change)]
    funnel[f"등락률 {cfg.min_change:g}~{cfg.max_change:g}%"] = len(s)
    rng = (s["High"] - s["Low"]).replace(0, np.nan)
    s = s.assign(
        ClosePos=(s["Close"] - s["Low"]) / rng,
        UpperWick=(s["High"] - s[["Open", "Close"]].max(axis=1)) / rng,
    )
    s = s[(s["ClosePos"] >= cfg.min_close_pos) & (s["UpperWick"] <= cfg.max_upper_wick)]
    funnel[f"고가권 마감(위치≥{cfg.min_close_pos:g}, 윗꼬리≤{cfg.max_upper_wick:g})"] = len(s)
    s = s.sort_values("Amount", ascending=False).head(cfg.prefilter)
    return s, funnel


# ---------------------------------------------------------------- 일봉 기반 지표 & 신호
def add_indicators(h: pd.DataFrame) -> pd.DataFrame:
    h = h.copy()
    h["MA5"] = h["Close"].rolling(5).mean()
    h["MA20"] = h["Close"].rolling(20).mean()
    h["VolAvg20"] = h["Volume"].shift(1).rolling(20).mean()     # 당일 제외 직전 20일
    h["High20"] = h["High"].shift(1).rolling(20).max()          # 당일 제외 직전 20일 최고가
    h["Chg"] = h["Close"].pct_change() * 100
    rng = (h["High"] - h["Low"]).replace(0, np.nan)
    h["ClosePos"] = (h["Close"] - h["Low"]) / rng
    h["UpperWick"] = (h["High"] - h[["Open", "Close"]].max(axis=1)) / rng
    h["VolRatio"] = h["Volume"] / h["VolAvg20"]
    h["MA20Gap"] = (h["Close"] / h["MA20"] - 1) * 100
    h["Amount"] = h["Close"] * h["Volume"]                      # 일봉엔 거래대금이 없어 근사 (종가×거래량)
    h["NextOpenRet"] = h["Open"].shift(-1) / h["Close"] - 1    # 종가 매수 → 다음날 시가 매도 (비용 전)
    return h


def signal_mask(h: pd.DataFrame, cfg: Config) -> pd.Series:
    """스크리닝과 동일한 조건을 과거 각 날짜에 적용 (시총 조건은 과거값이 없어 제외)."""
    m = (
        (h["Chg"] >= cfg.min_change) & (h["Chg"] <= cfg.max_change)
        & (h["ClosePos"] >= cfg.min_close_pos) & (h["UpperWick"] <= cfg.max_upper_wick)
        & (h["VolRatio"] >= cfg.min_vol_ratio)
        & (h["Amount"] >= cfg.min_amount_eok * 1e8) & (h["Close"] >= cfg.min_price)
        & (h["MA20Gap"] <= cfg.max_ma20_gap)
    )
    if cfg.require_trend:
        m &= (h["Close"] > h["MA5"]) & (h["MA5"] > h["MA20"])
    return m.fillna(False)


def splice_today(h: pd.DataFrame, row: pd.Series, today: pd.Timestamp) -> pd.DataFrame:
    """일봉 마지막 행을 스냅샷 값으로 맞춘다.
    - 장중 실행: 네이버 일봉에 오늘 부분 봉이 있으면 스냅샷으로 덮어쓰고, 없으면 붙인다.
    - 휴장일 실행: 스냅샷 = 마지막 거래일 봉이므로 그대로 둔다."""
    h = h[["Open", "High", "Low", "Close", "Volume"]].astype(float).copy()
    last = h.index[-1]
    snap = {k: float(row[k]) for k in ["Open", "High", "Low", "Close", "Volume"]}
    if last.normalize() >= today.normalize():
        h.loc[last, list(snap)] = list(snap.values())
    elif not (math.isclose(h["Close"].iloc[-1], snap["Close"]) and math.isclose(h["Volume"].iloc[-1], snap["Volume"])):
        h.loc[today.normalize()] = snap
    return h


def past_stats(rets: pd.Series, cost: float) -> dict:
    r = (rets.dropna() - cost) * 100
    if r.empty:
        return {"n": 0, "mean": np.nan, "median": np.nan, "win": np.nan, "worst": np.nan}
    return {"n": int(r.size), "mean": r.mean(), "median": r.median(),
            "win": (r > 0).mean() * 100, "worst": r.min()}


def evaluate(code: str, row: pd.Series, provider, cfg: Config, today: pd.Timestamp) -> dict | None:
    start = (today - pd.Timedelta(days=int(cfg.lookback_days * 1.6) + 60)).date().isoformat()
    try:
        h = provider.history(code, start)
    except Exception as e:
        return {"Code": code, "error": f"일봉 수신 실패: {e}"}
    if h is None or len(h) < 30:
        return {"Code": code, "error": "일봉 부족"}
    h = add_indicators(splice_today(h, row, today))
    t = h.iloc[-1]
    passes = bool(signal_mask(h, cfg).iloc[-1])
    hist = h.iloc[-(cfg.lookback_days + 1):-1]                 # 오늘 제외 과거 구간
    sig = past_stats(hist.loc[signal_mask(hist, cfg), "NextOpenRet"], cfg.round_trip_cost)
    base = past_stats(hist["NextOpenRet"].tail(60), cfg.round_trip_cost)
    return {
        "Code": code, "Name": row["Name"], "Market": row["Market"], "pass": passes,
        "Close": t["Close"], "Chg%": t["Chg"], "Amount억": row["Amount"] / 1e8, "Marcap억": row["Marcap"] / 1e8,
        "ClosePos": t["ClosePos"], "UpperWick": t["UpperWick"], "VolRatio": t["VolRatio"],
        "MA20Gap%": t["MA20Gap"], "Break20H": bool(t["Close"] > t["High20"]),
        "Trend": bool(t["Close"] > t["MA5"] > t["MA20"]),
        "sig_n": sig["n"], "sig_mean%": sig["mean"], "sig_median%": sig["median"],
        "sig_win%": sig["win"], "sig_worst%": sig["worst"],
        "base60_mean%": base["mean"], "base60_win%": base["win"],
        "last_bar": h.index[-1].date().isoformat(), "error": "",
    }


# ---------------------------------------------------------------- 리포트
def fmt(x, nd=2, suffix=""):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:,.{nd}f}{suffix}"


def build_report(res: pd.DataFrame, funnel: dict, cfg: Config, provider_name: str, run_at: dt.datetime,
                 admin_applied: bool, idx: dict, errors: pd.DataFrame) -> str:
    L = []
    L.append(f"# 오버나이트 후보 스크리닝 — {run_at:%Y-%m-%d %H:%M} KST\n")
    L.append("> 조건에 맞는 **후보와 근거**를 보여줄 뿐, 매수·매도 지시가 아닙니다. "
             "모든 숫자는 아래 출처에서 받아 코드로 계산했고, 실제 판단 전 HTS/MTS 원 시세로 다시 확인하세요.\n")
    if run_at.weekday() >= 5:
        session = "주말 실행 (마지막 거래일 종가)"
    elif (9, 0) <= (run_at.hour, run_at.minute) < (15, 31):
        session = "장중 스냅샷 (종가 미확정, 휴장일이면 마지막 거래일 종가)"
    elif run_at.hour < 9:
        session = "장 시작 전 (전 거래일 종가)"
    else:
        session = "장 마감 후 (종가 확정)"
    L.append(f"- 데이터 출처: {provider_name}")
    L.append(f"- 기준: {session}. 일봉 마지막 날짜는 표의 `기준일` 열 참고")
    L.append(f"- 왕복 비용 가정: {cfg.round_trip_cost*100:.3f}% (수수료 {cfg.fee_pct}%×2, 거래세 {cfg.sell_tax_pct}% [확인 필요], 슬리피지 {cfg.slippage_pct}%×2)")
    if not admin_applied:
        L.append("- ⚠ 관리종목 목록을 받지 못해 관리종목 필터는 적용되지 않았습니다 — 확인 필요")
    if idx:
        L.append("- 지수 등락 [사실]: " + ", ".join(f"{k} {fmt(v, 2, '%')}" for k, v in idx.items()))
    L.append("\n## 1. 걸러진 과정\n")
    L.append("| 단계 | 남은 종목 수 |\n|---|---:|")
    for k, v in funnel.items():
        L.append(f"| {k} | {v:,} |")

    passed = res[res["pass"]] if not res.empty else res
    L.append(f"\n## 2. 조건 통과 후보 ({len(passed)}개 중 상위 {min(cfg.top, len(passed))}개)\n")
    if passed.empty:
        L.append("오늘은 모든 조건을 통과한 종목이 없습니다. **억지로 후보를 만들지 않습니다.**\n")
    else:
        L.append("정렬: 과거 같은 신호 표본이 충분한(n≥{0}) 종목을 먼저, 그 안에서 비용 차감 평균 수익률 순. "
                 "표본 부족 종목은 그 뒤에 거래대금 순.\n".format(cfg.min_samples))
        L.append("| # | 종목 | 시장 | 종가 | 등락 | 거래대금(억) | 거래량배수 | 종가위치 | MA20 이격 | 20일 신고가 | 과거 신호 n | 평균 | 중앙값 | 승률 | 최악 | 기준일 |")
        L.append("|---:|---|---|---:|---:|---:|---:|---:|---:|:-:|---:|---:|---:|---:|---:|---|")
        for i, d in enumerate(passed.head(cfg.top).to_dict("records"), 1):
            enough = d["sig_n"] >= cfg.min_samples
            L.append(
                f"| {i} | {d['Name']} ({d['Code']}) | {d['Market']} | {fmt(d['Close'],0)} | {fmt(d['Chg%'],2,'%')} | "
                f"{fmt(d['Amount억'],0)} | {fmt(d['VolRatio'],1,'x')} | {fmt(d['ClosePos'],2)} | {fmt(d['MA20Gap%'],1,'%')} | "
                f"{'O' if d['Break20H'] else '–'} | {d['sig_n']}{'' if enough else ' ⚠'} | {fmt(d['sig_mean%'],2,'%')} | "
                f"{fmt(d['sig_median%'],2,'%')} | {fmt(d['sig_win%'],0,'%')} | {fmt(d['sig_worst%'],2,'%')} | {d['last_bar']} |")
        L.append("\n- `과거 신호` = 최근 {0}거래일 동안 이 종목에서 **같은 조건이 떴던 날** 종가 매수 → 다음날 시가 매도 수익률(비용 차감). "
                 "⚠ = 표본 {1}개 미만, 통계로 읽지 말 것.".format(cfg.lookback_days, cfg.min_samples))
        L.append("- `종가위치` 1.0 = 고가에 마감. `거래량배수` = 당일 거래량 / 직전 20일 평균.")

        L.append("\n## 3. 후보별 근거\n")
        for r in passed.head(cfg.top).to_dict("records"):
            L.append(f"**{r['Name']} ({r['Code']})**")
            L.append(f"- [사실] 등락 {fmt(r['Chg%'],2,'%')}, 거래대금 {fmt(r['Amount억'],0)}억, 거래량 20일 평균의 {fmt(r['VolRatio'],1)}배, "
                     f"종가위치 {fmt(r['ClosePos'],2)}, 윗꼬리 {fmt(r['UpperWick'],2)}, MA20 대비 {fmt(r['MA20Gap%'],1,'%')}"
                     f"{', 직전 20일 고가 돌파' if r['Break20H'] else ''}")
            if r["sig_n"] > 0:
                L.append(f"- [사실] 과거 {cfg.lookback_days}거래일 같은 신호 {r['sig_n']}회 → 익일 시가 평균 {fmt(r['sig_mean%'],2,'%')}, "
                         f"승률 {fmt(r['sig_win%'],0,'%')}, 최악 {fmt(r['sig_worst%'],2,'%')} (비용 차감)")
            else:
                L.append(f"- [사실] 과거 {cfg.lookback_days}거래일 동안 같은 신호가 뜬 적 없음")
            L.append(f"- [사실] 비교 기준: 최근 60거래일 전체 오버나이트 평균 {fmt(r['base60_mean%'],2,'%')}, 승률 {fmt(r['base60_win%'],0,'%')} (비용 차감)")
            notes = []
            if r["sig_n"] < cfg.min_samples:
                notes.append("과거 표본이 적어 신호의 통계적 의미를 판단할 수 없음")
            elif r["sig_mean%"] <= 0:
                notes.append("과거 같은 신호 후 평균이 비용 차감 시 0 이하 — 이 종목에선 조건이 유리하게 작동하지 않았음")
            if not math.isnan(r["sig_worst%"]) and r["sig_worst%"] <= -5:
                notes.append("과거 최악 갭하락이 -5% 이하 — 하방 꼬리 위험 확인 필요")
            if r["MA20Gap%"] >= 15:
                notes.append("MA20 대비 이격이 커서 단기 과열 구간일 수 있음")
            L.append("- [해석] " + ("; ".join(notes) if notes else "조건 충족, 과거 신호 통계상 특이점 없음") + "\n")

        L.append("## 4. 더 파볼 곳 (스크리너가 확인하지 못한 것)\n")
        L.append("- 오늘 급등 사유 (공시·뉴스): DART, 증권사 리포트 — 재료 소멸형 급등은 다음날 갭하락이 잦음 [해석]")
        L.append("- 밤사이 이벤트: 미국 지수·선물, 환율, 해당 업종 미국 동종주 실적 일정")
        L.append("- 다음날 예정 공시·실적·유상증자·보호예수 해제 일정")
        L.append("- 동시호가 예상 체결가와 호가 잔량 (스냅샷 이후 가격이 바뀔 수 있음)")

    if not errors.empty:
        L.append("\n## 부록. 일봉 수신 실패\n")
        for r in errors.itertuples():
            L.append(f"- {r.Code}: {r.error}")
    L.append("\n---\n이 리포트는 투자 자문이 아니며, 과거 통계는 미래 수익을 보장하지 않습니다.")
    return "\n".join(L)


# ---------------------------------------------------------------- 실행
def run(cfg: Config, provider, now: dt.datetime | None = None, out_dir: Path | None = None, verbose=True):
    now = now or dt.datetime.now(KST)
    today = pd.Timestamp(now.date())
    snap = normalize_snapshot(provider.snapshot())
    admin = provider.admin_codes()
    pre, funnel = prefilter(snap, cfg, admin)
    if verbose:
        print(f"1차 통과 {len(pre)}개 — 일봉 확인 중...", file=sys.stderr)
    rows = []
    for _, row in pre.iterrows():
        r = evaluate(row["Code"], row, provider, cfg, today)
        if r:
            rows.append(r)
    res = pd.DataFrame(rows)
    errors = res[res["error"] != ""] if not res.empty else res
    res = res[res["error"] == ""] if not res.empty else res
    if not res.empty:
        funnel[f"거래량≥{cfg.min_vol_ratio:g}배·추세·이격 (일봉 확인)"] = int(res["pass"].sum())
        res["enough"] = res["sig_n"] >= cfg.min_samples
        res = res.sort_values(["pass", "enough", "sig_mean%", "Amount억"], ascending=[False, False, False, False],
                              na_position="last")
    idx = {}
    for sym, label in [("KS11", "KOSPI"), ("KQ11", "KOSDAQ")]:
        v = provider.index_change(sym)
        if v is not None:
            idx[label] = v
    report = build_report(res, funnel, cfg, provider.name, now, bool(admin), idx, errors)
    out_dir = out_dir or HERE / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{now:%Y-%m-%d_%H%M}"
    (out_dir / f"{stem}.md").write_text(report, encoding="utf-8")
    if not res.empty:
        res.to_csv(out_dir / f"{stem}.csv", index=False, encoding="utf-8-sig")
    return res, report, out_dir / f"{stem}.md"


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔(cp949)에서 print 오류 방지
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = Config()
    for k, v in asdict(defaults).items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            p.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes", "y"), default=v)
        else:
            p.add_argument(flag, type=type(v), default=v)
    p.add_argument("--out", type=Path, default=None, help="리포트 저장 폴더 (기본: ./reports)")
    a = vars(p.parse_args())
    out = a.pop("out")
    cfg = Config(**a)
    _, report, path = run(cfg, FDRProvider(), out_dir=out)
    print(report)
    print(f"\n저장: {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
