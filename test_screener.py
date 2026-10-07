#!/usr/bin/env python3
"""가짜 데이터로 스크리너 로직을 검증한다 (네트워크 불필요).

  python3 test_screener.py
"""
import datetime as dt
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from screener import Config, KST, add_indicators, past_stats, run, signal_mask


def make_hist(n=300, seed=0, start_price=10000):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-10-06", periods=n)
    close = start_price * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - abs(rng.normal(0, 0.004, n)))
    vol = rng.integers(2_000_000, 3_000_000, n).astype(float)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)


def plant_signal(h, i, next_open_ret):
    """i번째 날을 신호일로 만들고, 다음날 시가를 지정 수익률로 고정."""
    prev = h["Close"].iloc[i - 1]
    c = prev * 1.05
    h.iloc[i, h.columns.get_loc("Open")] = prev * 1.005
    h.iloc[i, h.columns.get_loc("Low")] = prev * 1.0
    h.iloc[i, h.columns.get_loc("High")] = c * 1.002
    h.iloc[i, h.columns.get_loc("Close")] = c
    h.iloc[i, h.columns.get_loc("Volume")] = 10_000_000
    if i + 1 < len(h):
        h.iloc[i + 1, h.columns.get_loc("Open")] = c * (1 + next_open_ret)
        h.iloc[i + 1, h.columns.get_loc("High")] = max(h["High"].iloc[i + 1], c * (1 + next_open_ret))
        h.iloc[i + 1, h.columns.get_loc("Low")] = min(h["Low"].iloc[i + 1], c * (1 + next_open_ret))
    return h


class FakeProvider:
    name = "FAKE (테스트용 합성 데이터)"

    def __init__(self):
        self.h = {}
        # A: 상승 추세 + 오늘 신호 + 과거 신호 6회 (수익률 지정)
        a = make_hist(seed=1)
        a["Close"] *= np.linspace(0.7, 1.0, len(a)); a["Open"] *= np.linspace(0.7, 1.0, len(a))
        a["High"] *= np.linspace(0.7, 1.0, len(a)); a["Low"] *= np.linspace(0.7, 1.0, len(a))
        self.planted = [0.02, 0.01, -0.01, 0.03, 0.015, 0.005]
        for k, i in enumerate([100, 130, 160, 190, 220, 250]):
            a = plant_signal(a, i, self.planted[k])
        a = plant_signal(a, len(a) - 1, 0)
        self.h["111110"] = a
        # B: 오늘 등락률 2% 미만 → 1차 탈락
        self.h["222220"] = make_hist(seed=2)
        # C: 우선주 코드 → 탈락
        self.h["333335"] = plant_signal(make_hist(seed=3), 299, 0)
        # D: 오늘 신호지만 거래량 평범 → 일봉 단계 탈락
        d = make_hist(seed=4); d = plant_signal(d, 299, 0); d.iloc[-1, d.columns.get_loc("Volume")] = 2_500_000
        self.h["444440"] = d

    def snapshot(self):
        rows = []
        names = {"111110": "알파", "222220": "베타", "333335": "감마우", "444440": "델타"}
        for code, h in self.h.items():
            t, p = h.iloc[-1], h.iloc[-2]
            rows.append({"Code": code, "Name": names[code], "Market": "KOSPI", "Close": t.Close, "Open": t.Open,
                         "High": t.High, "Low": t.Low, "Volume": t.Volume, "Amount": t.Close * t.Volume,
                         "Marcap": 5e11, "ChagesRatio": (t.Close / p.Close - 1) * 100})
        return pd.DataFrame(rows)

    def admin_codes(self):
        return set()

    def history(self, code, start):
        return self.h[code]

    def index_change(self, symbol):
        return None


def test_all():
    fp = FakeProvider()
    cfg = Config()
    now = dt.datetime(2026, 10, 6, 16, 0, tzinfo=KST)
    with tempfile.TemporaryDirectory() as d:
        res, report, path = run(cfg, fp, now=now, out_dir=Path(d), verbose=False)
        assert path.exists()
    passed = res[res["pass"]]
    assert list(passed["Code"]) == ["111110"], passed
    assert "444440" in set(res["Code"]) and not res.set_index("Code").loc["444440", "pass"]
    a = res.set_index("Code").loc["111110"]
    # 과거 신호 통계가 심어둔 수익률과 일치하는지 (우연히 생긴 추가 신호가 없어야 함)
    h = add_indicators(fp.h["111110"]).iloc[-(cfg.lookback_days + 1):-1]
    expect = past_stats(h.loc[signal_mask(h, cfg), "NextOpenRet"], cfg.round_trip_cost)
    # 신호일은 심어둔 날 중에서만 나와야 하고(우연한 신호 없음), 수익률은 심어둔 값과 일치해야 함
    days = [h.index.get_loc(x) + (len(fp.h["111110"]) - cfg.lookback_days - 1) for x in h.index[signal_mask(h, cfg)]]
    planted_idx = dict(zip([100, 130, 160, 190, 220, 250], fp.planted))
    assert days and set(days) <= set(planted_idx), days
    assert a["sig_n"] == expect["n"] == len(days)
    manual = np.mean([(planted_idx[i] - cfg.round_trip_cost) * 100 for i in days])
    assert abs(a["sig_mean%"] - manual) < 1e-6, (a["sig_mean%"], manual)
    assert "매수·매도 지시가 아닙니다" in report
    test_track()
    print(report)
    print("\nALL TESTS PASSED")


def test_track():
    """성적 기록: 신호일 종가 → 다음날 시가, 비용 차감. 결과 없는 날(마지막 날)은 기록 안 함."""
    from daily import update_track
    fp, cfg = FakeProvider(), Config()
    a = fp.h["111110"]
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        days = [a.index[250].date().isoformat(), a.index[-1].date().isoformat()]
        for day in days:
            pd.DataFrame([{"Code": "111110", "Name": "알파", "pass": True}]).to_csv(d / f"{day}_1500.csv", index=False)
        t = update_track(fp, cfg, d)
        assert list(t["date"]) == days[:1], t
        assert abs(t["ret%"].iloc[0] - (0.005 - cfg.round_trip_cost) * 100) < 1e-6, t
        assert len(update_track(fp, cfg, d)) == 1          # 다시 돌려도 중복 없음


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    test_all()
