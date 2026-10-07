# overnight-screener

종가 매수 → 다음날 시가 매도(오버나이트) 조건에 맞는 한국 주식 후보를 골라 근거와 함께 리포트로 남깁니다.
매수 지시가 아니라 후보를 좁히는 1차 스크리너입니다. 조건과 규칙은 [SKILL.md](.claude/skills/overnight-screener/SKILL.md) 참고.

## 설치 (내 PC)

```bash
pip install finance-datareader pandas numpy holidays
```

## 실행

```bash
python screener.py                     # 기본 조건, reports/ 에 YYYY-MM-DD_HHMM.md / .csv 저장
python screener.py --min-amount 200 --top 15
python screener.py --help              # 모든 조건 옵션
python test_screener.py                # 합성 데이터로 로직 검증 (네트워크 불필요)
```

## 매일 자동 실행 + 텔레그램

```bash
python daily.py --dry-run               # 전송 대신 화면 출력
```

- GitHub Actions(`.github/workflows/daily.yml`)가 평일 14:40 KST에 `daily.py` 실행 (KRX 휴장일 건너뜀). PC 꺼져 있어도 됨
- 결과(`docs/`, `reports/`)는 Actions가 저장소에 커밋 → GitHub Pages(`/docs`)에 반영
- 지난 후보의 실제 결과(종가 매수 → 다음날 시가)를 `reports/track.csv`에 쌓고 메시지에 성적 요약을 붙임
- 저장소 Secrets에 `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` 필요
- 웹 대시보드: `docs/` (index.html + data.json). `docs/`가 GitHub Pages 저장소면 매일 자동 push, `SITE_URL` 환경변수를 주면 텔레그램 메시지에 링크 첨부

권장 실행 시각: 평일 15:00~15:15 KST. 데이터는 KRX(`data.krx.co.kr`)와 네이버 금융(`fchart.stock.naver.com`)에서 받습니다.
