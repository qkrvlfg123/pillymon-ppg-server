# pillymon-ppg-server

> ⚠️ **실험 단계 (Experimental)** — 연구·검증용 프로토타입입니다. 판정 결과는 의료적 진단이 아니며, API와 모델은 예고 없이 바뀔 수 있습니다.

**PILLYMON**의 PPG(광용적맥파, 심박 생체신호) 처리 서버입니다.
스마트폰 카메라에 손가락을 대고 측정한 원시 PPG 파형을 받아 HRV(심박변이도) 지표를 계산하고, 학습된 모델로 스트레스(부하) 점수를 판정한 뒤 결과를 구글 시트에 기록합니다.

- 메인 레포: [qkrvlfg123/keyboard-mouse-1st-modeling](https://github.com/qkrvlfg123/keyboard-mouse-1st-modeling)

## 스택

| 구분 | 사용 기술 |
| --- | --- |
| 언어 | Python 3 |
| 웹 서버 | Flask, flask-cors, gunicorn |
| 신호 처리 | NumPy, SciPy (Butterworth 밴드패스 + `filtfilt`, 큐빅 스플라인 업샘플, Welch PSD, `find_peaks`) |
| 판정 모델 | scikit-learn `StandardScaler` + `LogisticRegression` (joblib 번들 `ppg_stress_base_maus.pkl`) |
| 저장소 | Google Sheets (gspread + google-auth 서비스 계정) |
| 배포 | Render Web Service (`Procfile`) |

## 처리 흐름

```
원시 PPG 파형 + fps
  → 전처리: 0.5–4 Hz 밴드패스(제로위상) → 256 Hz 업샘플 → 정규화
  → 신호품질(SQI) 검사          ── 불량이면 status=low_quality (재측정 요청)
  → 피크 검출 → RR 간격 → 아티팩트 보정  ── 박동 부족이면 status=insufficient_beats
  → HRV 피처: 시간영역(mean_hr, rmssd, sdnn, pnn50) · Poincaré(sd1, sd2) · 주파수영역(lf, hf, lf_hf)
  → 스트레스 판정 (rmssd, sdnn, pnn50 → 0–100 점수)
  → 구글 시트에 한 줄 기록
```

30–60초 측정에서는 RMSSD와 평균 심박만 신뢰할 수 있습니다. SDNN과 주파수영역 지표는 계산은 하지만, 약 2분 미만이면 신뢰도가 낮은 것으로 표시됩니다.

## 판정 모델

- 버전 `base-v3-maus-logistic`: MAUS 데이터셋의 손끝 HRV(`rmssd`, `sdnn`, `pnn50`)로 학습한 로지스틱 회귀 모델입니다.
- 개인 기준(personal z-score) 방식입니다. 그 사용자의 평상시 측정이 5회 이상 있으면 개인 평균·표준편차로 정규화합니다. 5회 미만이면 인구 통계값을 쓰는 콜드스타트로 판정합니다.
- 점수 53 이상이면 `confirm`(스트레스)으로 판정합니다. 등급은 75 이상 `과부하`, 53 이상 `중간`, 그 미만 `저부하`입니다.
- LOSO AUC 약 0.71로, 아직 검증 단계의 1차 모델입니다.

## API

### `GET /health`

```json
{ "ok": true, "model_loaded": true, "sheet": true }
```

### `POST /measure`

요청:

```json
{
  "user_id": "KCS1234",
  "fps": 30,
  "trigger": "manual",
  "waveform": [123.4, 123.9, ...],
  "personal_calm_hrv": [ { "rmssd": 42.1, "sdnn": 51.0, "pnn50": 18.2 }, ... ]
}
```

`personal_calm_hrv`는 선택 항목입니다. 없으면 콜드스타트로 판정합니다.

응답 (성공):

```json
{ "status": "ok", "id": "a1b2c3d4e5f6", "score": 61, "verdict": "confirm", "grade": "중간", "used_reference": "coldstart" }
```

신호 품질이 나쁘면 `status`가 `low_quality` 또는 `insufficient_beats`이고, `"action": "재측정 요청"`이 함께 옵니다.

## 파일 구성

| 파일 | 역할 |
| --- | --- |
| `server_sheet.py` | Flask 앱 (`/`, `/health`, `/measure`), 구글 시트 기록 |
| `ppg_pipeline.py` | PPG 파형 → HRV 피처 파이프라인 |
| `stress_judge.py` | 모델 번들을 불러 점수·판정 산출 |
| `ppg_stress_base_maus.pkl` | 학습된 판정 모델 번들 |
| `Procfile` | `web: gunicorn server_sheet:app` |

## 환경변수

| 이름 | 설명 |
| --- | --- |
| `SHEET_ID` | 결과를 기록할 구글 시트 ID (URL의 `/d/`와 `/edit` 사이) |
| `GOOGLE_CREDENTIALS_JSON` | 서비스 계정 JSON 내용 전체. 로컬에서는 이 값 대신 `service_account.json` 파일을 두어도 됩니다. |
| `PORT` | 로컬 실행 포트 (기본 5000, Render가 자동 지정) |

구글 시트는 서비스 계정 이메일에 편집 권한으로 공유되어 있어야 합니다. `service_account.json`은 `.gitignore`에 포함되어 있으니 절대 커밋하지 마세요.

## 로컬 실행

```bash
pip install -r requirements.txt
export SHEET_ID=<시트 ID>            # Windows PowerShell: $env:SHEET_ID="<시트 ID>"
python server_sheet.py               # http://localhost:5000
```

## 알려진 한계 / 향후 개선

- **현재는 콜드스타트 판정만 동작합니다.** 모든 측정이 인구 평균·표준편차 기준으로 판정됩니다(`used_reference: "coldstart"`).
- **개인 기준 전환이 작동하지 않습니다.** 측정 페이지가 `personal_calm_hrv`를 보내지 않기 때문에, 평상시 측정이 5회 이상 쌓여도 개인 기준(`personal`)으로 바뀌지 않습니다.
- **향후 과제는 프론트 연동입니다.** 측정 페이지가 사용자의 평상시 HRV 기록을 모아 `personal_calm_hrv`로 함께 보내도록 연동해야 합니다.
