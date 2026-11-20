# pillymon-ppg-server

> ⚠️ **실험 단계 (Experimental)** — 연구·검증용 프로토타입입니다. 판정 결과는 의료적 진단이 아니며, API와 모델은 예고 없이 바뀔 수 있습니다.

**PILLYMON**의 PPG(광용적맥파, 심박 생체신호) 처리 서버입니다.
스마트폰 카메라에 손가락을 대고 측정한 원시 PPG 파형을 받아 HRV(심박변이도) 지표를 계산하고, 학습된 모델로 스트레스(부하) 점수를 판정한 뒤 결과를 구글 시트에 기록합니다.

- 메인 레포: `keyboard-mouse-1st-modeling` (PILLYMON 팀 비공개 저장소)

## 내가 맡은 파트

이 서버는 PILLYMON 팀 프로젝트(2팀 · 2026.09 ~ 11)에서 **DA(본인)가 직접 만든 파트**입니다.
PPG 신호 처리 파이프라인, HRV 피처 추출, MAUS 기반 스트레스 판정 모델 학습, Flask 서버·구글 시트 연동까지 전부 포함합니다.
팀 전체 구성과 다른 파트는 PILLYMON 팀 비공개 저장소(`keyboard-mouse-1st-modeling`)에서 관리합니다.

## 스택

| 구분 | 사용 기술 |
| --- | --- |
| 언어 | Python 3 |
| 웹 서버 | Flask, flask-cors, gunicorn |
| 신호 처리 | NumPy, SciPy (Butterworth 밴드패스 + `filtfilt`, 큐빅 스플라인 업샘플, Welch PSD, `find_peaks`) |
| 판정 모델 | scikit-learn `StandardScaler` + `LogisticRegression` (joblib 번들 `ppg_stress_base_maus_60s.pkl`, 없으면 `ppg_stress_base_maus.pkl`) |
| 저장소 | Google Sheets (gspread + google-auth 서비스 계정) |
| 배포 | Render Web Service (`Procfile`) |

## 처리 흐름

```
원시 PPG 파형 + fps
  → 입력 검증: fps 10–240, 30초 이상        ── 짧으면 status=too_short (재측정 요청)
  → 전처리: 0.5–4 Hz 밴드패스(제로위상) → 256 Hz 업샘플 → 정규화
  → 신호품질(SQI) 검사          ── 불량이면 status=low_quality (재측정 요청)
  → 피크 검출 → RR 간격 → 아티팩트 보정  ── 박동 부족이면 status=insufficient_beats
  → HRV 피처: 시간영역(mean_hr, rmssd, sdnn, pnn50) · Poincaré(sd1, sd2) · 주파수영역(lf, hf, lf_hf)
  → 스트레스 판정 (mean_hr, rmssd, sdnn, pnn50 → 0–100 점수)
  → 구글 시트에 한 줄 기록
```

30–60초 측정에서는 RMSSD와 평균 심박만 신뢰할 수 있습니다. SDNN과 주파수영역 지표는 계산은 하지만, 약 2분 미만이면 신뢰도가 낮은 것으로 표시됩니다.
판정 모델은 SDNN을 입력으로 쓰고 60초 창으로 학습했으므로, 60초보다 짧은 측정의 점수는 학습 조건과 다릅니다. 30초 미만은 판정하지 않습니다.

`fps`는 클라이언트가 보낸 값을 그대로 씁니다. 실제 카메라 프레임 속도와 다르면 RR 간격이 그 비율만큼 틀어져 심박과 HRV가 모두 달라지므로, 측정 페이지에서 프레임 타임스탬프로 실측한 값을 보내야 합니다.

## 판정 모델

서버는 `ppg_stress_base_maus_60s.pkl`이 있으면 그것을, 없으면 `ppg_stress_base_maus.pkl`을 불러옵니다. 현재 저장소에는 둘 다 있어 **v5가 사용됩니다.**

| 번들 | 버전 | 입력 피처 | 판정 기준 점수 | LOSO AUC |
| --- | --- | --- | --- | --- |
| `ppg_stress_base_maus_60s.pkl` (사용 중) | `base-v5-maus-hr-60s` | `mean_hr`, `rmssd`, `sdnn`, `pnn50` (60초 창) | 49 | 0.72 |
| `ppg_stress_base_maus.pkl` (이전) | `base-v3-maus-logistic` | `rmssd`, `sdnn`, `pnn50` | 53 | 0.71 |

- MAUS 데이터셋의 손끝 PPG에서 계산한 HRV로 학습한 로지스틱 회귀 모델입니다. v5에서는 평균 심박이 가장 큰 계수를 가집니다(심박이 높을수록 점수 상승).
- 개인 기준(personal z-score) 방식입니다. 그 사용자의 평상시 측정이 5회 이상 있으면 개인 평균·표준편차로 정규화합니다. 5회 미만이면 인구 통계값을 쓰는 콜드스타트로 판정합니다.
- 점수가 번들의 기준 점수(v5는 49) 이상이면 `confirm`(스트레스)으로 판정합니다. 등급은 판정과 같은 기준을 써서, `confirm`이면서 75 이상은 `과부하`, 그 외 `confirm`은 `중간`, `deny`는 `저부하`입니다.
- LOSO AUC 약 0.72는 MAUS 22명 안에서의 값입니다. MAUS의 라벨은 N-back 과제 난이도(인지 부하)이며 일상 스트레스가 아니고, 측정 기기도 전용 손끝 센서라 스마트폰 카메라와 다릅니다. 카메라 측정에서의 성능은 아직 검증하지 않았습니다.
- 판정 기준 점수는 MAUS에서 정한 임시값이며, 자체 수집 데이터로 다시 정할 예정입니다.

## 데이터

- **MAUS** — Beh, W.-K., Wu, Y.-H., & Wu, A.-Y. (2021). *MAUS: A Dataset for Mental Workload Assessment on N-back Task Using Wearable Sensor.* [arXiv:2111.02561](https://arxiv.org/abs/2111.02561) · 데이터: [IEEE DataPort](https://ieee-dataport.org/open-access/maus-dataset-mental-workload-assessment-n-back-task-using-wearable-sensor)
  - 22명이 N-back 과제(난이도별 정신적 부하)를 수행하는 동안 ECG·손끝 PPG·손목 PPG·GSR을 약 35분간 기록한 공개 데이터셋입니다.
  - 이 서버의 판정 모델은 **손끝(fingertip) PPG**에서 계산한 HRV로 학습했습니다. 스마트폰 카메라에 손가락을 대는 측정 방식과 가장 가깝기 때문입니다.
  - 원본 데이터는 라이선스(학술 목적) 문제로 저장소에 포함하지 않고, 학습된 모델 번들만 포함합니다. 학습 스크립트는 이 저장소에 없습니다.

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
{ "status": "ok", "id": "a1b2c3d4e5f6", "score": 61, "verdict": "confirm", "grade": "중간",
  "used_reference": "coldstart", "model_version": "base-v5-maus-hr-60s", "record_seconds": 60.0 }
```

신호 품질이 나쁘면 `status`가 `low_quality` 또는 `insufficient_beats`, 측정이 30초보다 짧으면 `too_short`이고, `"action": "재측정 요청"`이 함께 옵니다.
`fps`가 범위를 벗어나거나 본문이 JSON이 아니면 HTTP 400과 `status: "error"`를 돌려줍니다.

환경변수 `API_KEY`를 설정하면 `X-API-Key` 헤더가 일치하는 요청만 받습니다(불일치 시 401). 설정하지 않으면 인증 없이 누구나 호출할 수 있고 그 결과가 시트에 기록되므로, 공개 배포에서는 설정해야 합니다.

## 파일 구성

| 파일 | 역할 |
| --- | --- |
| `server_sheet.py` | Flask 앱 (`/`, `/health`, `/measure`), 구글 시트 기록 |
| `ppg_pipeline.py` | PPG 파형 → HRV 피처 파이프라인 |
| `stress_judge.py` | 모델 번들을 불러 점수·판정 산출 |
| `ppg_stress_base_maus_60s.pkl` | 학습된 판정 모델 번들 (v5, 사용 중) |
| `ppg_stress_base_maus.pkl` | 이전 번들 (v3, 위 파일이 없을 때만 사용) |
| `Procfile` | `web: gunicorn server_sheet:app` |

## 환경변수

| 이름 | 설명 |
| --- | --- |
| `SHEET_ID` | 결과를 기록할 구글 시트 ID (URL의 `/d/`와 `/edit` 사이) |
| `GOOGLE_CREDENTIALS_JSON` | 서비스 계정 JSON 내용 전체. 로컬에서는 이 값 대신 `service_account.json` 파일을 두어도 됩니다. |
| `API_KEY` | (선택) 설정하면 `/measure`에 `X-API-Key` 헤더를 요구 |
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
