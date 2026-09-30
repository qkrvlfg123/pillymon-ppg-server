# -*- coding: utf-8 -*-
"""
server.py — 손가락 PPG 수집·판정 서버 (구글 시트 저장 · 클라우드 배포용)
======================================================================
폰이 보낸 원파형을 검증된 ppg_pipeline으로 처리해 HRV·판정을 계산하고,
결과 한 줄을 구글 시트에 자동 저장한다. Render 등 클라우드에 올리면
네 PC를 꺼도 24시간 수집된다.

[필요 파일 — 같은 폴더]
  ppg_pipeline.py, stress_judge.py, ppg_stress_base_maus_60s.pkl
  (구글 인증) service_account.json  ← 구글 클라우드에서 발급 (아래 설명)

[환경변수]
  SHEET_ID   : 구글 시트 문서 ID (시트 URL의 /d/ 와 /edit 사이 문자열)
  GOOGLE_CREDENTIALS_JSON : (Render용) service_account.json 내용을 통째로 넣는 값
                            (로컬 테스트면 service_account.json 파일만 있어도 됨)

[설치]
  pip install flask flask-cors numpy scipy scikit-learn joblib gspread google-auth

[실행]
  python server.py
"""
import os, json, uuid
from datetime import datetime
import numpy as np
from flask import Flask, request, jsonify

import ppg_pipeline as pp
try:
    from stress_judge import judge_stress
    MODEL_FILE = "ppg_stress_base_maus_60s.pkl" if os.path.exists("ppg_stress_base_maus_60s.pkl") else "ppg_stress_base_maus.pkl"
    HAVE_MODEL = os.path.exists(MODEL_FILE)
except Exception:
    HAVE_MODEL = False
    MODEL_FILE = "ppg_stress_base_maus_60s.pkl"

# ── 구글 시트 연결 ─────────────────────────────────────────
SHEET_ID = os.environ.get("SHEET_ID", "").strip()
SHEET_HEADER = ["timestamp", "user_id", "trigger", "self_ox", "self_level", "attempt",
                "mean_hr", "rmssd", "sdnn", "pnn50", "sd1", "score", "verdict", "grade",
                "used_reference", "fps", "n_samples", "status", "model_version", "id"]
_sheet = None

def _get_sheet():
    """구글 시트 워크시트 핸들 (첫 호출 때 연결, 헤더 없으면 추가)."""
    global _sheet
    if _sheet is not None:
        return _sheet
    if not SHEET_ID:
        return None
    import gspread
    from google.oauth2.service_account import Credentials
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    raw = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    if raw:                                   # Render: 환경변수에 JSON 통째로
        info = json.loads(raw)
        creds = Credentials.from_service_account_info(info, scopes=scopes)
    else:                                     # 로컬: 파일
        creds = Credentials.from_service_account_file("service_account.json", scopes=scopes)
    gc = gspread.authorize(creds)
    ws = gc.open_by_key(SHEET_ID).sheet1
    if ws.row_count == 0 or ws.acell("A1").value != "timestamp":
        try:
            if not ws.acell("A1").value:
                ws.update("A1", [SHEET_HEADER])
        except Exception:
            pass
    _sheet = ws
    return ws

def _append_sheet(row: dict):
    ws = _get_sheet()
    if ws is None:
        print("[경고] SHEET_ID 미설정 — 시트 저장 건너뜀"); return False
    ws.append_row([row.get(k, "") for k in SHEET_HEADER],
                  value_input_option="USER_ENTERED")
    return True


app = Flask(__name__)
try:
    from flask_cors import CORS; CORS(app)
except Exception:
    @app.after_request
    def _cors(resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type,ngrok-skip-browser-warning"
        resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        return resp


def _grade(score):
    if score >= 75: return "과부하"
    if score >= 53: return "중간"
    return "저부하"


@app.route("/")
def home():
    return "PPG 수집 서버 작동 중 (구글 시트 저장). POST /measure 로 파형을 보내세요."


@app.route("/health")
def health():
    ok_sheet = False
    try:
        ok_sheet = _get_sheet() is not None
    except Exception as e:
        return jsonify({"ok": True, "model_loaded": HAVE_MODEL, "sheet": False, "sheet_error": str(e)})
    return jsonify({"ok": True, "model_loaded": HAVE_MODEL, "sheet": ok_sheet})


@app.route("/measure", methods=["POST"])
def measure():
    data = request.get_json(force=True)
    wave = np.asarray(data.get("waveform", []), dtype=float)
    fps = float(data.get("fps", 30))
    user_id = str(data.get("user_id", "anon"))
    trigger = str(data.get("trigger", "manual"))
    calm = data.get("personal_calm_hrv")
    # 자가보고(측정 전 O/X·1~5)·재측정 횟수
    self_ox = data.get("self_ox")        # 1=받음 / 0=안받음 / None
    self_level = data.get("self_level")  # 1~5 (5=매우 많이) / None
    attempt = data.get("attempt", 1)     # 몇 번째 시도(재측정 카운트)

    mid = uuid.uuid4().hex[:12]
    ts = datetime.now().isoformat(timespec="seconds")
    row = {"id": mid, "user_id": user_id, "timestamp": ts, "trigger": trigger,
           "self_ox": self_ox if self_ox is not None else "",
           "self_level": self_level if self_level is not None else "",
           "attempt": attempt,
           "fps": fps, "n_samples": int(wave.size)}

    try:
        res = pp.process_ppg(wave, fps)
    except Exception as e:
        row["status"] = f"error:{e}"; _append_sheet(row)
        return jsonify({"status": "error", "message": str(e), "id": mid}), 200

    row["status"] = res["status"]
    out = {"status": res["status"], "id": mid}

    if res["status"] == "ok":
        h = res["hrv"]
        for k in ("mean_hr", "rmssd", "sdnn", "pnn50", "sd1"):
            row[k] = round(h[k], 2) if isinstance(h.get(k), (int, float)) else ""
        if HAVE_MODEL:
            try:
                j = judge_stress(h, MODEL_FILE, personal_calm_hrv=calm)
                row.update(score=j["score"], verdict=j["verdict"], grade=_grade(j["score"]),
                           used_reference=j["used_reference"], model_version=j["model_version"])
                out.update(score=j["score"], verdict=j["verdict"], grade=_grade(j["score"]),
                           used_reference=j["used_reference"])
            except Exception as e:
                out["model_error"] = str(e)
    else:
        out["action"] = "재측정 요청"

    try:
        _append_sheet(row)
    except Exception as e:
        out["sheet_error"] = str(e)
    return jsonify(out), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))   # Render는 PORT 환경변수를 줌
    print("=" * 56)
    print(" 손가락 PPG 수집·판정 서버 (구글 시트 저장)")
    print(f"  · 모델 로드: {'O' if HAVE_MODEL else 'X'}")
    print(f"  · 구글 시트: {'설정됨' if SHEET_ID else 'X (SHEET_ID 환경변수 없음)'}")
    print(f"  · 포트: {port}")
    print("=" * 56)
    app.run(host="0.0.0.0", port=port, debug=False)
