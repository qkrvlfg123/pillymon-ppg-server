# -*- coding: utf-8 -*-
"""
stress_judge.py — 손가락 PPG 스트레스 판정 함수
================================================
※ 이 파일은 WESAD와 무관합니다. 판정에 쓰는 모델은 MAUS로 학습한
   ppg_stress_base_maus.pkl 입니다. 이 모듈은 그 모델을 불러 점수·판정을 낼 뿐입니다.
   (예전에는 이 함수가 wesad_train.py 안에 있어 이름이 헷갈렸어요.)

judge_stress(hrv, bundle_path, personal_calm_hrv):
  hrv               : ppg_pipeline.process_ppg(...)["hrv"]  (이번 측정)
  bundle_path       : 학습된 모델 파일 (기본 ppg_stress_base_maus.pkl)
  personal_calm_hrv : 그 유저의 '차분한(baseline)' 측정 HRV 딕셔너리 리스트. 없으면 콜드스타트.
  반환: {score(0~100), verdict(confirm|deny), used_reference(personal|coldstart|absolute),
         cutoff, model_version}
"""
import numpy as np


def judge_stress(hrv: dict, bundle_path: str = "ppg_stress_base_maus.pkl",
                 personal_calm_hrv: list | None = None) -> dict:
    import joblib
    b = joblib.load(bundle_path)
    x = np.array([hrv[f] for f in b["features"]], dtype=float)

    if b.get("mode") == "personal":
        n = len(personal_calm_hrv or [])
        if n >= b.get("min_personal_n", 5):
            calm = np.array([[c[f] for f in b["features"]] for c in personal_calm_hrv], dtype=float)
            mu, sd = calm.mean(0), calm.std(0); sd[sd == 0] = 1.0
            used = "personal"
        else:  # 콜드스타트: 인구 평균/표준편차 사용 (첫날부터 작동)
            mu = np.array([b["pop_stats"][f][0] for f in b["features"]])
            sd = np.array([b["pop_stats"][f][1] for f in b["features"]])
            used = "coldstart"
        x = (x - mu) / sd
    else:
        used = "absolute"

    p = float(b["model"].predict_proba(x.reshape(1, -1))[0, 1])
    return {"score": round(p * 100),
            "verdict": "confirm" if p >= b["cutoff_prob"] else "deny",
            "used_reference": used,
            "cutoff": b["cutoff_score"],
            "model_version": b["model_version"]}
