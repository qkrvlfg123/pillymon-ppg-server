"""
ppg_pipeline.py
================
스마트폰 카메라 손가락 PPG 파형 -> HRV 피처 산출 파이프라인.

역할(전체 설계 기준):
  브라우저(PPG 웹)가 카메라 30~60초를 찍어 '프레임별 평균 픽셀값 = raw PPG 파형(1D)' +
  fps 를 서버로 넘긴다. 이 모듈은 그 파형을 받아 다음을 수행한다.

    파형 -> 전처리(필터/업샘플) -> 신호품질(SQI) -> 피크검출 -> RR(IBI)
         -> RR 아티팩트 보정 -> HRV 피처(시간/주파수/비선형)

  출력 HRV 딕셔너리가 그다음 '판정 로직(1·2번)'의 입력이 된다.
  판정(개인 baseline 대비 / 콜드스타트 폴백)은 이 모듈 밖에서 한다. (하단 인터페이스 스텁 참고)

의존성: numpy, scipy 만 사용 (블랙박스 없이 각 단계가 보이도록).
  프로덕션에서 더 견고하게 가려면 neurokit2 로 preprocess/peak/hrv 단계를 교체 가능.

측정 길이에 대한 핵심 주의:
  30~60초 측정에서는 RMSSD(그리고 평균 HR)만 신뢰할 수 있다.
  SDNN 은 길이에 민감하고, 주파수영역(특히 LF)은 최소 ~2분이 필요해
  30~60초에서는 신뢰 불가 -> 계산은 하되 reliability 플래그로 표시한다.
  따라서 스트레스 판정은 RMSSD·HR 을 주축으로 설계할 것.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps
from scipy import interpolate, stats


# ---------------------------------------------------------------------------
# 0. (선택) 영상 프레임 -> PPG 파형
#    핸드오프가 b1(브라우저가 파형 전송)이면 이 함수는 안 씀.
#    서버에서 영상을 직접 처리하게 될 경우를 대비한 참조 구현.
# ---------------------------------------------------------------------------
def extract_ppg_from_frames(frames: np.ndarray, channel: str = "g") -> np.ndarray:
    """(N_frames, H, W, 3) RGB 프레임 스택 -> 프레임별 평균 픽셀값 1D 파형.

    손가락 접촉 PPG는 보통 green 채널의 대비가 가장 좋다(red 는 포화되기 쉬움).
    """
    idx = {"r": 0, "g": 1, "b": 2}[channel]
    # 각 프레임에서 관심 채널의 공간 평균 -> 시간축 신호
    return frames[..., idx].reshape(frames.shape[0], -1).mean(axis=1)


# ---------------------------------------------------------------------------
# 1. 전처리: 밴드패스 + 제로위상 필터 + 업샘플(양자화 완화) + 정규화
# ---------------------------------------------------------------------------
def preprocess(
    raw: np.ndarray,
    fs: float,
    target_fs: float = 256.0,
    band=(0.5, 4.0),
) -> tuple[np.ndarray, float]:
    """raw PPG -> (필터링·업샘플된 신호, 새 샘플레이트).

    - 밴드패스 0.5~4Hz: 30~240 bpm 심박 대역만 통과(기저 흔들림/고주파 노이즈 제거).
    - filtfilt: 위상 지연 없음(피크 위치가 안 밀림 -> RR 타이밍 정확).
    - 업샘플 256Hz: 30fps 원신호는 RR 양자화가 ~33ms라 RMSSD가 뭉개짐.
      큐빅 스플라인 보간으로 피크 위치 해상도를 올려 이 손실을 줄인다.
    """
    raw = np.asarray(raw, dtype=float)
    raw = raw[np.isfinite(raw)]
    if raw.size < int(fs * 5):  # 최소 5초는 있어야 의미 있음
        raise ValueError(f"신호가 너무 짧음: {raw.size} samples @ {fs}Hz")

    # (a) 밴드패스 (Butterworth 2차, 제로위상)
    nyq = fs / 2.0
    low, high = band[0] / nyq, min(band[1] / nyq, 0.99)
    b, a = sps.butter(2, [low, high], btype="band")
    filtered = sps.filtfilt(b, a, raw)

    # (b) 큐빅 스플라인으로 target_fs 업샘플
    n = filtered.size
    t_old = np.arange(n) / fs
    t_new = np.arange(0.0, t_old[-1], 1.0 / target_fs)
    spline = interpolate.CubicSpline(t_old, filtered)
    up = spline(t_new)

    # (c) 정규화(z-score) -> 이후 피크검출/SQI 스케일 안정
    up = (up - up.mean()) / (up.std() + 1e-12)
    return up, target_fs


# ---------------------------------------------------------------------------
# 2. 신호품질 지수(SQI): 미달이면 '재측정' 반환
# ---------------------------------------------------------------------------
def compute_sqi(sig: np.ndarray, fs: float) -> dict:
    """여러 SQI를 계산하고 통과 여부를 반환.

    - skewness SQI: 깨끗한 PPG는 systolic 상승이 가팔라 양의 왜도를 보인다
      (Elgendi 2016). |skew| 이 작거나 음수면 노이즈/모션 의심.
    - band power ratio: 심박 대역(0.7~3.5Hz) 파워 / 전체 파워. 높을수록 깨끗.
    - HR plausibility: 우세 주파수로 추정한 HR 이 40~180 bpm 범위인지.
    """
    skew = float(stats.skew(sig))

    f, pxx = sps.welch(sig, fs=fs, nperseg=min(len(sig), int(fs * 8)))
    band = (f >= 0.7) & (f <= 3.5)
    band_ratio = float(pxx[band].sum() / (pxx.sum() + 1e-12))

    dom_f = float(f[band][np.argmax(pxx[band])]) if band.any() else 0.0
    hr_est = dom_f * 60.0

    is_valid = (skew > 0.0) and (band_ratio > 0.5) and (40 <= hr_est <= 180)
    return {
        "skew_sqi": skew,
        "band_power_ratio": band_ratio,
        "hr_spectral_est": hr_est,
        "is_valid": bool(is_valid),
    }


# ---------------------------------------------------------------------------
# 3. 피크검출 (systolic peak)
# ---------------------------------------------------------------------------
def detect_peaks(sig: np.ndarray, fs: float, max_hr: float = 180.0) -> np.ndarray:
    """systolic 피크 인덱스.

    - distance: 최소 박동 간격 = 60/max_hr 초 -> 이중검출 방지.
    - prominence: 신호 표준편차 비례 적응형 -> 진폭 변동에 견고.
    """
    min_dist = int(fs * 60.0 / max_hr)
    prom = 0.4 * np.std(sig)
    peaks, _ = sps.find_peaks(sig, distance=min_dist, prominence=prom)
    return peaks


# ---------------------------------------------------------------------------
# 4. RR(IBI) 산출 + 아티팩트 보정
# ---------------------------------------------------------------------------
def compute_rr(peaks: np.ndarray, fs: float) -> np.ndarray:
    """피크 인덱스 -> RR 간격(ms)."""
    peak_times = peaks / fs
    return np.diff(peak_times) * 1000.0  # ms


def correct_rr(rr_ms: np.ndarray, tol: float = 0.25, reject_ratio: float = 0.30):
    """RR 아티팩트 보정 (Kubios 계열, 보간 방식).

    핵심 개선: 아티팩트를 '삭제'하지 않고 '보간'한다.
      삭제하면 양옆 박동이 붙어 그 차분이 커져 RMSSD가 부풀어(=이전 버그).
      보간하면 길이 유지 + 차분이 정상화돼 RMSSD가 생리 범위로 돌아온다.

    1) 생리 범위: [300, 1500] ms (40~200 bpm). 밖은 아티팩트로 표시.
    2) 국소 중앙값 대비 tol(25%) 이상 이탈 박동을 2회 반복 탐지.
    3) 아티팩트를 이웃 정상 박동에서 선형 보간해 대체.
    4) 아티팩트 비율이 reject_ratio(30%) 초과면 rejected=True (창 통째 폐기 대상).

    반환: (보정된 RR, {n_removed, removed_ratio, rejected})
    """
    rr = np.asarray(rr_ms, dtype=float)
    n0 = rr.size
    if n0 < 3:
        return rr, {"n_removed": 0, "removed_ratio": 0.0, "rejected": n0 < 2}

    art = ~((rr >= 300) & (rr <= 1500))            # (1) 물리적 범위
    idx = np.arange(n0)
    for _ in range(2):                              # (2) 국소 중앙값 이탈 반복 탐지
        good = ~art
        if good.sum() < 3:
            break
        rr_ref = np.interp(idx, idx[good], rr[good])
        for i in range(n0):
            lo, hi = max(0, i - 3), min(n0, i + 4)
            local_med = np.median(rr_ref[lo:hi])
            if abs(rr[i] - local_med) > tol * local_med:
                art[i] = True

    good = ~art
    removed_ratio = float(art.mean())
    if good.sum() < 3:
        return rr[good], {"n_removed": int(art.sum()), "removed_ratio": removed_ratio, "rejected": True}

    rr_corr = rr.copy()                             # (3) 보간 대체
    rr_corr[art] = np.interp(idx[art], idx[good], rr[good])
    info = {"n_removed": int(art.sum()), "removed_ratio": removed_ratio,
            "rejected": removed_ratio > reject_ratio}  # (4) 창 폐기 판단
    return rr_corr, info


# ---------------------------------------------------------------------------
# 5. HRV 피처
# ---------------------------------------------------------------------------
def time_domain_hrv(rr: np.ndarray) -> dict:
    """시간영역 HRV. 30~60초에서 RMSSD·mean_hr 이 신뢰 축."""
    if rr.size < 2:
        return {k: np.nan for k in ("mean_rr", "mean_hr", "sdnn", "rmssd", "pnn50")}
    diff = np.diff(rr)
    return {
        "mean_rr": float(np.mean(rr)),
        "mean_hr": float(60000.0 / np.mean(rr)),
        "sdnn": float(np.std(rr, ddof=1)),
        "rmssd": float(np.sqrt(np.mean(diff**2))),
        "pnn50": float(np.mean(np.abs(diff) > 50) * 100.0),
    }


def poincare_hrv(rr: np.ndarray) -> dict:
    """비선형: Poincaré SD1(단기변이=부교감), SD2(장기변이)."""
    if rr.size < 3:
        return {"sd1": np.nan, "sd2": np.nan}
    diff = np.diff(rr)
    sd1 = float(np.sqrt(0.5) * np.std(diff, ddof=1))
    sdnn = np.std(rr, ddof=1)
    sd2 = float(np.sqrt(max(2 * sdnn**2 - 0.5 * np.std(diff, ddof=1) ** 2, 0)))
    return {"sd1": sd1, "sd2": sd2}


def frequency_domain_hrv(rr: np.ndarray, min_seconds_for_lf: float = 120.0) -> dict:
    """주파수영역 HRV (LF/HF).

    RR tachogram 을 4Hz 로 보간 후 Welch PSD.
    ★ 신뢰도 경고: LF(0.04~0.15Hz)는 최소 ~2분 기록 필요.
      30~60초 측정에서는 lf_reliable=False -> 판정에 쓰지 말 것.
    """
    out = {"lf": np.nan, "hf": np.nan, "lf_hf": np.nan, "lf_reliable": False}
    if rr.size < 4:
        return out

    t = np.cumsum(rr) / 1000.0  # 박동 시각(초)
    duration = float(t[-1] - t[0])
    fs_i = 4.0
    t_i = np.arange(t[0], t[-1], 1.0 / fs_i)
    if t_i.size < 8:
        return out
    rr_i = np.interp(t_i, t, rr)
    rr_i = rr_i - rr_i.mean()

    f, pxx = sps.welch(rr_i, fs=fs_i, nperseg=min(len(rr_i), 256))
    lf = float(pxx[(f >= 0.04) & (f < 0.15)].sum())
    hf = float(pxx[(f >= 0.15) & (f < 0.40)].sum())
    out.update(
        lf=lf,
        hf=hf,
        lf_hf=float(lf / hf) if hf > 0 else np.nan,
        lf_reliable=bool(duration >= min_seconds_for_lf),
        record_seconds=duration,
    )
    return out


# ---------------------------------------------------------------------------
# 6. 메인 엔트리: 파형 -> 전체 결과
# ---------------------------------------------------------------------------
def process_ppg(raw: np.ndarray, fs: float) -> dict:
    """raw PPG 파형(+fs) -> HRV 피처 + 품질/보정 메타.

    반환 딕셔너리 구조:
      status: "ok" | "low_quality" | "insufficient_beats"
      sqi:    신호품질 지표
      hrv:    {mean_hr, rmssd, sdnn, pnn50, sd1, sd2, lf, hf, lf_hf, ...}
      meta:   {n_beats, rr_correction, reliable_metrics}
    이 hrv 가 판정 로직(1·2번)의 입력.
    """
    # 1) 전처리
    sig, new_fs = preprocess(raw, fs)

    # 2) 품질
    sqi = compute_sqi(sig, new_fs)
    if not sqi["is_valid"]:
        return {"status": "low_quality", "sqi": sqi, "hrv": None,
                "meta": {"action": "재측정 요청"}}

    # 3) 피크 -> 4) RR -> 보정
    peaks = detect_peaks(sig, new_fs)
    rr_raw = compute_rr(peaks, new_fs)
    rr, corr = correct_rr(rr_raw)

    if rr.size < 5:  # HRV 최소 박동 수 미달
        return {"status": "insufficient_beats", "sqi": sqi, "hrv": None,
                "meta": {"n_beats": int(rr.size + 1), "action": "재측정 요청"}}
    if corr.get("rejected"):  # 아티팩트 과다 창 폐기 (모션 등)
        return {"status": "low_quality", "sqi": sqi, "hrv": None,
                "meta": {"rr_correction": corr, "action": "재측정 요청(아티팩트 과다)"}}

    # 5) HRV
    hrv = {}
    hrv.update(time_domain_hrv(rr))
    hrv.update(poincare_hrv(rr))
    hrv.update(frequency_domain_hrv(rr))

    # 30~60초에서 신뢰 가능한 지표만 표시(판정 설계 가이드)
    reliable = ["mean_hr", "rmssd", "sd1"]
    if hrv.get("lf_reliable"):
        reliable += ["lf", "hf", "lf_hf", "sdnn"]

    return {
        "status": "ok",
        "sqi": sqi,
        "hrv": hrv,
        "meta": {
            "n_beats": int(rr.size + 1),
            "rr_correction": corr,
            "reliable_metrics": reliable,
        },
    }


# ---------------------------------------------------------------------------
# 7. 판정 인터페이스 스텁 (1·2번에서 구현)
#    오늘은 시그니처만. 다음 단계에서 개인 baseline/콜드스타트 로직 채움.
# ---------------------------------------------------------------------------
def judge_stress(hrv: dict, personal_baseline: dict | None,
                 coldstart_ref: dict) -> dict:
    """HRV -> 스트레스 확인 결과 (confirm/deny).

    설계(다음 단계에서 구현):
      - personal_baseline(②) 있으면: 그 사람 평소 RMSSD/HR 대비 상대 편차로 판정
      - 없으면(콜드스타트): coldstart_ref(①, WESAD/문헌) 절대 임계로 판정
      - 짧은 측정이므로 RMSSD·HR 주축 (주파수영역은 lf_reliable=True일 때만)
    """
    raise NotImplementedError("1·2번 단계에서 구현")


# ---------------------------------------------------------------------------
# 자체 검증: 합성 PPG로 파이프라인이 실제로 도는지 확인
#   (실데이터 없이도, 알려진 HR/RMSSD를 넣어 복원되는지 본다)
# ---------------------------------------------------------------------------
def _make_synthetic_ppg(mean_hr=72.0, rmssd_ms=42.0, seconds=45.0,
                        fps=30.0, noise=0.05, seed=0):
    """알려진 HR·RMSSD를 갖는 합성 손가락 PPG(30fps) 생성."""
    rng = np.random.default_rng(seed)
    mean_rr = 60000.0 / mean_hr  # ms
    # 연속 RR 차분의 std = rmssd 가 되도록 생성 (RMSSD=sqrt(mean(diff^2)))
    n_beats = int(seconds / (mean_rr / 1000.0)) + 2
    steps = rng.normal(0, rmssd_ms / np.sqrt(2), n_beats)
    rr = mean_rr + np.cumsum(steps) * 0.0 + steps  # 평균 근처에서 변동
    rr = np.clip(rr, 400, 1200)
    beat_times = np.cumsum(rr) / 1000.0
    beat_times = beat_times[beat_times < seconds]

    t = np.arange(0, seconds, 1.0 / fps)
    sig = np.zeros_like(t)
    for bt in beat_times:
        # PPG 단일 박동: systolic(큰 가우시안) + dicrotic(작은 가우시안)
        sig += 1.0 * np.exp(-((t - bt) ** 2) / (2 * 0.045**2))
        sig += 0.35 * np.exp(-((t - (bt + 0.22)) ** 2) / (2 * 0.05**2))
    sig += 0.15 * np.sin(2 * np.pi * 0.15 * t)     # 기저 흔들림
    sig += rng.normal(0, noise, t.size)             # 백색 노이즈
    true_rmssd = float(np.sqrt(np.mean(np.diff(rr[:len(beat_times)]) ** 2)))
    return sig, fps, {"true_mean_hr": mean_hr, "true_rmssd": true_rmssd}


if __name__ == "__main__":
    print("=" * 60)
    print("합성 PPG로 파이프라인 자체 검증")
    print("=" * 60)
    for hr, rmssd in [(72, 42), (88, 22), (65, 60)]:
        raw, fps, truth = _make_synthetic_ppg(mean_hr=hr, rmssd_ms=rmssd)
        res = process_ppg(raw, fps)
        print(f"\n[입력]  HR={hr}bpm, RMSSD~{truth['true_rmssd']:.1f}ms  (30fps, 45s)")
        print(f"[상태]  {res['status']}  | SQI valid={res['sqi']['is_valid']} "
              f"(skew={res['sqi']['skew_sqi']:.2f}, "
              f"band={res['sqi']['band_power_ratio']:.2f})")
        if res["status"] == "ok":
            h = res["hrv"]
            print(f"[복원]  HR={h['mean_hr']:.1f}bpm, RMSSD={h['rmssd']:.1f}ms, "
                  f"SDNN={h['sdnn']:.1f}, SD1={h['sd1']:.1f}, pNN50={h['pnn50']:.1f}%")
            print(f"[박동]  {res['meta']['n_beats']}개, "
                  f"RR제거 {res['meta']['rr_correction']['n_removed']}개")
            print(f"[신뢰]  30~60초에서 믿을 지표: {res['meta']['reliable_metrics']}")
            print(f"        LF/HF 신뢰={h['lf_reliable']} (짧아서 판정에 쓰지 말 것)")
