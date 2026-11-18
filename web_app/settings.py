"""Web controls, presets, and validation shared by the server and GPU worker."""

from math import isfinite

GROUPS = {
    "main": "핵심 설정",
    "blend": "신원 혼합",
    "reconstruction": "재구성 · 고급",
    "noise": "노이즈 보정 · 고급",
    "watermark": "워터마크 · 고급",
}
ALL_BLOCKS = tuple(f"b{2**power}" for power in range(2, 11))
LORA_BLOCKS = (("b16", "b32"), ALL_BLOCKS[1:6], ALL_BLOCKS)
CONV_BLOCKS = ((), ("b16", "b32"), ALL_BLOCKS[1:])
PROMPTS = {"nose": "코", "eyes": "눈", "eyebrows": "눈썹", "lips": "입술"}


def slider(group, label, lo, hi, step, help_text, *, unit="", levels=None):
    result = {
        "group": group, "label": label, "kind": "range", "min": lo, "max": hi,
        "step": step, "help": help_text, "unit": unit,
    }
    if levels:
        result["levels"] = levels
    return result


FIELDS = {
    "rank": slider("main", "LoRA 랭크", 2, 32, 1,
                   "높을수록 변환에 사용할 수 있는 파라미터가 늘어납니다. 논문은 8을 사용했고, 이번 사진의 튜닝은 16입니다."),
    "lora_scope": slider("main", "LoRA 적용 범위", 0, 2, 1,
                         "어느 해상도 블록을 조정할지 정합니다. 전체 블록은 배경과 세부 질감까지 맞추기 쉽지만 신원 특징도 달라질 수 있습니다.",
                         levels=["논문 중간 블록 · b16·b32", "확장 중간 블록 · b8–b128", "전체 블록 · b4–b1024"]),
    "conv_scope": slider("main", "합성 가중치 학습 범위", 0, 2, 1,
                         "LoRA에 더해 합성층 원래 가중치를 학습합니다. 넓히면 외형을 더 잘 맞출 여지가 생기고 VRAM 사용량도 늘 수 있습니다.",
                         levels=["끄기 · LoRA만", "중간 블록 · b16·b32", "넓게 · b8–b1024"]),
    "reconstruction_steps": slider("main", "재구성 최대 반복", 1, 2400, 1,
                                   "값을 높이면 사진을 더 오래 맞춥니다. LPIPS 목표에 먼저 도달하면 멈춥니다. 반복 증가가 SSIM 상승을 보장하지는 않습니다.", unit="회"),
    "noise_steps": slider("main", "노이즈 맵 보정 반복", 0, 1600, 1,
                          "StyleGAN의 사진별 노이즈 맵을 학습해 머리카락과 배경 같은 세부 픽셀을 맞춥니다. 0회는 보정을 건너뜁니다.", unit="회"),
    "epsilon": slider("main", "워터마크 픽셀 변화 상한 ε", 0.001, 0.05, 0.001,
                      "픽셀 값이 0~1일 때 워터마크가 한 채널을 움직일 수 있는 최대값입니다. 크게 하면 인식 점수가 내려갈 수 있고 픽셀 차이도 커질 수 있습니다. 자동 모드에서는 후보의 중심값입니다."),
    "tau": slider("blend", "FaRL 유사도 임계값 τ", 0.05, 1.0, 0.01,
                  "외형이 비슷한 신원 후보만 고르는 기준입니다. 통과한 후보가 없으면 이 구현은 유사도 상위 후보를 사용합니다. 논문 값은 0.75입니다."),
    "candidate_count": slider("blend", "후보 수 m", 2, 60, 1,
                              "혼합 후보를 이 개수까지 사용하고, 속성 방향 계산에도 상위·하위 각각 이 개수만큼 사용합니다. 논문 값은 30입니다.", unit="개"),
    "order_weight": slider("blend", "속성 순서 가중치 λR", 0.0, 5.0, 0.05,
                           "속성 방향을 학습할 때 후보의 텍스트 유사도 순서를 얼마나 반영할지 정합니다. 논문 값은 1입니다."),
    "learning_rate": slider("reconstruction", "LoRA 학습률", 0.0001, 0.006, 0.0001,
                            "재구성 한 단계에서 LoRA 가중치를 얼마나 크게 갱신할지 정합니다. 너무 높으면 결과가 불안정할 수 있습니다. 논문 값은 0.0003입니다."),
    "conv_learning_rate": slider("reconstruction", "합성 가중치 학습률", 0.0001, 0.004, 0.0001,
                                 "합성 가중치 학습 범위를 켰을 때 적용되는 보폭입니다. 크면 빨리 맞출 수 있지만 색이나 질감이 흔들릴 수 있습니다."),
    "identity_lock": slider("reconstruction", "혼합 신원 유지 λID", 0.0, 1.0, 0.01,
                            "재구성하면서 처음 혼합된 얼굴의 신원 특징을 유지하는 정도입니다. 크게 하면 원본에 맞추는 작업과 충돌할 수 있습니다. 논문 값은 0.1입니다."),
    "l2_weight": slider("reconstruction", "L2 픽셀 오차 가중치", 0.0, 6.0, 0.1,
                        "원본 RGB 값에 얼마나 가깝게 맞출지 정합니다. 높이면 픽셀 오차가 줄 수 있지만 디테일이 부드러워질 수 있습니다."),
    "ssim_weight": slider("reconstruction", "SSIM 손실 가중치", 0.0, 2.0, 0.05,
                          "밝기와 윤곽, 질감의 구조를 원본에 맞추는 비중입니다. 0은 공개 구현과 같이 이 추가 손실을 끕니다."),
    "lpips_threshold": slider("reconstruction", "LPIPS 조기 종료 목표", 0.0, 0.20, 0.005,
                              "재구성 도중 LPIPS가 이 값 이하가 되면 종료합니다. 낮을수록 더 오래 실행될 수 있습니다. 논문 보충자료 값은 0.06입니다."),
    "cosine_decay": {"group": "reconstruction", "label": "학습률 점진 감소", "kind": "toggle",
                     "help": "재구성 동안 학습률을 천천히 줄여 뒤로 갈수록 작은 수정만 하게 합니다."},
    "noise_lr": slider("noise", "노이즈 맵 학습률", 0.005, 0.12, 0.005,
                       "노이즈 맵을 한 번에 얼마나 바꿀지 정합니다. 너무 높으면 세부 질감이 흔들릴 수 있습니다."),
    "noise_reg": slider("noise", "노이즈 맵 규제 가중치", 0.0, 0.02, 0.001,
                        "노이즈 맵의 이웃 값이 반복적인 무늬를 이루는 것을 얼마나 억제할지 정합니다."),
    "noise_ssim": slider("noise", "노이즈 보정 SSIM 비중", 0.0, 3.0, 0.05,
                         "노이즈 보정 단계에서 외형 구조를 원본에 맞추는 비중입니다. 높이면 결과가 입력 사진에 더 강하게 맞춰질 수 있습니다."),
    "auto_sweep": {"group": "watermark", "label": "워터마크 후보 24개 비교", "kind": "toggle",
                   "help": "켜면 네 가지 프롬프트와 여러 ε·반복 횟수를 비교합니다. 끄면 고른 프롬프트와 현재 설정으로 한 번만 적용합니다."},
    "prompt": {"group": "watermark", "label": "속성 프롬프트", "kind": "select", "options": PROMPTS,
               "help": "어느 얼굴 부위의 속성 방향을 쓸지 정합니다. 자동 모드에서는 네 부위를 모두 비교하고 선택한 부위를 추가로 탐색합니다."},
    "watermark_steps": slider("watermark", "워터마크 기본 반복", 1, 88, 1,
                              "속성 교란을 몇 번 갱신할지 정합니다. 자동 모드에서는 기본 반복과 그 두 배를 탐색합니다. 최대 픽셀 변화는 ε로 제한됩니다.", unit="회"),
    "watermark_step_px": slider("watermark", "워터마크 1회 보폭", 0.25, 4.0, 0.25,
                                "한 번의 갱신에서 픽셀 채널을 움직이는 크기입니다. 값이 커도 최종 변화는 ε를 넘지 않습니다.", unit="/ 255"),
    "jpeg_cutoff": slider("watermark", "자동 선택 JPEG 85 CosSim 상한", -0.2, 0.9, 0.01,
                          "이 점수 이하인 결과 중 SSIM이 가장 높은 후보를 고릅니다. 변환 자체에는 영향을 주지 않는 선택 기준입니다."),
}

TUNED = {
    "rank": 16, "lora_scope": 2, "conv_scope": 2, "reconstruction_steps": 1200, "noise_steps": 800,
    "epsilon": 0.02, "tau": 0.75, "candidate_count": 30, "order_weight": 1.0, "learning_rate": 0.002,
    "conv_learning_rate": 0.0015, "identity_lock": 0.1, "l2_weight": 3.0, "ssim_weight": 0.5,
    "lpips_threshold": 0.03, "cosine_decay": True, "noise_lr": 0.06, "noise_reg": 0.004,
    "noise_ssim": 1.5, "auto_sweep": True, "prompt": "eyebrows", "watermark_steps": 44,
    "watermark_step_px": 1.0, "jpeg_cutoff": 0.20,
}
PRESETS = {
    "tuned": TUNED,
    "paper": {
        **TUNED, "rank": 8, "lora_scope": 0, "conv_scope": 0, "reconstruction_steps": 450,
        "noise_steps": 0, "learning_rate": 0.0003, "l2_weight": 1.0, "ssim_weight": 0.0,
        "lpips_threshold": 0.06, "cosine_decay": False, "auto_sweep": False, "prompt": "nose",
    },
}


def validate(data):
    if not isinstance(data, dict) or data.keys() - FIELDS.keys():
        raise ValueError("설정 형식이 올바르지 않습니다.")
    merged = {**TUNED, **data}
    for key, value in merged.items():
        field = FIELDS[key]
        kind = field["kind"]
        if kind == "range":
            good = type(value) in (int, float) and field["min"] <= value <= field["max"] and isfinite(value)
            if good and isinstance(field["step"], int):
                good = value == int(value)
                if good:
                    merged[key] = int(value)
        elif kind == "select":
            good = isinstance(value, str) and value in field["options"]
        else:
            good = type(value) is bool
        if not good:
            raise ValueError(f"{field['label']} 값이 허용 범위를 벗어났습니다.")
    return merged


def trials(settings):
    epsilon, steps, prompt = (settings[key] for key in ("epsilon", "watermark_steps", "prompt"))
    if not settings["auto_sweep"]:
        return [(prompt, epsilon, steps)]
    return [
        (part, round(epsilon * ratio, 6), steps)
        for part in PROMPTS for ratio in (0.4, 0.7, 1.0, 1.4)
    ] + [
        (prompt, round(epsilon * ratio, 6), count)
        for ratio in (0.5, 0.55, 0.60, 0.65) for count in (steps, 2 * steps)
    ]
