# DeepProtect 실행 가이드

기준은 [BACKAI/DeepProtect_CVPR26](https://github.com/BACKAI/DeepProtect_CVPR26/tree/dc40084baf36d3f73e6a31e7fba5ad383d5f7080)의 `dc40084baf36d3f73e6a31e7fba5ad383d5f7080`입니다. WSL2 Ubuntu 22.04, Python 3.10 및 RTX 4060 8GB를 초기 대상으로 삼았습니다. Windows에서 확인한 환경과 범위는 [노트북 가이드](README-LAPTOP.ko.md#검증-범위)에 기록합니다.

## 원본 코드 흐름

| 위치 | 역할 |
|---|---|
| `README.md`, `requirements.txt` | 원본 환경, 모델 파일 형식, 갤러리 및 추론 CLI |
| `scripts/align_faces.py` | dlib 랜드마크를 사용해 얼굴을 FFHQ 형식으로 정렬 |
| `scripts/build_feature_bank.py`, `deep_protect/feature_bank.py` | 신원별 대표 이미지 선택, 세 모델로 특징 추출, HDF5/NPZ 입출력 |
| `scripts/inference.py`, `deep_protect/pipeline.py` | 단일 이미지 추론, 중간 산출물과 최종 이미지 저장 |
| `deep_protect/blending.py` | FaRL 유사도로 후보를 검색하고 W+ 스타일 인덱스 2–6을 치환 |
| `deep_protect/optimization.py`, `deep_protect/models/lora.py` | StyleGAN2-ADA b16/b32 affine에 LoRA를 적용해 L2·LPIPS·신원 고정 손실을 최적화 |
| `deep_protect/watermark.py`, `deep_protect/oal_da.py` | 프롬프트 유사도 상·하위 집합에서 신원 공간 방향을 구하고 픽셀 교란을 투영 |
| `networks.py`, `torch_utils/`, `models/e4e/` | 공식 체크포인트 역직렬화와 모델 실행용 런타임 |
| `scripts/train.py` | 같은 파이프라인을 이미지 디렉터리에 반복 적용 |

## 1. WSL2 및 CUDA 확인

WSL이 없는 Windows에서는 관리자 PowerShell에서 `wsl --install -d Ubuntu-22.04`를 실행하고 재부팅 후 Linux 사용자 초기화를 완료합니다. `wsl -l -v`에서 Ubuntu-22.04의 VERSION이 2인지 확인합니다. [Microsoft WSL 설치 문서](https://learn.microsoft.com/windows/wsl/install)

Ubuntu에서 프로젝트 디렉터리를 열고 아래 명령을 실행합니다. Linux 파일시스템에 프로젝트를 복사하면 파일 입출력 지연을 줄일 수 있습니다. NVIDIA 드라이버는 Windows에 설치하며, WSL의 CUDA 가속은 호스트 드라이버를 사용합니다. [NVIDIA CUDA on WSL 가이드](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

```bash
set -o pipefail
sudo apt update
sudo apt install -y python3.10 python3.10-venv git curl
nvidia-smi
mkdir -p outputs/logs
bash scripts/setup_wsl.sh 2>&1 | tee outputs/logs/setup.log
source .venv/bin/activate
python -m pip freeze > outputs/logs/pip-freeze.txt
```

`requirements-wsl.txt`는 Python 3.10용 직접 의존성을 고정합니다. PyTorch 2.7.1과 torchvision 0.22.1은 CUDA 12.8 휠을 사용합니다. 커스텀 StyleGAN 연산자 빌드에 실패하면 원본 PyTorch 참조 구현으로 전환됩니다. CUDA 텐서는 이 경우에도 GPU에서 계산됩니다. [PyTorch 휠 목록](https://pytorch.org/get-started/previous-versions/)

## 2. 체크포인트와 데이터 준비

| 파일 | 다운로드 출처 및 호환성 | 사용 조건 |
|---|---|---|
| `ffhq.pkl` | [NVIDIA FFHQ 1024 StyleGAN2-ADA](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl). 공식 PyTorch pickle의 `G_ema`. | [NVIDIA Source Code License-NC](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/LICENSE.txt) |
| `e4e_ffhq_encode.pt` | [e4e FFHQ Inversion](https://github.com/omertov/encoder4editing#pretrained-models). `opts`, encoder/decoder state dict, `latent_avg`를 포함. | [e4e 코드 MIT](https://github.com/omertov/encoder4editing/blob/master/LICENSE). 별도 체크포인트 사용권은 확인이 필요합니다. |
| `model_ir_se50.pth` | [pSp의 IR-SE50 링크](https://github.com/eladrich/pixel2style2pixel#pretrained-models)에서 제공하는 TreB1eN ArcFace. DeepProtect 로더에서 검증한 파일입니다. 논문 저자가 사용한 파일과 동일한 해시인지는 확인되지 않았습니다. | [TreB1eN 코드 MIT](https://github.com/TreB1eN/InsightFace_Pytorch/blob/master/LICENSE). 별도 체크포인트 사용권은 확인이 필요합니다. |
| `FaRL-Base-Patch16-LAIONFace20M-ep64.pth` | [FacePerceiver 공식 ep64 릴리스](https://github.com/FacePerceiver/FaRL/releases/tag/pretrained_weights). CLIP ViT-B/16 이미지·텍스트 인코더 가중치. | [FaRL 저장소 MIT](https://github.com/FacePerceiver/FaRL/blob/main/LICENSE). 가중치와 학습 데이터의 별도 사용 조건은 배포처에서 확인합니다. |
| AlexNet LPIPS | [LPIPS v0.1](https://github.com/richzhang/PerceptualSimilarity)의 AlexNet 보정 가중치와 [torchvision AlexNet](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.alexnet.html). torchvision이 AlexNet 가중치를 최초 실행 때 내려받습니다. | [LPIPS 코드 BSD-2-Clause](https://github.com/richzhang/PerceptualSimilarity/blob/master/LICENSE). AlexNet 및 학습 데이터 조건은 각각 배포처에서 확인합니다. |
| `shape_predictor_68_face_landmarks.dat` | 비정렬 이미지에만 필요. [dlib 공식 다운로드](https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2). | [dlib 예제의 iBUG 300-W 비상업 조건 안내](https://dlib.net/face_landmark_detection.py.html) |

체크포인트와 레거시 NPZ는 신뢰하는 출처의 파일만 사용합니다. 다운로드 스크립트의 SHA-256은 이번 실행에서 내려받은 파일에 고정되어 있습니다. LPIPS를 포함한 GPU 전방 연산의 출력 크기와 유한값은 검사 스크립트로 확인합니다.

```bash
bash scripts/download_models.sh checkpoints
STYLE=checkpoints/ffhq.pkl
E4E=checkpoints/e4e_ffhq_encode.pt
ARC=checkpoints/model_ir_se50.pth
FARL=checkpoints/FaRL-Base-Patch16-LAIONFace20M-ep64.pth
BANK=data/banks/vggface2_hq_4605.h5
models=(--stylegan "$STYLE" --e4e "$E4E" --arcface "$ARC" --farl "$FARL")
python -u scripts/validate_inputs.py "${models[@]}" --lpips --trusted-inputs --device cuda \
  2>&1 | tee outputs/logs/models.log
```

[공식 VGGFace2-HQ](https://github.com/NNNNAI/VGGFace2-HQ)에서 라이선스와 접근 조건을 확인하고 `data/VGGFace2-HQ/aligned/<identity>/<image>` 구조로 준비합니다. 배포 설명에는 학술 얼굴 편집 용도가 명시되어 있습니다. 직접 배포 링크에 접근할 수 없는 경우, 공개 아카이브 미러의 훈련 분할에서 신원당 첫 JPEG만 가져오는 아래 명령을 사용할 수 있습니다. 스크립트는 고정된 아카이브 인덱스와 이미지 CRC를 확인합니다. 미러가 DeepProtect 저자의 원본 갤러리와 같은 대표 이미지 목록을 사용했는지는 확인되지 않았습니다. [VGGFace2-HQ 아카이브 미러](https://huggingface.co/datasets/RichardErkhov/VGGFace2-HQ/tree/6771c0319f29a7b77e4f71ebfb09f7b40066e2cc/original) 원본 VGGFace2 이미지와 HQ 변환본은 출처와 조건을 확인한 뒤 사용합니다. 논문의 정확한 대표 이미지 목록과 기존 갤러리 파일은 기준 체크아웃에서 확인되지 않았습니다. [DeepProtect 갤러리 자료 이슈](https://github.com/BACKAI/DeepProtect_CVPR26/issues/2)

추론 원본은 FFHQ 형식으로 정렬된 1024×1024 RGB 얼굴을 `data/source.png`에 준비합니다. 논문 평가를 비교하려면 해당 평가 데이터셋(예: [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans#preparing-datasets)), 원본·대상 쌍, 딥페이크 모델 및 평가 조건이 필요합니다. 비정렬 이미지에는 아래 선택 절차를 사용합니다.

```bash
sudo apt install -y build-essential cmake python3.10-dev bzip2
python -m pip install 'dlib==19.24.9'
curl -fL https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 \
  -o checkpoints/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -dk checkpoints/shape_predictor_68_face_landmarks.dat.bz2
python scripts/align_faces.py --input-dir data/raw --output-dir data/aligned \
  --shape-predictor checkpoints/shape_predictor_68_face_landmarks.dat \
  2>&1 | tee outputs/logs/align.log
```

공개 미러를 사용하는 경우 이미지 준비 단계에서 실행합니다. 기본 4,605개를 선택하고, 다시 실행하면 CRC가 일치하는 로컬 파일은 재사용합니다.

```bash
python -u scripts/prepare_vggface2hq.py --limit 4605 \
  2>&1 | tee outputs/logs/gallery-prepare.log
```

## 3. 신원 특징 데이터베이스 생성 및 검증

첫 번째 하위 폴더가 신원 ID입니다. 스크립트는 신원별 경로 정렬 순서의 첫 이미지 하나를 사용하고 최대 4,605개 신원을 선택합니다. 검증은 W+ `[N,18,512]`, FaRL 및 ArcFace `[N,512]`, 유한값, 비어 있는 임베딩, 중복 경로 및 행 수를 확인합니다. 기본 속성 후보 수 30에는 서로 겹치지 않는 상·하위 각 30개를 위해 최소 60개 신원이 필요합니다.

```bash
python -u scripts/build_feature_bank.py \
  --images data/VGGFace2-HQ/aligned --output "$BANK" \
  --e4e "$E4E" --arcface "$ARC" --farl "$FARL" \
  --max-identities 4605 --batch-size 1 --device cuda \
  2>&1 | tee outputs/logs/bank-build.log
python scripts/validate_inputs.py --feature-bank "$BANK" --expected-identities 4605 \
  2>&1 | tee outputs/logs/bank-validate.log
```

## 4. 단계별 점검 및 최종 추론

브라우저에서 같은 전체 과정과 사진별 워터마크 후보 비교를 실행하려면 [로컬 웹앱](web_app/README.md)을 사용합니다.

처음에는 `smoke_components.py`로 실제 가중치의 계산 경로를 확인할 수 있습니다. 이 스크립트의 60행 특징은 임시 테스트 입력이며 실제 갤러리 추출·추론은 아래 명령으로 수행합니다.

```bash
python -u scripts/smoke_components.py "${models[@]}" --device cuda \
  2>&1 | tee outputs/logs/components.log

common=("${models[@]}" --feature-bank "$BANK" --image data/source.png --prompt nose --device cuda --profile)
python -u scripts/inference.py "${common[@]}" --output-dir outputs/blend --stop-after blend \
  2>&1 | tee outputs/logs/blend.log
python -u scripts/inference.py "${common[@]}" --output-dir outputs/reconstruction --stop-after reconstruction \
  2>&1 | tee outputs/logs/reconstruction.log
python -u scripts/inference.py "${common[@]}" --output-dir outputs/attribute --mode attribute \
  2>&1 | tee outputs/logs/attribute.log
python -u scripts/inference.py "${common[@]}" --output-dir outputs/combined \
  2>&1 | tee outputs/logs/combined.log
test -s outputs/combined/protected.png
python scripts/measure_images.py outputs/combined/source.png outputs/combined/protected.png \
  --output outputs/combined/metrics.json 2>&1 | tee outputs/logs/metrics.log
```

혼합에서는 `latent_original.pt`, `latent_blended.pt`, `identity_blended.png`가 남습니다. 재구성 단계에서 `generator_optimized.png`와 `lora_state.pt`가 추가됩니다. 속성 단독 모드는 정렬된 원본에 워터마크를 적용합니다. 전체 모드는 재구성 결과에 교란을 적용해 `protected.png`를 생성합니다. 출력 디렉터리는 실행마다 새 경로를 사용하면 이전 결과와 혼동하지 않습니다.

`--profile`은 모델 로드·혼합·재구성·속성 연산의 CUDA 동기화 시간과 할당/예약 메모리 최고치를 로그와 성공 시 `metadata.json`에 기록합니다. 수치는 PyTorch 할당자 기준이며 CUDA 컨텍스트나 타 프로세스 메모리는 `nvidia-smi`에서 별도로 확인합니다. SSIM은 RGB 채널 평균 및 Gaussian 창, PSNR은 8비트 RGB, LPIPS는 AlexNet v0.1을 사용합니다. 이 지표는 원본 대비 영상 유사도를 측정합니다.

### 기존 Windows 환경에서 재실행

프로젝트 루트에서 PowerShell을 엽니다. 같은 작업 폴더에 만들어 둔 가상환경과 체크포인트를 사용합니다. 먼저 `data\source.png`에 보호할 사진을 준비합니다. 파이프라인은 입력을 RGB로 읽고 모델용 1024×1024 입력을 만듭니다. 얼굴 위치와 각도 정렬은 입력 단계에서 준비합니다.

```powershell
$py = '..\..\work\testenv\Scripts\python.exe'
$encoders = @(
    '--e4e', '..\..\work\checkpoints\e4e.pt',
    '--arcface', '..\..\work\checkpoints\arcface.pth',
    '--farl', '..\..\work\checkpoints\FaRL.pth'
)
$models = @('--stylegan', '..\..\work\checkpoints\ffhq.pkl') + $encoders
$bank = 'data\banks\vggface2_hq_4605.h5'
if (-not (Test-Path -LiteralPath $bank -PathType Leaf)) {
    & $py -u scripts/prepare_vggface2hq.py --limit 4605 --workers 16
    if ($LASTEXITCODE -ne 0) { throw '갤러리 준비 실패' }
    & $py -u scripts/build_feature_bank.py @encoders --images 'data\VGGFace2-HQ\aligned' --output $bank --max-identities 4605 --batch-size 8 --device cuda
    if ($LASTEXITCODE -ne 0) { throw '특징 뱅크 생성 실패' }
}
& $py scripts/validate_inputs.py --feature-bank $bank --expected-identities 4605
if ($LASTEXITCODE -ne 0) { throw '특징 뱅크 검증 실패' }

$run = 'outputs\combined-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
New-Item -ItemType Directory -Path $run | Out-Null
& $py -u scripts/inference.py @models --image 'data\source.png' --feature-bank $bank --output-dir $run --mode combined --prompt nose --device cuda --profile 2>&1 |
    Tee-Object -FilePath "$run\inference.log"
if ($LASTEXITCODE -ne 0 -or -not (Test-Path "$run\protected.png")) { throw '추론 실패' }
& $py scripts/measure_images.py "$run\source.png" "$run\protected.png" --output "$run\metrics.json"
```

### 원본 외형을 우선하는 전체 모드 튜닝

같은 정렬 사진에서 시험한 설정입니다. 신원 혼합, 재구성, 속성 교란을 모두 실행합니다. 1단계는 혼합 잠재벡터에 맞춰 합성 가중치를 학습하고, 2단계는 저장된 가중치를 불러와 노이즈 맵을 추가 학습한 뒤 워터마크를 적용합니다.

프로젝트 루트의 PowerShell에서 실행합니다.

```powershell
$py = '..\..\work\testenv\Scripts\python.exe'
$common = @(
    '--stylegan', '..\..\work\checkpoints\ffhq.pkl',
    '--arcface', '..\..\work\checkpoints\arcface.pth',
    '--farl', '..\..\work\checkpoints\FaRL.pth',
    '--feature-bank', 'data\banks\vggface2_hq_4605.h5'
)
$blocks = @('b4','b8','b16','b32','b64','b128','b256','b512','b1024')
$convBlocks = $blocks[1..8]
$base = 'outputs\tuned-base-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
$final = "$base-refined"

& $py -u scripts/inference.py @common --e4e '..\..\work\checkpoints\e4e.pt' --image '..\..\work\source-aligned.png' --output-dir $base --mode combined --prompt nose --device cuda --profile --lora-blocks @blocks --lora-rank 16 --learning-rate 0.002 --trainable-conv-blocks @convBlocks --conv-learning-rate 0.0015 --cosine-decay --reconstruction-l2-weight 3 --reconstruction-ssim-weight 0.5 --lpips-threshold 0.03 --max-optimization-steps 1200
if ($LASTEXITCODE -ne 0) { throw '1단계 재구성 실패' }

& $py -u scripts/refine_combined_noise.py $base $final @common --device cuda --steps 800 --noise-learning-rate 0.06 --noise-regularization-weight 0.004 --ssim-weight 1.5 --prompt eyebrows --watermark-epsilon 0.014 --watermark-steps 44
if ($LASTEXITCODE -ne 0) { throw '2단계 보정 실패' }
& $py scripts/measure_images.py "$final\source.png" "$final\protected.png" --output "$final\metrics.json"
```

정렬된 사진을 바꾸려면 `--image`에 해당 경로를 지정합니다. 실험에서 사용한 설정은 [웹 화면과 슬라이더 사용법](web_app/README.md)에 있습니다.

### VRAM이 부족한 경우

1. `nvidia-smi`에서 여유 메모리를 점검하고 다른 GPU 작업을 종료합니다. 이미지는 프로세스당 한 장씩 추론하고 갤러리는 `--batch-size 1`을 사용합니다.
2. 단계별 명령을 각기 새 프로세스로 실행해 OOM 지점과 최고치를 식별합니다. StyleGAN의 기본 사본과 e4e 디코더는 CPU에 유지됩니다.
3. CUDA 할당자에서 예약량이 할당량보다 크게 나타날 때는 WSL에서 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`로 재시도할 수 있습니다. 해당 환경에서 옵션이 지원되는지 로그를 확인합니다. [PyTorch CUDA 메모리 관리](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
4. 그래도 실패하면 더 큰 VRAM 또는 `--device cpu`로 동일한 설정을 시도할 수 있습니다. 낮은 해상도, 적은 최적화 횟수, `--no-generator-optimization`을 적용한 결과는 각각 별도 진단 실행으로 기록합니다.

