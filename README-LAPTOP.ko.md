# DeepProtect 노트북 실험

노트북에서 신원 혼합, 이미지 재구성, 얼굴 속성 교란을 실행하고 보호 전후를 비교합니다. 설치한 노트북의 브라우저에서 실행하며, 업로드한 사진은 해당 노트북의 작업 폴더에 저장합니다.

## 빠른 시작

1. **64비트 Windows 10/11**, NVIDIA GPU와 CUDA 12.8을 지원하는 드라이버, [Python 3.11 64비트](https://www.python.org/downloads/windows/), [Git for Windows](https://git-scm.com/install/windows)를 준비합니다. PowerShell에서 `python --version` 또는 `py -3.11 --version`으로 Python을, `git --version`과 `nvidia-smi`로 Git과 GPU를 확인합니다.
2. 소스 ZIP을 `C:\DeepProtect`처럼 짧은 경로에 풉니다. 압축을 푼 폴더에 `setup-windows.cmd`가 보여야 합니다. **개인 실험 자료 ZIP**이 있으면 같은 폴더에 풀어 `checkpoints`, `data\banks`, `.runtime\torch`가 코드 폴더 바로 아래에 오도록 합니다.
3. `setup-windows.cmd`를 실행합니다. 인터넷에 연결하면 PyTorch·정렬 도구를 설치하고 SHA-256, CUDA, 체크포인트 전방 연산과 특징 뱅크를 검사합니다. 이미 있는 검증된 가중치는 재사용합니다. 완료될 때까지 창을 열어 둡니다.
4. `start-windows.cmd`를 실행하고 [로컬 테스트 페이지](http://127.0.0.1:8000/)를 엽니다. JPG 또는 PNG를 넣어 정렬을 확인하고 **보호 이미지 만들기**를 누릅니다. 포트가 사용 중이면 PowerShell에서 `.\start-windows.cmd --port 8001`을 실행하고 `http://127.0.0.1:8001/`을 엽니다.

결과 이미지와 지표는 화면에서 내려받을 수 있습니다. `outputs/web-jobs/<작업 ID>/`에 후보 PNG, `metrics.json`, `run.log`가 저장됩니다. 자동 선택 기준을 통과한 후보가 있으면 `protected.png`도 저장합니다. 웹 화면을 쓰면 `data/source.png`를 직접 만들 필요가 없습니다.

## 기존 뱅크 없이 설치

이미 준비한 4,605행 HDF5 파일이 있으면 `data\banks\vggface2_hq_4605.h5`에 둡니다. [모델·데이터 사용 조건과 출처](README.ko.md)를 확인한 뒤, 데이터에서 직접 만들 때는 아래 중 해당하는 절차를 사용합니다.

- 정렬된 VGGFace2-HQ 갤러리가 있으면 `data\VGGFace2-HQ\aligned\<신원>\<이미지>`에 넣고 `setup-windows.cmd`를 실행합니다.
- 기존 실험에 사용한 공개 미러에서 신원당 JPEG 한 장씩 받아 만들려면 압축을 푼 폴더에서 PowerShell을 열고 아래 명령을 실행합니다. 다운로드가 제한되면 나중에 같은 명령을 다시 실행합니다. 기존에 검사된 JPEG는 재사용합니다.

```powershell
.\setup-windows.cmd -DownloadGallery
```

미러의 대표 사진과 논문 저자의 갤러리 목록이 일치하는지는 확인되지 않았습니다. 개인 실험 자료는 같은 사용자의 노트북 실험에 사용하는 파일입니다. 다시 공유할 때는 각 출처의 이용 조건을 확인합니다.

## 첫 실행과 문제 해결

- 처음에는 **공개 구현 기본값** 프리셋으로 실행합니다. 짧은 동작 확인은 재구성 1회, 노이즈 보정 0회, 자동 워터마크 끄기, 워터마크 1회로 설정할 수 있습니다. 이렇게 줄인 설정은 기능 점검에 쓰입니다.
- CUDA 오류나 메모리 부족이 나면 GPU를 사용하는 다른 프로그램을 닫고 다시 실행합니다. 같은 오류가 반복되면 `run.log`의 오류를 확인합니다. 내장 그래픽만 있는 노트북은 이 웹 추론 구성의 실행 대상에서 제외됩니다.
- micromamba가 경로 길이 오류를 내면 소스와 실험 자료를 더 짧은 새 폴더에 다시 풀고 설치합니다. `.venv`와 `.runtime\align`은 설치한 폴더에서 새로 생성합니다.
- SSIM과 LPIPS는 정렬된 사진 대비 영상 유사도이고 ArcFace CosSim은 해당 로컬 모델의 신원 유사도입니다. 실제 얼굴 교체 방어는 별도 평가입니다.

## 검증 범위

RTX 4070 SUPER가 장착된 Windows PC에서 Python 3.11.9, PyTorch 2.7.1+cu128로 확인했습니다. Windows 설치에서 CUDA 역전파, dlib 가져오기, 체크포인트 4개와 LPIPS 전방 연산, 특징 뱅크 4,605행 검사를 통과했습니다. 설치 중 패키지 캐시 권한 문제는 프로젝트 내부의 캐시를 지정해 해결했고, 긴 경로 문제는 짧은 폴더에서 해결했습니다.

ZIP을 푼 복사본에서 웹 서버를 실행해 24개 설정, PNG 업로드와 FFHQ 정렬, CUDA 신원 혼합·재구성·노이즈 보정·속성 교란, 결과 PNG와 지표 JSON 다운로드를 시험했습니다. 이 검사는 재구성 2회, 노이즈 보정 1회, 워터마크 1회를 사용했습니다. 노트북과 RTX 4060 8GB에서 이 배포본의 전체 튜닝 프리셋을 실행하는 작업은 미검증입니다.

[웹 화면과 슬라이더 사용법](web_app/README.md) · [CLI 및 WSL 설치 가이드](README.ko.md)
