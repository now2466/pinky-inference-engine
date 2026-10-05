# Pinky 실시간 추론 엔진

모델 학습 저장소와 독립된 **이미지 입력 → 전처리 → 추론 → 후처리 → 결과 전송** 엔진입니다.
모델 교체는 TorchScript 파일과 설정 교체로 수행합니다. 현재 LaneUNet 4/5클래스 출력을 지원하며
기존 모델을 자동으로 대체하거나 학습하거나 로봇에 주행 명령을 보내지 않습니다.

## 구조

```text
configs/default.json       입력 크기, 클래스, ROI, 후처리, 자원 제한
src/pinky_engine/
  config.py                명시적인 모델 계약
  preprocess.py            BGR uint8 → RGB float32 /255 → NCHW
  backend.py               TorchScript 로딩, warm-up, CPU/CUDA, 출력 검사
  postprocess.py           ROI/신뢰도/선택적 작은 연결성분 필터
  engine.py                파이프라인, 원본 좌표 복원, PNG 결과 직렬화
  server.py                HTTP 이미지 수신 / 결과 응답
  stream.py                최신 프레임 1개 버퍼
  client.py                카메라/영상 수신, API 전송, JSONL/이미지 저장
tests/                     계약 및 실제 HTTP 왕복 테스트
docs/specification.md      결과 형식, 지연, 안전성, 확장 규격
models/                    로컬 신뢰 모델만 배치 (Git 제외)
outputs/                   실행 결과 (Git 제외)
```

## 설치

Python 3.10 이상. CPU도 지원하며 GPU에서는 CUDA 지원 PyTorch 환경을 사용합니다.

```bash
git clone https://github.com/now2466/pinky-inference-engine.git
cd pinky-inference-engine
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
```

기존 환경으로도 실행할 수 있습니다. 학습 프로젝트 코드 import는 필요 없습니다.
TorchScript만 지원하며 학습 checkpoint `best.pt`는 학습 저장소에서 먼저 export해야 합니다.
모델 출력은 `[1, C, 240, 320]` logits, `C=4|5`이며 클래스 순서는 설정과 같아야 합니다.
파일명만으로 클래스 의미를 검증할 수 없으므로 신뢰할 수 있는 export의 클래스 순서를 확인하세요.

## 서버

```bash
.venv/bin/pinky-engine --model models/v11-original.torchscript.pt --config configs/default.json
```

또는 패키지가 설치된 Python 환경 사용:

```bash
python -m pinky_engine.server \
  --model models/v11-original.torchscript.pt --config configs/default.json
```

모델 클래스 수가 설정과 다르면 시작하지 않습니다. 예전 4클래스 파일은 설정에서
`speed_bump`를 제거해야 하며 방지턱을 추론할 수 없습니다.
**모델 파일은 저장소에 포함하지 않습니다.** 팀에서 전달받은 신뢰할 수 있는 TorchScript 파일을
`models/v11-original.torchscript.pt`에 복사하거나 `--model`에 해당 파일 경로를 지정하세요.
학습·export 코드: [pinky-lane-segmentation](https://github.com/now2466/pinky-lane-segmentation).

```bash
curl http://127.0.0.1:8770/health
curl --fail -H 'Content-Type: image/jpeg' -H 'X-Frame-Id: frame001' \
  --data-binary @sample.jpg http://127.0.0.1:8770/infer
```

## 실시간 카메라 / 영상 테스트

```bash
.venv/bin/pinky-stream --source 0 --output outputs/camera-session.jsonl \
  --artifacts outputs/camera-frames

.venv/bin/pinky-stream --source /path/to/video.mp4 --output outputs/video-session.jsonl \
  --artifacts outputs/video-frames --max-frames 30
```

영상은 원본 FPS로 재생하며 입력이 처리보다 빠르면 오래된 대기 프레임을 버립니다.
녹화 영상의 특정 구간부터 테스트하려면 `--start-seconds 16`처럼 지정합니다.
모든 프레임을 오프라인 추론하는 도구가 아닙니다. 클라이언트가 받은 뒤 500ms가 지난 결과는
기본적으로 전달하지 않습니다. CPU 테스트가 느리면 `--max-age-ms 2000` 등으로 조절하세요.
시계가 다른 호스트끼리 `captured_at`을 빼서 지연을 계산하지 않습니다.

API는 마스크/신뢰도 PNG를 포함합니다. 클라이언트 JSONL은 이진 이미지 대신 요약과 저장 경로를 기록합니다.
`--artifacts`가 없으면 클라이언트는 이미지 페이로드를 저장하지 않습니다.
출력 파일/이미지 디렉터리는 새 경로를 사용해야 합니다. 기존 결과를 덮어쓰지 않습니다.

## 결과와 후처리

클래스: 0 배경, 1 왼쪽 차선, 2 오른쪽 차선, 3 횡단보도, 4 과속방지턱, 255 무시/불확실.
마스크는 색상 그림이 아니라 클래스 ID PNG입니다. 신뢰도 PNG는 uint16, `/65535`로 복원합니다.
좌표는 **입력 원본 이미지 좌표**입니다. `overlay`는 배경을 원본 그대로 보존합니다.
신뢰도는 softmax 값이며 정확도가 보장되거나 보정된 확률이 아닙니다.

기본 전처리는 학습과 동일합니다. CLAHE/샤프닝/회전은 자동 적용하지 않습니다.
기본 후처리는 모델 크기 기준 상단 110행을 무시합니다. 신뢰도/성분 필터는 꺼져 있습니다.
작은 실제 방지턱이 삭제될 수 있어 면적 필터를 기본 활성화하지 않았습니다.
차선의 화면 좌우 위치만으로 클래스를 바꾸거나, 마스크를 팽창시켜 채우지 않습니다.
필터로 제거한 픽셀은 배경이 아니라 255이며 해당 신뢰도는 0입니다.

## 네트워크 보안 및 제한

기본 loopback 바인딩입니다. 외부 바인딩은 `PINKY_ENGINE_TOKEN` 환경변수를 설정해야 합니다.
클라이언트도 같은 환경변수를 설정합니다. 토큰은 코드/설정 파일에 저장하지 않습니다.
HTTP 자체는 암호화되지 않으므로 신뢰 LAN 또는 TLS 역방향 프록시에서만 사용하세요.
인터넷에 이 개발 서버를 직접 공개하지 마세요. 신뢰할 수 없는 모델을 로드하지 마세요.
본문 8MiB, 입력 약 2MP, 동시 HTTP 연결 8개, 동시 추론 1개 제한입니다.
과부하 추론 요청은 429로 거부하며 FIFO 대기열에 적재하지 않습니다.
OpenCV 카메라 드라이버 내부 버퍼나 원격 스트림 지연까지 제거하는 것은 아닙니다.

ROS 어댑터, lane 중심점 계산, TensorRT/ONNX, 시간축 역할 보정은 다음 확장 지점입니다.
현재 HTTP 이미지 엔진을 ROS 노드라고 주장하거나 실제 로봇 연동이 검증됐다고 보지 않습니다.

초기 실행 및 테스트 기록은 [검증 기록](docs/verification.md)에 있습니다.

## 커밋 전 보안 검사

[Gitleaks](https://github.com/gitleaks/gitleaks)를 설치하고 저장소에서 다음을 실행하면
커밋할 때 staged 변경을 자동 검사합니다. 검사 도구가 없거나 유출 후보가 발견되면 커밋을 중단합니다.

```bash
git config core.hooksPath .githooks
gitleaks git --pre-commit --staged --redact --no-banner
```

모델·영상·출력·인증 파일은 올리지 않습니다. `.gitignore`는 이미 추적 중인 파일에는 적용되지
않으므로 `git diff --cached --name-only`로 업로드 목록도 확인하세요.

추론 실행은 `eval()`과 `inference_mode()`를 함께 사용합니다.
[PyTorch 공식 문서](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html)
