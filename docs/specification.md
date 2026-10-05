# 추론 엔진 명세 v0.1

## 범위와 결정

2026-10-05: 모델 개선과 분리된 새 폴더에 독립 패키지를 구현.
같은 프레임의 전/후처리와 모델은 직접 함수 호출하여 불필요한 IPC를 피함.
ROS 없이도 테스트하고 다른 팀의 수신 프로그램에 연결하기 위해 경계 전송은 HTTP.
외부 API와 로봇 제어는 별도 실패 영역. 이번 단계는 인지 결과만 생성.

입력: `POST /infer`, Content-Type image/jpeg 또는 image/png, Content-Length 필수.
선택 헤더: `X-Frame-Id`, `X-Captured-At`(불투명 문자열), `X-Include-Overlay: 1`.
이미지 순서 BGR은 내부 API, 네트워크는 정상 JPEG/PNG. EXIF/상하반전은 입력 제공자가 정규화.
원본 aspect ratio를 320×240으로 직접 resize하여 기존 학습 방식과 일치.
`GET /health`: 모델 warm-up/입출력 계약 확인 후에만 서버가 ready.

## 응답 JSON

- `schema`: pinky.segmentation.v1
- `frame_id`, `captured_at`, `completed_at`: 프레임 상관관계 및 서버 완료 Unix 시간
- `model_sha256`: 재현성 모델 식별자
- `width/height`: 원본 크기, `model_width/model_height`: 추론 크기
- `coordinate_system`: input_image_pixels_top_left
- `classes`: ID→이름, `ignore_label`: 255
- `mask_png_base64`: 원본 H×W uint8 클래스 ID PNG, nearest resize
- `confidence_png_base64`: 원본 H×W uint16 PNG, scale 65535, 무시 영역 0
- `class_stats`: 원본 픽셀 수와 남은 픽셀의 softmax 평균
- `postprocess_changed_model_pixels`: ROI 제외 후 필터가 변경한 모델 좌표계 픽셀 수
- `valid_pixels`, `status`: 유효 영역 상태, 주행 가능 여부 판단값이 아님
- `timing_ms`: preprocess/inference/postprocess/encode/total, 서버 계산 구간만 포함
- 선택 `overlay_png_base64`: 원본 위 전경만 덧칠

HTTP 400 잘못된 이미지/입력, 401 인증 실패, 404 경로, 408 읽기 시간 초과,
413 잘못된 본문 크기, 415 미지원 형식, 429 추론 중, 500 추론 실패.
서버는 오류 결과를 성공 마스크로 전송하지 않으며 내부 오류 상세는 응답에 노출하지 않음.

## 실시간 계약

클라이언트 최신 대기 프레임 1개. 입력 소비와 추론 통신을 분리하며 교체 횟수를 기록.
파일은 FPS에 맞춰 소비. 처리 후 결과 age를 로컬 monotonic 시계로 측정.
500ms 기본 한계 초과 결과는 저장/전달하지 않음. 이는 센서에서 노출된 시각부터의
절대 지연이 아니라 클라이언트 수신 이후 지연. 네트워크/드라이버 원본 타임스탬프는 별도 통합 필요.
각 클라이언트 버퍼만 최신 우선이고 다중 클라이언트 공정성/로봇별 QoS는 아직 제공하지 않음.
처리 결과 0개이면 CLI 실패: 아무것도 측정하지 않은 성공을 방지.

## 안전성과 확장

이 엔진은 학습 오류를 해결하지 않음. 오탐/좌우 혼동이 있는 모델은 그대로 위험을 가짐.
신뢰도 임계값/성분 삭제는 명시적인 설정과 재평가 없이 주행에 기본 적용하지 않음.
교차로 경로 의도, 좌우 역할 선택, 중심점, 속도/조향 및 비상 정지는 외부 제어 모듈 책임.
결과 없음/오류/지연에는 소비자가 마지막 결과를 무한 재사용하지 않아야 함.
실시간 하드 데드라인, 인터넷 보안 서비스, 로봇 안전 인증은 이번 구현 범위 밖.
OpenCV decode 뒤 픽셀 제한을 검사하므로 악의적 압축 이미지의 decode 메모리 사용까지
강제 제한하지 못함. 신뢰 입력만 허용하고 공개 서비스에는 이미지 헤더 검증/격리 프로세스 필요.

## 검증 기준

전처리 RGB/정규화, 출력 클래스 계약, 원본 좌표 복원/ROI, 마스크 uint8,
신뢰도 uint16, 기본 작은 실제 방지턱 보존, 좌우 클래스 자동 전환 없음,
배경 원본 보존, bounded 최신 프레임, HTTP 왕복·인증·busy·잘못된 입력,
실제 신뢰 TorchScript 학습치로 영상 일부 입력과 결과 저장 확인.
GPU FPS와 실제 카메라/로봇 연결은 장비별 별도 검증 대상.
