# 초기 구현 검증 기록

2026-10-05 로컬 CPU 환경에서 검증. 개발 환경 Python 3.10, PyTorch 2.9.1+cpu,
OpenCV 4.12.0. 실시간 API 엔진 자체가 동작하는지 확인하며 모델 정확도 평가와 분리.

- pytest: **14개 통과** (2.76초).
- 실제 v11 원본 TorchScript SHA256:
  `9ece437b1a86c3454b747a48cbe37ba865d9018377180f1ab2792929a0a982f7`.
- 새 밝은 영상 `teleop_20260927_165104_hq_upright.mp4`의 16초 이후
  5프레임 HTTP 왕복 및 원본 크기 마스크/신뢰도/오버레이 파일 저장 통과.
- 서버 평균 처리 시간 약 **75.1ms**, 클라이언트 최대 수신 이후 결과 age 약 **104.6ms**.
- 해당 짧은 실행에서 pending 교체/지연 폐기/429 모두 0.
- 같은 원본 이미지로 기존 `pinky_lane.inference.predict`와 새 엔진의
  기본 후처리 마스크를 비교: 원본 좌표 nearest resize 후 **픽셀 단위 완전 일치**.
- Gitleaks: 유출 후보 없음.

단기 CPU smoke 결과이지 지속 FPS 보장 또는 실제 카메라/로봇 연결 검증이 아님.
사용 모델은 실험 후보가 아닌 기존 v11 원본의 export 복사본. 기존 학습치를 덮어쓰지 않았음.
테스트 생성 마스크/영상/모델은 outputs/models에만 저장되고 Git에서 제외.
세부 기계 판독 결과는 `outputs/verification.json`, 영상 추론 요약은
`outputs/v11-smoke.jsonl`, PNG는 `outputs/v11-smoke-frames`.

## 아직 별도 확인할 항목

실제 카메라 드라이버와 네트워크 장시간 지연, GPU 처리 속도,
ROS Image/CompressedImage 입력 및 결과 토픽 어댑터, 다른 팀 수신 프로그램 연결,
주행 상태·교차로 경로 정보 기반 해석. 현재 구현이 검증했다고 주장하지 않음.
