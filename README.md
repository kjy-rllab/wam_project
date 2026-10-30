# wam_project

WAM 프로젝트 작업 저장소입니다.

`dreamzero/`는 [DreamZero](https://github.com/dreamzero0/dreamzero) (ab790c1) 원본에 아래 변경을 더한 것입니다.

## Noised context (tmrl)

[tmrl](https://github.com/matthewh6/tmrl)처럼 context에 노이즈를 준 채로 WAM을 학습합니다 (`action_head_cfg.config.context_noise=true`).

- 노이즈 대상: teacher forcing의 깨끗한 영상 latent 흐름, `y`의 첫 프레임 latent, 첫 프레임 CLIP 특징. 언어와 로봇 상태는 깨끗하게 둡니다.
- 샘플마다 수준 하나: `tcont ~ U[0,1)`, `σ = 5·tcont / (1 + 4·tcont)` (모델 스케줄의 shift 5), `x̃ = (1-σ)·x + σ·ε`. CLIP 특징은 RMS 정규화 후 같은 σ로 섞습니다.
- 모델에는 깨끗한 토큰의 timestep으로 수준을 알려 줍니다 (`aug_t = 1000·σ`). `tcont=0`이면 원래 모델과 같습니다.
- 코드: `dreamzero/groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py` (`WANPolicyHead.forward`)

## RoboCasa365 데이터

- 설정: `data=dreamzero/robocasa` (PandaOmron, 3개 뷰 224×224를 2×2로 배치, `frame_seqlen=784`)
- 사전학습 human 데이터 300개 작업 (113GB): HF `nvidia/PhysicalAI-Robotics-Manipulation-Kitchen-Demos`의 `pretrain/{atomic,composite}/<Task>/<date>/lerobot.tar`를 같은 경로에 풀어 씁니다. robocasa 다운로더의 Box 링크는 2026-10-30 기준 404입니다.

## 학습 (Wan2.1-I2V-14B-480P)

```bash
cd dreamzero
# 노드 1개
ROBOCASA_DATA_ROOT='/path/to/robocasa/v1.0/pretrain/*/*/*/lerobot' bash scripts/train/robocasa_training_context_noise.sh
# hangang 여러 노드 (기본 4노드, /scratch에서 제출)
sbatch scripts/train/robocasa_training_context_noise.sbatch
```

- 비교용 기준 모델: `CONTEXT_NOISE=false`
- 환경: DreamZero `pyproject.toml` 기준에 `pyarrow<20`, `pandas<3`, `matplotlib`이 더 필요합니다 (numpy 1.26). aarch64(GH200)에는 torch 2.8.0 cu128 휠이 없어 torch 2.9.0을 씁니다.
