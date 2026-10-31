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
# hangang 여러 노드 (기본 16노드·25k 스텝, /scratch에서 제출)
sbatch scripts/train/robocasa_training_context_noise.sbatch
```

- hangang 작업은 24시간 제한이 있어서, 작업이 시작할 때 다음 작업을 하나 예약합니다 (최대 `CHAIN_JOBS=4`개). 다음 작업은 마지막 체크포인트에서 이어서 학습하고, 학습이 끝나면 남은 예약은 취소됩니다.
- 한 번 시작한 학습은 노드 수를 바꾸지 않습니다 (ZeRO 옵티마이저 상태가 GPU 수대로 나뉘어 저장됨).
- 비교용 기준 모델: `CONTEXT_NOISE=false`
- 환경: DreamZero `pyproject.toml` 기준에 `pyarrow<20`, `pandas<3`, `matplotlib`이 더 필요합니다 (numpy 1.26). aarch64(GH200)에는 torch 2.8.0 cu128 휠이 없어 torch 2.9.0을 씁니다.

## 추론에서 context 노이즈 수준 주기

```python
result_batch, video_pred = policy.lazy_joint_forward_causal(batch, tcont_context=0.3)  # policy: GrootSimPolicy
```

- `tcont_context`: 0~1 실수, 또는 샘플마다 다른 값이면 `[B]` 텐서. `None`(기본)이면 원래 추론과 같습니다.
- 학습과 같은 식으로 KV cache에 넣는 관측 latent를 노이즈하고, 그 토큰의 timestep을 `1000·σ`로 줍니다.
- 첫 프레임 `y`와 CLIP 특징은 KV cache를 새로 시작하는 호출의 수준으로 한 번 노이즈합니다. 학습처럼 에피소드마다 수준 하나를 쓰는 게 기본이고, 호출마다 바꾸면 새로 들어오는 관측 프레임에만 적용됩니다.
- 노이즈는 KV cache를 시작할 때 seed를 고정한 generator에서 뽑습니다. 그래서 같은 입력이면 결과가 같고, GPU 2개로 CFG를 나눠 돌릴 때도 두 GPU가 같은 노이즈를 씁니다.
- 코드: `WANPolicyHead.lazy_joint_video_action`. DROID용 서버(`socket_test_optimized_AR.py` 등)는 아직 이 값을 받지 않습니다.
