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
- natStr는 사용자당 작업 5개까지이고, 예약돼 기다리는 작업도 셉니다. 연구실이 계정 하나를 같이 쓰므로, 학습 중에는 5개 중 2개(실행 1개, 예약 1개)를 씁니다. 자리가 없어 예약이 안 되면 로그에 `Queued next job: none`이 남고, 그 작업으로 학습이 멈춥니다.
- 학습이 시작 후 15분 안에 멈추면 같은 노드에서 다시 시작합니다 (최대 3번). 모델을 GPU로 옮기다 GPU 메모리 부족으로 실패한 작업이 2개 있었습니다. GH200은 CPU 메모리가 차면 GPU 메모리를 CPU 메모리로 쓰는데, 그때 약 54GB가 그렇게 쓰이고 있었습니다.
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

## RoboCasa 시뮬레이션에서 추론

`dreamzero/eval_utils/serve_robocasa.py`가 RoboCasa 클라이언트(`robocasa_workspace/server_inference`)의 ZeroMQ(RLDX) 규약으로 DreamZero를 서빙합니다. 클라이언트는 고치지 않고 `tcp://` 주소로 연결합니다.

```bash
# GPU 서버 (dreamzero 폴더, pyzmq 필요)
torchrun --nproc_per_node=1 eval_utils/serve_robocasa.py --model-path /path/to/checkpoint-7000 --port 5555
# 로컬 (robocasa_workspace, robocasa 환경)
python -m server_inference.run_robocasa --server tcp://127.0.0.1:5555 \
  --task CoffeeSetupMug --split pretrain --replan-steps 24 --max-steps 600 --clip-actions
```

- 학습처럼 행동 24 step 한 chunk가 그 구간의 프레임 8개(3 step 간격)와 짝입니다. 그래서 클라이언트는 chunk를 끝까지 실행해야 합니다 (`--replan-steps 24`).
- 매 호출에서 서버는 방금 실행한 chunk의 프레임 8개와 그 앞 프레임 1개를 받습니다 (`video.delta_indices = -24, -21, …, 0`). 에피소드 첫 호출은 현재 프레임 하나로 KV cache를 새로 시작합니다.
- KV cache는 첫 프레임 + chunk 4개(96 step)까지 쌓고, 다음 호출에서 현재 프레임으로 다시 시작합니다 (DreamZero 원래 동작).
- `--clip-actions`: 학습 데이터의 팔 위치·base 행동은 controller 범위 끝(±1)까지 쓰입니다. 그래서 모델 출력이 범위를 조금 넘을 수 있고, 넘는 값은 범위 끝으로 자릅니다.
- gripper와 control mode는 데이터에 -1/1로 저장돼 있습니다. 클라이언트는 Gym 형식(0/1, 0.5 이상이면 닫기)을 받으므로 서버가 `(a+1)/2`로 바꿔 보냅니다.
- `--tcont-context 0.3`처럼 context 노이즈 수준을 줍니다. 요청의 `options["tcont_context"]`가 있으면 그 호출에는 그 값을 씁니다.
- `--nproc_per_node=2`면 CFG의 두 branch를 GPU 2개에 나눠 돌립니다.
- 서버 하나는 클라이언트 세션 하나만 받습니다. 다른 세션이 에피소드를 시작하면 이전 세션의 다음 요청은 오류로 끝납니다.
- 학습 환경에 더해 `pyzmq`, `tianshou==0.5.1`, `gymnasium`이 필요합니다 (`sim_policy.py`).
- hangang: `sbatch scripts/inference/robocasa_serve.sbatch <checkpoint> [--tcont-context 0.3]`로 GPU 2개, 4시간 작업을 띄웁니다. 노드를 다른 작업과 나눠 쓰므로 포트는 작업마다 다르고, 로컬에서 돌릴 터널 명령이 작업 로그의 `Local tunnel:` 줄에 찍힙니다. 학습 중인 checkout을 건드리지 않으려면 다른 checkout에서 제출합니다. 스크립트가 `PYTHONPATH`를 제출한 checkout으로 잡습니다.
- 속도 (hangang GH200 2개): 첫 호출은 compile 때문에 2분쯤 걸리고, 이후에는 호출당 왕복 약 2.2초입니다.

## 검증: context 노이즈 수준별 비교

체크포인트 하나를 context 노이즈 수준(`tcont`)마다 평가합니다. 모든 수준이 같은 샘플과 같은 노이즈, 같은 에피소드를 씁니다. 아래에서 task는 RoboCasa 작업(예: CloseMicrowave)입니다.

### 데이터 loss

```bash
# hangang, 학습 중이 아닌 checkout의 dreamzero 폴더 (GPU 4개)
sbatch scripts/inference/robocasa_val_loss.sbatch <checkpoint> --resume-steps 0 9000 18000
```

- `eval_utils/robocasa_val_loss.py`는 run 설정으로 학습 때의 shard 순서를 다시 만들어 샘플을 세 묶음으로 고르고, 각 샘플의 dynamics loss와 action loss를 수준마다 잽니다.
  - trained: 학습한 샘플. 학습 작업마다 data worker가 처음 방문한 shard는 shuffle까지 다시 만들 수 있어서, 그 shard에서 실제로 학습한 샘플만 씁니다.
  - heldout: 학습한 task에서 어느 worker도 방문하지 않은 shard
  - unseen: shard를 하나도 방문하지 않은 task
- `--resume-steps`: run의 작업들이 시작한 global step. 이어서 학습할 때마다 데이터 seed가 `42 + 시작 step`으로 바뀌므로 필요합니다.
- 기본값: 묶음마다 task 40개 (trained와 heldout은 같은 task, atomic과 composite 반씩), task마다 샘플 4개, 수준 `0, 1/32, 1/16, 1/8, 1/4, 1/2, 1`. batch가 bf16으로 바뀌므로 수준은 2의 거듭제곱으로 둡니다.
- 결과: `<run>/val_loss/<checkpoint>/summary.json` (묶음·수준별 평균과 표준오차), `records.jsonl` (샘플별). 로그 끝에 두 loss의 표가 찍힙니다.
- 데이터셋 순서가 파일 시스템의 glob 순서라 학습한 클러스터에서 돌립니다.
- rank들이 모델을 하나씩 차례로 불러옵니다. 불러오는 동안 rank 하나가 CPU 메모리를 약 120GB 쓰는데, 4개가 한꺼번에 불러오니 CPU 메모리(약 480GB)가 차서 GPU 메모리로 넘쳤고, 그 GPU가 메모리 부족으로 멈췄습니다.
- 시간 (GH200 4개): 준비에 약 13분 (모델을 차례로 불러오는 데 10분), 그 뒤 GPU마다 shard 하나(샘플 1개 × 수준 3개)에 약 20초. checkpoint-21000으로 shard 12개를 시험하는 데 15분 걸렸고, 기본값(shard 약 100개)이면 1시간쯤으로 예상합니다.
- checkpoint-21000에서 다시 만든 shard 순서가 학습 로그의 shard 대기 기록(작업 3개, 1,792개)과 모두 맞았습니다.

### 시뮬레이션 성공률

```bash
# hangang: 수준마다 서버 하나 (GPU 4개 = 수준 2개 × CFG용 GPU 2개)
SERVE_LEVELS="0 0.125" sbatch --gres=gpu:4 --time=08:00:00 scripts/inference/robocasa_serve.sbatch <checkpoint>
# 로컬: 작업 로그의 `Local tunnel:` 명령으로 연결한 뒤, dreamzero 폴더에서 robocasa 환경으로 (작업 로그의 `Local eval:` 줄)
python eval_utils/robocasa_sim_eval.py --servers 0=tcp://127.0.0.1:5555 0.125=tcp://127.0.0.1:5556 --output-dir <dir>
```

- task: 학습에서 가장 많이 뽑힌 atomic task 8개(seen)와, 뽑힌 적은 없지만 반대 task는 학습한 atomic task 8개(unseen, 예: CloseMicrowave ↔ OpenMicrowave). `--seen`, `--unseen`으로 바꿉니다.
- 에피소드 e는 seed e로 장면을 만들어서 모든 수준이 같은 장면을 씁니다. 기본 10 에피소드, `--split pretrain`, 최대 step은 RoboCasa dataset registry의 task horizon(300~750)입니다.
- 서버는 앞 서버가 포트를 연 다음에 하나씩 시작합니다 (모델을 한꺼번에 불러오지 않도록). 서버가 모두 뜨면 `Local tunnel:` 줄이 찍힙니다.
- 수준끼리는 동시에 돌고, 한 수준 안에서는 에피소드를 하나씩 돌립니다 (서버 하나는 세션 하나만 받음). 끝난 에피소드는 건너뛰므로 같은 명령을 다시 돌리면 이어서 합니다.
- 결과: task×수준별 성공/에피소드 표, seen·unseen 성공률과 한 번이라도 성공한 task 수, `results.json`. 에피소드마다 `client.log`와 `summary.json`이 남습니다.
- 속도: 서버마다 처음 두 호출은 compile 때문에 약 2분 15초와 30초가 걸리고, 그 뒤로는 서버 2개가 동시에 돌아도 호출당 약 2.2초입니다. 기본 설정(task 16개 × 10 에피소드, 평균 horizon 525 step)이면 수준들이 동시에 돌아서 전체 3시간쯤으로 예상합니다. 다 돌면 서버 작업은 `scancel`로 끝냅니다.
