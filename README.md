# wam_project

WAM 프로젝트 작업 저장소입니다.

`dreamzero/`는 [DreamZero](https://github.com/dreamzero0/dreamzero) (ab790c1) 원본에 아래 변경을 더한 것입니다.

## Noised context (tmrl)

[tmrl](https://github.com/matthewh6/tmrl)처럼 context에 노이즈를 준 채로 WAM을 학습합니다 (`action_head_cfg.config.context_noise=true`).

- 노이즈 대상: teacher forcing의 깨끗한 영상 latent 흐름, `y`의 첫 프레임 latent(`context_noise_y_tail=true`면 `y`의 latent 전체), 첫 프레임 CLIP 특징. 언어와 로봇 상태는 기본으로 깨끗하게 둡니다 (아래 `context_noise_text`, `context_noise_state`).
- 샘플마다 수준 하나: `tcont ~ U[0,1)`, `σ = 5·tcont / (1 + 4·tcont)` (모델 스케줄의 shift 5), `x̃ = (1-σ)·x + σ·ε`. CLIP 특징은 RMS 정규화 후 같은 σ로 섞습니다.
- 모델에는 깨끗한 토큰의 timestep으로 수준을 알려 줍니다 (`aug_t = 1000·σ`). `tcont=0`이면 원래 모델과 같습니다.
- 코드: `dreamzero/groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py` (`WANPolicyHead.forward`)
- 언어와 로봇 상태 (`context_noise_text=true`, `context_noise_state=true`; 학습 스크립트의 `CONTEXT_NOISE_TEXT`, `CONTEXT_NOISE_STATE`): 샘플마다 입력별로 수준 `tcont ~ U[0,1)`을 따로 뽑아 같은 식으로 섞습니다. 수준은 모델에 알려주지 않습니다 (tmrl의 policy처럼 입력에서 읽습니다).
  - 언어: T5 임베딩(512 토큰 × 4096)의 모든 위치에 RMS 0.085인 노이즈를 줍니다 (`context_noise_text_rms`, RoboCasa 프롬프트 토큰의 RMS 0.0847). 패딩 위치(0)도 노이즈로 바꿔서 토큰 수가 새지 않게 합니다. CFG의 negative prompt는 깨끗하게 둡니다.
  - 상태: q99 정규화된 실제 차원(`state_mask`)에 N(0,1) 노이즈를 줍니다 (정규화된 상태의 RMS는 약 0.75).
  - action은 입력이 아니라 flow matching의 예측 대상이라 노이즈 대상이 아닙니다.
  - 정보가 사라지는 수준은 입력마다 다릅니다. 이상적 관찰자가 두 context를 구분하는 척도 Δ = (1−σ)‖c−c′‖/(σ·노이즈 표준편차)에서, 프롬프트 사이 거리는 노이즈 표준편차의 약 840배(video latent는 약 100배, 상태는 약 2배)입니다. 그래서 언어 정보는 σ ≈ 0.99(tcont ≈ 0.95) 근처에서야, 상태 정보는 σ ≈ 0.6(tcont ≈ 0.25)에서 사라지므로, sweep의 수준을 입력에 맞춥니다.
- `y`의 뒤 latent: `y`는 `[첫 프레임, 0 프레임들]`의 VAE latent입니다. causal VAE 인코더가 뒤 latent(특히 1, 2번째)에 첫 프레임 정보를 담습니다. 노이즈를 첫 latent에만 주고 학습한 `dreamzero_robocasa_context_noise`는 그 latent에서 장면을 읽어서, `tcont=1`에서도 장면이 그대로 남습니다 (아래 "예측 분포"). `context_noise_y_tail=true`면 뒤 latent에도 같은 σ로 노이즈를 줍니다. 학습 스크립트는 이것이 기본이고, 이 옵션이 없던 checkpoint는 전처럼 첫 latent에만 줍니다. checkpoint-25000에서 이 옵션으로 5000 step 이어서 학습한 `dreamzero_robocasa_context_noise_y_tail`은 `tcont=1`에서 장면을 모릅니다.

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
- 비교용 기준 모델: `CONTEXT_NOISE=false`. `CONTEXT_NOISE_Y_TAIL=false`면 `y`의 첫 latent에만 노이즈를 줍니다 (이 옵션 전의 run).
- 언어와 상태까지: `CONTEXT_NOISE_TEXT=true CONTEXT_NOISE_STATE=true`. `dreamzero_robocasa_context_noise_all`은 `dreamzero_robocasa_context_noise_y_tail`의 checkpoint-5000에서 이 두 옵션으로 5000 step 이어서 학습한 run입니다.

  ```bash
  OUTPUT_DIR=$WORK/runs/dreamzero_robocasa_context_noise_all MAX_STEPS=5000 CHAIN_JOBS=1 CONTEXT_NOISE_TEXT=true CONTEXT_NOISE_STATE=true \
    sbatch scripts/train/robocasa_training_context_noise.sbatch pretrained_model_path=$WORK/runs/dreamzero_robocasa_context_noise_y_tail/checkpoint-5000
  ```
- 다른 run의 checkpoint 가중치로 새 run을 시작할 수 있습니다. optimizer와 LR 스케줄(warmup 5%, cosine)은 새로 시작합니다.

  ```bash
  OUTPUT_DIR=<새 run 폴더> MAX_STEPS=5000 CHAIN_JOBS=1 sbatch scripts/train/robocasa_training_context_noise.sbatch pretrained_model_path=<checkpoint>
  ```

  - 데이터 seed가 `42 + 0`이라 원래 run의 첫 작업과 같은 shard 순서를 다시 밟습니다. 그래서 새 run이 그 작업보다 짧으면 원래 run의 heldout shard는 방문하지 않습니다. 검증할 때는 `--shards-from`에 원래 checkpoint를 줍니다 (아래 "검증").
  - sbatch는 `PYTHONPATH`를 제출한 checkout으로 잡습니다. 환경의 editable install은 학습 checkout(`projects/wam_project`)을 가리킵니다.
- 환경: DreamZero `pyproject.toml` 기준에 `pyarrow<20`, `pandas<3`, `matplotlib`이 더 필요합니다 (numpy 1.26). aarch64(GH200)에는 torch 2.8.0 cu128 휠이 없어 torch 2.9.0을 씁니다.

## 추론에서 context 노이즈 수준 주기

```python
result_batch, video_pred = policy.lazy_joint_forward_causal(batch, tcont_context=0.3)  # policy: GrootSimPolicy
```

- `tcont_context`: 0~1 실수, 또는 샘플마다 다른 값이면 `[B]` 텐서. `None`(기본)이면 원래 추론과 같습니다.
- `tcont_text`, `tcont_state`: 같은 형식의 언어·상태 수준 (`None`이면 깨끗). 언어 노이즈는 KV cache를 시작하는 호출에서 positive prompt에 한 번 주고 에피소드 동안 유지하며, 상태 노이즈는 호출마다 그 호출의 상태에 줍니다. 노이즈는 모두 아래의 seed 고정 generator에서 뽑습니다.
- 학습과 같은 식으로 KV cache에 넣는 관측 latent를 노이즈하고, 그 토큰의 timestep을 `1000·σ`로 줍니다.
- 첫 프레임 `y`(첫 latent, `context_noise_y_tail`이면 전체)와 CLIP 특징은 KV cache를 새로 시작하는 호출의 수준으로 한 번 노이즈합니다. 학습처럼 에피소드마다 수준 하나를 쓰는 게 기본이고, 호출마다 바꾸면 새로 들어오는 관측 프레임에만 적용됩니다.
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
- `--shards-from <checkpoint>`: 다른 run의 checkpoint에서 시작한 run(`pretrained_model_path`)은 그 checkpoint로 shard 묶음을 만듭니다. `--resume-steps`도 그 run의 것을 줍니다. 아래 "예측 분포"도 같습니다.
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
- 수준끼리는 동시에 돌고, 서버 하나는 세션 하나만 받습니다. 같은 수준을 여러 번 쓰면(`SERVE_LEVELS="0 0"`) 그 수준의 서버들이 에피소드를 나눠 돌립니다. 끝난 에피소드는 건너뛰므로 같은 명령을 다시 돌리면 이어서 합니다.
- 결과: task×수준별 성공/에피소드 표, seen·unseen 성공률과 한 번이라도 성공한 task 수, `results.json`. 에피소드마다 `client.log`와 `summary.json`이 남습니다.
- 예측 영상: `--predicted-videos N`이면 앞 N개 task의 에피소드 0에서, 모델이 행동과 함께 예측한 영상을 서버에서 받아 저장합니다 (서버 `get_video`: video latent를 VAE로 디코딩). `predicted.mp4`는 왼쪽·오른쪽·손목 카메라, `compare.mp4`는 같은 step의 시뮬레이터 화면(위, 모델 입력처럼 카메라마다 가운데 95%를 잘라 224로 줄임)과 예측(아래)입니다. 24 step 청크마다 8프레임이라 프레임 i가 step 3i입니다.
- 속도: 서버마다 처음 두 호출은 compile 때문에 약 2분 15초와 30초가 걸리고, 그 뒤로는 서버 2개가 동시에 돌아도 호출당 약 2.2초입니다. 기본 설정(task 16개 × 10 에피소드, 평균 horizon 525 step)이면 수준들이 동시에 돌아서 전체 3시간쯤으로 예상합니다. 다 돌면 서버 작업은 `scancel`로 끝냅니다.

### 예측 분포

```bash
# hangang, 학습 중이 아닌 checkout의 dreamzero 폴더 (GPU 4개)
sbatch scripts/inference/robocasa_context_noise_samples.sbatch <checkpoint> --resume-steps 0 9000 18000
# y의 첫 latent 뒤 latent들을 다른 context의 것으로 바꿔서
sbatch --time=02:00:00 scripts/inference/robocasa_context_noise_samples.sbatch <checkpoint> --resume-steps 0 9000 18000 --y-tail swap --seeds 2
# 수준을 영상 context 대신 언어(text)나 상태(state)에, 또는 셋 모두에 (나머지 입력은 깨끗)
sbatch scripts/inference/robocasa_context_noise_samples.sbatch <checkpoint> --resume-steps 0 --shards-from <원래 checkpoint> --modalities text --levels 0 0.5 0.75 0.875 0.9375 0.96875 0.984375 1
```

- `eval_utils/robocasa_context_noise_samples.py`는 수준마다 비디오와 action을 여러 seed로 뽑아, context 노이즈가 context 조건부 분포(수준 0)와 marginal(수준 1) 사이를 잇는지, 그 사이의 비디오가 그럴듯한지 봅니다.
- context: 데이터 loss의 heldout shard에서 task 24개 × trajectory 2개 = 48개. 임의의 step i부터 48 step 동안 지시문이 같은 곳을 고릅니다.
- 시뮬레이션 서버처럼 두 번 호출합니다. 첫 호출은 step i 프레임 1장, 둘째 호출은 step i … i+24 프레임 9장을 받고, 각각 action 24개와 프레임 8장(step i+3 … i+48)을 예측합니다. 예측 latent는 그 호출이 받은 GT latent 뒤에 붙여 디코딩합니다.
- 기본값: 수준 `0, 1/32, 1/16, 1/8, 3/16, 1/4, 3/8, 1/2, 1`, seed 5개 (`--levels`, `--seeds`로 바꿉니다). 같은 seed는 모든 수준과 context에서 같은 초기 노이즈와 context 노이즈를 씁니다.
- 결과 (`<run>/context_noise_samples/<checkpoint>[_y_tail_swap][_<modalities>]/`): `samples/<context>.npz` (예측·GT action, 예측·GT·VAE 복원 프레임의 CLIP 특징과 오차·선명도·움직임, 지시문의 T5 임베딩 `text_emb`와 두 호출의 정규화된 상태 `state`), `latents/<context>.pt` (예측 video latent), `videos/<context>_seed<k>.mp4` (앞 `--video-seeds`개 seed, 기본 2; 행은 GT와 수준들, 열은 왼쪽·오른쪽·손목 카메라).
- `--modalities`: 수준을 받는 입력 (`video`, `text`, `state` 중 여러 개, 기본 `video`). 나머지 입력은 깨끗합니다. 입력마다 정보가 사라지는 σ가 다르므로 (위 "Noised context") `--levels`를 맞춥니다.
- `--y-tail swap`: `y`의 첫 latent 뒤 latent들을 context 목록에서 절반 떨어진 context(다른 task)의 첫 프레임으로 만든 것으로 바꿉니다. latent마다 바뀐 비율은 `tail_change`로 저장됩니다.
- 끝난 context는 건너뛰므로 같은 명령을 다시 내면 이어서 합니다.
- 시간 (GH200 4개): 준비 약 15분, 그 뒤 GPU마다 context 12개를 context당 약 6분(첫 context는 compile 포함 약 9분)씩 돌려 전체 약 1시간 30분. `--y-tail swap --seeds 2`는 context당 약 3분, 전체 약 50분. 수준 8개 × seed 12개는 context당 약 13분, 전체 약 3시간이라 `--time=05:00:00`을 줍니다.
- checkpoint-25000: 수준 1에서도 예측 비디오가 CLIP으로 48개 context 중 자기 context를 97% 이상 맞혀, marginal이 아닙니다. `--y-tail swap`이면 첫 호출 프레임의 67–94%가 바꿔 넣은 장면을 그립니다 (위 "알려진 문제"). 노이즈가 지우는 것은 주로 둘째 호출의 새 관측이고, action은 σ ≥ 0.83에서야 크게 바뀝니다. 중간 수준의 비디오는 흐려지지 않고 action과 함께 움직입니다.
- `dreamzero_robocasa_context_noise_y_tail` checkpoint-5000 (checkpoint-25000에서 `context_noise_y_tail=true`로 5000 step): 수준 1에서는 48개 context 중 자기 것을 25%만 맞힙니다. 같은 task 안에서는 53–55%(우연 50%)라, 언어와 state만 아는 분포에 가깝습니다. `--y-tail swap`도 결과를 바꾸지 않습니다. 수준 0의 action 오차는 그대로이고 비디오 오차는 조금 작습니다 (0.081 / 0.083, 15.0 / 12.6; checkpoint-25000은 0.092 / 0.086, 16.1 / 13.3). 수준을 올리면 예측이 수준 1 쪽으로 고르게 옮겨 갑니다. σ ≤ 0.83의 비디오는 그럴듯하고, 수준 1에서는 지시문에 맞는 다른 부엌을 그립니다 (realism 0.744, checkpoint-25000은 0.799). checkpoint-2000에서 이미 대부분 바뀌었습니다.
- 같은 checkpoint-5000을 σ 0.83–1에 몰린 수준으로 다시 봤습니다 (`--levels 0 0.25 0.5 0.625 0.75 0.875 0.9375 1 --seeds 12 --video-seeds 12`). tmrl의 interpolation처럼 동작합니다. 같은 task의 두 context는 비디오 예측이 σ 0.893–0.972에서 겹치고 (nearest-neighbor TV 0.85 → 0.37), 다른 task는 지시문 때문에 끝까지 갈립니다. 두 호출의 겹침은 노이즈 단위로 잰 두 context의 거리 (1−σ)‖c−c′‖/σ에 대해 한 곡선에 놓입니다 (tmrl Theorem 1의 척도). 중간 σ는 수준 0과 1의 혼합이 아니라 점진적 변화이고, 샘플은 흐리거나 섞인 비디오가 아니라 다른 부엌이며 action과 함께 움직입니다. 가까운 context에서 빌려온다는 증거는 없습니다.
