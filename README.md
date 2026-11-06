# wam_project

DreamZero(Wan2.1-I2V-14B) 기반 world action model에 [tmrl](https://github.com/matthewh6/tmrl)식 context noise를 주어 RoboCasa365 human-teleop 데이터로 학습하고, 노이즈 수준이 context 조건부 분포와 marginal 사이를 잇는지 확인하는 프로젝트입니다. 코드는 `dreamzero/`([DreamZero](https://github.com/dreamzero0/dreamzero) ab790c1에 변경을 더한 것, 브랜치 `context-noise`), 학습·추론·검증 사용법은 [docs/usage.md](docs/usage.md), 실험의 질문·결과·그림·비디오는 비공개 페이지 https://claude.ai/artifact/FUuPLmcJbNsCcsnSyU5mgX 에 있습니다.

- 지금까지: video context 노이즈로 25k step 사전학습(ckpt-25000) → `y` 전체 노이즈로 5k step 이어 학습(ft-5000) → language·state 노이즈로 5k step(all-5000). video 노이즈는 tmrl의 interpolation처럼 동작하고(같은 task의 예측이 σ 0.89–0.97에서 겹침), language 정보는 σ 0.14–0.83에서 서서히 사라지며, state는 video가 있으면 무시됩니다(ft-5000도 같음). text 노이즈 없이 학습한 ft-5000은 text σ=1에서 깨지므로, 노이즈 아래의 매끄러운 반응은 학습된 것입니다. 시뮬레이션 성공률은 ckpt-25000의 tcont 0에서 50/100.
- checkpoint·샘플 (hangang `/scratch/t526a02/jooyoung/runs/`): `dreamzero_robocasa_context_noise/checkpoint-25000`, `dreamzero_robocasa_context_noise_y_tail/checkpoint-5000`, `dreamzero_robocasa_context_noise_all/checkpoint-5000`; 샘플은 각 run의 `context_noise_samples/`.
- 결과 (로컬 `~/project/WAM/results/`, 폴더마다 `results.md`): `dreamzero_robocasa_context_noise/`(학습 곡선), `context_noise_ckpt25000_sim_eval/`(시뮬레이션), `context_noise_interpolation_ckpt25000/`(y tail 누출), `context_noise_y_tail_finetune/`(ft-5000), `tmrl_interpolation_ft5000/`(tmrl 검증), `tmrl_interpolation_all/`(language·state); 페이지 원본은 `site/index.html`.
- 남은 것 (2026-11-06, 돌고 있는 job 없음): 중간 σ의 시뮬레이션 성공률, 노이즈 없이 학습한 baseline, ft-5000에 다른 task의 지시문을 바꿔 넣는 실험.
