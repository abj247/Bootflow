# ESAC — follow-up experiments

Command matrix for the remaining multi-seed ablations and controls. Written for a 4-GPU
machine (A100s): with ~4 jobs per GPU you have ~16 slots, so Waves 1+2 can run
simultaneously and everything below finishes in roughly a day.

## Setup (once per machine)

```bash
git clone https://github.com/abj247/ESAC.git && cd ESAC
# Locomotion env (py3.9): jax[cuda12]==0.4.27 + requirements.txt + `pip install -e .`
# MetaWorld env (py3.10): same + metaworld
wandb login            # or use the offline fallback below
mkdir -p nohup_rebuttal
```

Every command uses this prefix (spell it out literally — zsh does not word-split variables):

```bash
setsid nohup env XLA_FLAGS='--xla_gpu_deterministic_ops=true' XLA_PYTHON_CLIENT_MEM_FRACTION=0.10 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=<gpu> python -u <script> <flags> > nohup_rebuttal/<name>.out 2>&1 &
```

**No wandb login?** Some wandb versions crash at init even offline ("No API key configured").
Add `WANDB_MODE=offline WANDB_API_KEY=0000000000000000000000000000000000000000` to the prefix
and `wandb sync <run_dir>/wandb/offline-run-*` after logging in.

**After any launch:** check the `.out` file within 10 min for `ptxas fatal` / `Traceback`
(a bad CUDA/driver combo kills runs at startup on some nodes). Verify flags via the new run's
`logs/<env>/<run>/config.yaml`.

## Wave 1 — highest value

### 1a. UCB multi-seed, flow-matching backbone (10 runs, ~8 h ea)
Completes the multi-seed exploration-strategy table (TIDE vs UCB). Ant + Walker2d s200/300
may already be done elsewhere — check `logs/` before launching duplicates.

```bash
i=0
for ENV in Walker2d-v4 Hopper-v4 HalfCheetah-v4; do
  for SEED in 200 300 400 500; do
    [ "$ENV" = Walker2d-v4 ] && [ $SEED -lt 400 ] && continue   # s200/300 running elsewhere
    GPU=$((i % 4)); i=$((i+1))
    setsid nohup env XLA_FLAGS='--xla_gpu_deterministic_ops=true' XLA_PYTHON_CLIENT_MEM_FRACTION=0.10 \
      XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$GPU python -u scripts/train_fullbootflow.py \
      --num_heads 5 --exploration ucb --ucb_beta 1.0 --redq_m 2 --flow_matching \
      --env $ENV --seed $SEED --suffix ucb_fm_s$SEED --wandb_project rebuttal-ucb-fm \
      > nohup_rebuttal/ucb_fm_${ENV%-v4}_s$SEED.out 2>&1 &
    sleep 5
  done
done
```

### 1b. K=1 flow-matching control (17 runs, ~2 h ea — cheap fillers)
Single-head flow matching: separates the backbone effect from the ensembling effect.

```bash
i=0
for ENV in Ant-v4 Walker2d-v4 Hopper-v4 HalfCheetah-v4; do
  for SEED in 200 300 400 500; do
    GPU=$((i % 4)); i=$((i+1))
    setsid nohup env XLA_FLAGS='--xla_gpu_deterministic_ops=true' XLA_PYTHON_CLIENT_MEM_FRACTION=0.10 \
      XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$GPU python -u scripts/train_bootflow.py \
      --num_heads 1 --flow_matching --env $ENV --seed $SEED --suffix k1_fm_s$SEED \
      --wandb_project rebuttal-k1-fm > nohup_rebuttal/k1_fm_${ENV%-v4}_s$SEED.out 2>&1 &
    sleep 5
  done
done
# plus HalfCheetah seed 100 (no K=1 FM run exists there at all):
# ... same command with --env HalfCheetah-v4 --seed 100
```

## Wave 2

### 2a. TIDE design ablation, Ant-v4 (6 runs, ~8 h ea)
Isolates TIDE's two components. Needs the `--idfm_num_levels` flag (in this repo).

```bash
for SEED in 100 200 300; do
  # gate OFF (raw disagreement bonus):
  ... scripts/train_fullbootflow.py --num_heads 5 --exploration idfm --idfm_beta 2.0 --idfm_mode raw \
      --redq_m 2 --flow_matching --env Ant-v4 --seed $SEED --suffix tide_raw_b2_s$SEED \
      --wandb_project rebuttal-tide-ablation
  # terminal-only disagreement (single probe point t=0):
  ... scripts/train_fullbootflow.py --num_heads 5 --exploration idfm --idfm_beta 2.0 --idfm_mode corr \
      --idfm_num_levels 1 --redq_m 2 --flow_matching --env Ant-v4 --seed $SEED --suffix tide_d0_b2_s$SEED \
      --wandb_project rebuttal-tide-ablation
done
```

### 2b. Compute-matched baselines, Ant-v4 (6 runs)
Needs the `--updates_per_step` flag in `scripts/train_mujoco.py` (in this repo).
SDAC ×5 updates = same total gradient count as the K=5 ensemble; SAC ×5 = standard UTD 1.0.

```bash
for SEED in 100 200 300; do
  ... scripts/train_mujoco.py --alg sdac --updates_per_step 5 --env Ant-v4 --seed $SEED \
      --suffix sdac_upi5_s$SEED --wandb_project rebuttal-compute-matched      # ~10 h
  ... scripts/train_mujoco.py --alg sac  --updates_per_step 5 --env Ant-v4 --seed $SEED \
      --suffix sac_upi5_s$SEED  --wandb_project rebuttal-compute-matched      # ~3 h
done
```

### 2c. Thompson multi-seed, flow-matching (8 runs, ~7 h ea — lowest priority, cut first)

```bash
# ENV in {Ant-v4, Walker2d-v4} x SEED in {200,300,400,500}
... scripts/train_fullbootflow.py --num_heads 5 --exploration thompson --redq_m 2 --flow_matching \
    --env $ENV --seed $SEED --suffix thompson_fm_s$SEED --wandb_project rebuttal-thompson-fm
```

## Wave 3 — optional

- **Swimmer SDAC seeds** (4 runs, ~3 h ea): the Swimmer analysis baseline has only 1 completed
  seed. `scripts/train_mujoco.py --alg sdac --env Swimmer-v4 --seed {200..500} --suffix swimmer_sdac_s<seed>`
- **MetaWorld peg-insert diagnostic** (2 runs, running elsewhere; needs the py3.10 env):
  `scripts/train_fullbootflow.py --num_heads 5 --exploration idfm --idfm_beta 2.0 --idfm_mode corr --redq_m 2 --env peg-insert-side-v3 --seed 100 --total_step 500000 --start_step 5000`
  and the K=1 equivalent via `train_bootflow.py`; afterwards run `scripts/probe_q_modality_metaworld.py`
  (Q-landscape distribution + action-space UMAP on the saved checkpoints).

## Aggregating results

Runs land in `logs/{env}/{alg}_{timestamp}_s{seed}_{suffix}/log.csv` with checkpoints every
`total_step/20`. Suffixes are greppable: `ucb_fm_`, `k1_fm_`, `tide_raw_`, `tide_d0_`,
`sdac_upi5_`, `sac_upi5_`, `thompson_fm_`, `swimmer_sdac_`. The final-return convention used
for all tables: max of the 10-point rolling mean of eval return over the last 50% of training,
reported as mean ± std over seeds.
