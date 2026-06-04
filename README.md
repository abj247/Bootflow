# Principled Ensemble Exploration for Generative RL Policies

[![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![JAX 0.4.27](https://img.shields.io/badge/JAX-0.4.27-green.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> We prove that naive ensemble exploration fails for generative RL policies due to a gradient alignment failure mode, and introduce BootFlow with independent Q-targets and TIDE exploration to resolve it. Up to 7x improvement on MuJoCo locomotion.
>
> **[Report](https://drive.google.com/file/d/1VQIPrwMNuE8TFQq6mTTeI1EP9OQJCRyJ/view?usp=sharing)** &nbsp;|&nbsp; **Built on [SDAC](https://arxiv.org/abs/2502.00361) (Ma et al., ICML 2025)**

![BootFlow method overview](images/method_1.png)

---


Run any experiment:
```bash
XLA_FLAGS='--xla_gpu_deterministic_ops=true' CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=.1 \
  python scripts/train_fullbootflow.py --num_heads 5 --exploration ucb --redq_m 2 <env flag> --seed 100
```



## Quick Start

### Install

```bash
uv venv --python 3.9 && source .venv/bin/activate
uv pip install "jax[cuda12]==0.4.27" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
uv pip install -r requirements.txt && uv pip install -e .
```

<details>
<summary>Alternative: conda</summary>

```bash
conda create -n bootflow python=3.9 -y && conda activate bootflow
pip install "jax[cuda12]==0.4.27" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt && pip install -e .
```
</details>

### Reproduce main result

```bash
# FullBootFlow UCB K=5 
XLA_FLAGS='--xla_gpu_deterministic_ops=true' CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=.1 \
  python scripts/train_fullbootflow.py \
    --num_heads 5 --exploration ucb --ucb_beta 1.0 --redq_m 2 \
    --env Ant-v4 --seed 100
```

```bash
# FullBootFlow TIDE K=5 
XLA_FLAGS='--xla_gpu_deterministic_ops=true' CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=.1 \
  python scripts/train_fullbootflow.py \
    --num_heads 5 --exploration idfm --idfm_beta 2.0 --idfm_mode corr --redq_m 2 \
    --env Ant-v4 --seed 100
```

```bash
# SDAC baseline (K=1, reduces to vanilla SDAC)
XLA_FLAGS='--xla_gpu_deterministic_ops=true' CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=.1 \
  python scripts/train_bootflow.py --num_heads 1 --env Ant-v4 --seed 100
```

### With flow matching backbone

Add `--flow_matching` to any command above:

```bash
XLA_FLAGS='--xla_gpu_deterministic_ops=true' CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=.1 \
  python scripts/train_fullbootflow.py \
    --num_heads 5 --exploration ucb --redq_m 2 --flow_matching \
    --env Ant-v4 --seed 100
```

### Multiple seeds

```bash
for SEED in 100 200 300; do
  XLA_FLAGS='--xla_gpu_deterministic_ops=true' CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=.1 \
    python scripts/train_fullbootflow.py \
      --num_heads 5 --exploration ucb --redq_m 2 --env Ant-v4 --seed $SEED &
done
```

Run `python scripts/train_fullbootflow.py --help` for all available flags.



---

## Project Structure

```
                                                                                                                                                                                          
  ├── scripts/                                                                                                                                                                                 
  │   ├── train_fullbootflow.py        # FullBootFlow: K heads + K Q-nets (main method)                                                                                                        
  │   ├── train_bootflow.py            # BootFlow: K heads, shared twin-Q                                                                                                                      
  │   ├── train_mujoco.py              # SDAC + 9 baselines (--alg sdac|sac|...)                                                                                                               
  │   ├── evaluate_checkpoints.py      # Batch evaluation across seeds                                                                                                                         
  │   └── inspect_results.py           # Plot training curves from log.csv                                                                                                                     
  │                                                                                                                                                                                            
  ├── relax/                                                                                   
  │   ├── algorithm/                                                                           
  │   │   ├── sdac_fullbootflow.py     # FullBootFlow algorithm (UCB, Thompson, TIDE)
  │   │   ├── sdac_bootflow.py         # BootFlow (shared Q variant)                           
  │   │   └── sdac.py                  # Original SDAC 
  │   │                                       
  │   ├── network/                                                                             
  │   │   ├── diffv2_fullbootflow.py   # K-head + K Q-net architecture 
  │   │   └── diffv2_bootflow.py       # K-head + shared Q architecture                        
  │   │                                                      
  │   ├── utils/                               
  │   │   └── diffusion.py             # GaussianDiffusion + FlowMatching classes                                                                                                              
  │   │                                                                                                                                                                                        
  │   └── trainer/                                                                                                                                                                             
  │       ├── off_policy.py            # Training loop with Thompson hooks                     
  │       └── evaluator.py             # Parallel evaluation subprocess
  │
  └── requirements.txt                      
```

---

## Baselines

```bash
# SDAC (diffusion policy, our base method)
XLA_FLAGS='--xla_gpu_deterministic_ops=true' CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=.1 \
  python scripts/train_mujoco.py --alg sdac --env Ant-v4 --seed 100

# SAC (Gaussian policy)
XLA_FLAGS='--xla_gpu_deterministic_ops=true' CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=.1 \
  python scripts/train_mujoco.py --alg sac --env Ant-v4 --seed 100
```

Other algorithms: `dpmd`, `idem`, `qvpo`, `dacer`, `dipo`, `qsm`, `dsact`.

---

## Weights & Biases

```bash
wandb login
# or
export WANDB_API_KEY='your-api-key-here'
```

W&B project is auto-detected from environment name. Override with `--wandb_project my_project`.

---


## Acknowledgement

Built on [SDAC/DPMD](https://github.com/mahaitongdae/diffusion_policy_online_rl) (Ma et al., ICML 2025) and [DACER](https://github.com/happy-yan/DACER-Diffusion-with-Online-RL.git). Inspired by [Bootstrapped DQN](https://arxiv.org/abs/1602.04621) (Osband et al., NeurIPS 2016).
