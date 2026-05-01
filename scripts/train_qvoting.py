"""
Q-Weighted Head Voting training script.

Separate pipeline from BootFlow — always uses ALL K heads for action selection.

Usage:
    # Q-voting with disagreement bonus (K=5)
    XLA_FLAGS='--xla_gpu_deterministic_ops=true' CUDA_VISIBLE_DEVICES=0 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=.1 \
    python scripts/train_qvoting.py --env Ant-v4 --num_heads 5 \
        --voting_mode q_plus_disagree --voting_beta 1.0 --seed 100

    # Q-voting without disagreement (pure Q selection across K heads)
    python scripts/train_qvoting.py --env Ant-v4 --num_heads 5 \
        --voting_mode q_only --seed 100
"""

import argparse
import os.path
from pathlib import Path
import time
from functools import partial
import yaml

import jax, jax.numpy as jnp

from relax.algorithm.sdac_qvoting import QVotingSDAC
from relax.network.diffv2_qvoting import create_qvoting_net
from relax.buffer import TreeBuffer
from relax.trainer.off_policy import OffPolicyTrainer
from relax.env import create_env, create_vector_env, _is_metaworld
from relax.utils.experience import Experience
from relax.utils.fs import PROJECT_ROOT
from relax.utils.random_utils import seeding
from relax.utils.log_diff import log_git_details

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Environment & training
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--suffix", type=str, default="qvoting")
    parser.add_argument("--num_vec_envs", type=int, default=5)
    parser.add_argument("--hidden_num", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--diffusion_steps", type=int, default=20)
    parser.add_argument("--diffusion_hidden_dim", type=int, default=256)
    parser.add_argument("--start_step", type=int, default=int(3e4))
    parser.add_argument("--total_step", type=int, default=int(1e6))
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_schedule_end", type=float, default=3e-5)
    parser.add_argument("--alpha_lr", type=float, default=7e-3)
    parser.add_argument("--delay_alpha_update", type=float, default=250)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--num_particles", type=int, default=32)
    parser.add_argument("--noise_scale", type=float, default=0.1)
    parser.add_argument("--cluster", default=False, action="store_true")
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="W&B project name (default: 'metaworld' for -v3 envs, 'diffusion_online_rl' otherwise)")
    parser.add_argument("--beta_schedule_scale", type=float, default=0.8)
    parser.add_argument("--beta_schedule_type", type=str, default='linear')
    parser.add_argument("--flow_matching", action="store_true", default=False,
                        help="Use flow matching (velocity prediction) instead of DDPM diffusion (noise prediction)")

    # Q-Voting specific
    parser.add_argument("--num_heads", type=int, default=5,
                        help="Number of velocity network heads")
    parser.add_argument("--voting_mode", type=str, default="q_plus_disagree",
                        choices=["q_only", "q_plus_disagree"],
                        help="q_only: select by Q-value only. "
                             "q_plus_disagree: Q + beta*disagreement (optimistic)")
    parser.add_argument("--voting_beta", type=float, default=1.0,
                        help="Disagreement bonus scale for voting (only used with q_plus_disagree)")
    parser.add_argument("--bootstrap_prob", type=float, default=0.8,
                        help="Probability each sample is included for each head")
    parser.add_argument("--prior_scale", type=float, default=1.0,
                        help="Scale for randomized prior functions")
    parser.add_argument("--disagree_coeff", type=float, default=0.0,
                        help="Disagreement reward bonus coefficient")
    parser.add_argument("--diversity_coeff", type=float, default=0.0,
                        help="Diversity loss coefficient (MED-RL)")
    args = parser.parse_args()

    if args.debug:
        from jax import config
        config.update("jax_disable_jit", True)

    master_seed = args.seed
    master_rng, _ = seeding(master_seed)
    env_seed, env_action_seed, eval_env_seed, buffer_seed, init_network_seed, train_seed = map(
        int, master_rng.integers(0, 2**32 - 1, 6)
    )
    init_network_key = jax.random.key(init_network_seed)
    train_key = jax.random.key(train_seed)
    del init_network_seed, train_seed

    if _is_metaworld(args.env):
        env, obs_dim, act_dim = create_env(args.env, env_seed, env_action_seed)
        print(f"MetaWorld env detected: {args.env} (single env, no vectorization)")
    elif args.num_vec_envs > 0:
        env, obs_dim, act_dim = create_vector_env(args.env, args.num_vec_envs, env_seed, env_action_seed, mode="futex")
    else:
        env, obs_dim, act_dim = create_env(args.env, env_seed, env_action_seed)
    eval_env = None

    hidden_sizes = [args.hidden_dim] * args.hidden_num
    diffusion_hidden_sizes = [args.diffusion_hidden_dim] * args.hidden_num

    buffer = TreeBuffer.from_experience(obs_dim, act_dim, size=int(1e6), seed=buffer_seed)

    def mish(x: jax.Array):
        return x * jnp.tanh(jax.nn.softplus(x))

    print(f"QVoting | K={args.num_heads} | mode={args.voting_mode} | "
          f"beta={args.voting_beta} | bootstrap_prob={args.bootstrap_prob} | "
          f"prior_scale={args.prior_scale} | disagree_coeff={args.disagree_coeff}")

    agent, params = create_qvoting_net(
        init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
        num_timesteps=args.diffusion_steps,
        num_particles=args.num_particles,
        noise_scale=args.noise_scale,
        beta_schedule_scale=args.beta_schedule_scale,
        num_heads=args.num_heads,
        prior_scale=args.prior_scale,
        flow_matching=args.flow_matching,
    )

    algorithm = QVotingSDAC(
        agent, params,
        lr=args.lr,
        alpha_lr=args.alpha_lr,
        delay_alpha_update=args.delay_alpha_update,
        lr_schedule_end=args.lr_schedule_end,
        bootstrap_prob=args.bootstrap_prob,
        disagree_coeff=args.disagree_coeff,
        diversity_coeff=args.diversity_coeff,
        voting_beta=args.voting_beta,
        voting_mode=args.voting_mode,
    )

    if args.cluster:
        PROJECT_ROOT = Path('/n/netscratch/nali_lab_seas/Lab/haitongma/sdac_logs')

    # Build experiment name
    parts = ["qvoting", time.strftime("%Y-%m-%d_%H-%M-%S"), f"s{args.seed}",
             f"K{args.num_heads}", args.voting_mode,
             f"vb{args.voting_beta}", f"p{args.bootstrap_prob}",
             f"dc{args.disagree_coeff}", f"dv{args.diversity_coeff}"]
    if args.suffix:
        parts.append(args.suffix)
    exp_name = "_".join(parts)
    exp_dir = PROJECT_ROOT / "logs" / args.env / exp_name

    wandb_project = args.wandb_project if args.wandb_project else ("metaworld" if _is_metaworld(args.env) else "diffusion_online_rl")

    trainer = OffPolicyTrainer(
        env=env,
        algorithm=algorithm,
        buffer=buffer,
        start_step=args.start_step,
        total_step=args.total_step,
        sample_per_iteration=1,
        evaluate_env=eval_env,
        save_policy_every=int(args.total_step / 20),
        warmup_with="random",
        log_path=exp_dir,
        update_log_n_step=1 if args.debug else 1000,
        wandb_project=wandb_project,
        env_name=args.env,
    )

    trainer.setup(Experience.create_example(obs_dim, act_dim, trainer.batch_size))

    log_git_details(log_file=os.path.join(exp_dir, 'dacer.diff'))

    args_dict = vars(args)
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as yaml_file:
        yaml.dump(args_dict, yaml_file)
    trainer.run(train_key)
