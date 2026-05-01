"""
Unified BootFlow training script.

Usage:
    # Vanilla SDAC behavior (K=1)
    XLA_FLAGS='--xla_gpu_deterministic_ops=true' CUDA_VISIBLE_DEVICES=0 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=.1 \
    python scripts/train_bootflow.py --num_heads 1 --seed 100

    # BootFlow with Thompson sampling (K=5)
    XLA_FLAGS='--xla_gpu_deterministic_ops=true' CUDA_VISIBLE_DEVICES=0 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=.1 \
    python scripts/train_bootflow.py --num_heads 5 --exploration thompson --seed 100

    # BootFlow without Thompson (per-step random, for ablation)
    XLA_FLAGS='--xla_gpu_deterministic_ops=true' CUDA_VISIBLE_DEVICES=0 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=.1 \
    python scripts/train_bootflow.py --num_heads 5 --exploration random_per_step --seed 100
"""

import argparse
import os.path
from pathlib import Path
import time
from functools import partial
import yaml

import jax, jax.numpy as jnp

from relax.algorithm.sdac_bootflow import BootflowSDAC
from relax.network.diffv2_bootflow import create_bootflow_net
from relax.buffer import TreeBuffer
from relax.trainer.off_policy import OffPolicyTrainer
from relax.env import create_env, create_vector_env, _is_metaworld
from relax.utils.experience import Experience, BootflowExperience
from relax.utils.fs import PROJECT_ROOT
from relax.utils.random_utils import seeding
from relax.utils.log_diff import log_git_details

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Environment & training
    parser.add_argument("--env", type=str, default="HalfCheetah-v4",
                        help="MuJoCo: HalfCheetah-v4, Hopper-v4, Walker2d-v4, Ant-v4, Humanoid-v4. "
                             "MetaWorld: reach-v3, drawer-close-v3, door-open-v3, "
                             "button-press-topdown-v3, peg-insert-side-v3, shelf-place-v3, sweep-into-v3")
    parser.add_argument("--suffix", type=str, default="bootflow")
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
    parser.add_argument("--updates_per_step", type=int, default=1,
                        help="Number of gradient updates per env step (increase for single-env like MetaWorld)")
    parser.add_argument("--upi_decay", action="store_true", default=False,
                        help="Linearly decay updates_per_step from initial value to 1 over training")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="W&B project name (default: 'metaworld' for -v3 envs, 'diffusion_online_rl' otherwise)")
    parser.add_argument("--beta_schedule_scale", type=float, default=0.8)
    parser.add_argument("--beta_schedule_type", type=str, default='linear')
    parser.add_argument("--flow_matching", action="store_true", default=False,
                        help="Use flow matching (velocity prediction) instead of DDPM diffusion (noise prediction)")

    # BootFlow-specific
    parser.add_argument("--num_heads", type=int, default=5,
                        help="Number of velocity network heads (K=1 = vanilla SDAC)")
    parser.add_argument("--exploration", type=str, default="thompson",
                        choices=["thompson", "random_per_step", "ucb"],
                        help="thompson: pick one head per episode. random_per_step: random head each step. "
                             "ucb: pool candidates from all K heads, select by Q + beta*disagreement")
    parser.add_argument("--bootstrap_prob", type=float, default=0.8,
                        help="Probability each sample is included for each head")
    parser.add_argument("--mask_mode", type=str, default="random",
                        choices=["random", "stored"],
                        help="random: generate masks during update. stored: store masks in buffer")
    parser.add_argument("--prior_scale", type=float, default=1.0,
                        help="Scale for randomized prior functions (Osband 2018). 0=disable priors")
    parser.add_argument("--disagree_coeff", type=float, default=0.0,
                        help="Disagreement exploration bonus coefficient (Pathak 2019). "
                             "0.0=original BootFlow, >0=add head disagreement as intrinsic reward")
    parser.add_argument("--diversity_coeff", type=float, default=0.0,
                        help="Diversity loss coefficient (MED-RL, ICLR 2022). "
                             "0.0=off, >0=penalize heads for producing similar outputs")
    parser.add_argument("--timestep_disagree", action="store_true", default=False,
                        help="Enable timestep-stratified disagreement (strategic vs execution uncertainty)")
    parser.add_argument("--adaptive_mask", action="store_true", default=False,
                        help="Enable adaptive bootstrap masking based on Q-disagreement")
    parser.add_argument("--ot_diversity_coeff", type=float, default=0.0,
                        help="OT diversity regularization coefficient (0=off)")
    parser.add_argument("--ucb_beta", type=float, default=1.0,
                        help="UCB exploration bonus scale (only used with --exploration ucb). "
                             "Higher = more optimistic exploration toward disagreement states")
    parser.add_argument("--independent_backbones", action="store_true", default=False,
                        help="Use K fully independent backbone networks instead of shared backbone + K heads. "
                             "~K× more parameters but enables genuine head diversity for Thompson sampling.")
    args = parser.parse_args()

    if args.debug:
        from jax import config
        config.update("jax_disable_jit", True)

    # K=1 overrides: no bootstrapping needed
    if args.num_heads == 1:
        args.exploration = "random_per_step"
        args.bootstrap_prob = 1.0
        args.mask_mode = "random"

    master_seed = args.seed
    master_rng, _ = seeding(master_seed)
    env_seed, env_action_seed, eval_env_seed, buffer_seed, init_network_seed, train_seed = map(
        int, master_rng.integers(0, 2**32 - 1, 6)
    )
    init_network_key = jax.random.key(init_network_seed)
    train_key = jax.random.key(train_seed)
    del init_network_seed, train_seed

    if _is_metaworld(args.env):
        # MetaWorld envs don't support the custom vector env wrappers
        env, obs_dim, act_dim = create_env(args.env, env_seed, env_action_seed)
        print(f"MetaWorld env detected: {args.env} (single env, no vectorization)")
    elif args.num_vec_envs > 0:
        env, obs_dim, act_dim = create_vector_env(args.env, args.num_vec_envs, env_seed, env_action_seed, mode="futex")
    else:
        env, obs_dim, act_dim = create_env(args.env, env_seed, env_action_seed)
    eval_env = None

    hidden_sizes = [args.hidden_dim] * args.hidden_num
    diffusion_hidden_sizes = [args.diffusion_hidden_dim] * args.hidden_num

    # Buffer: use BootflowExperience if mask_mode=stored, else standard Experience
    if args.mask_mode == "stored":
        buffer_example = BootflowExperience.create_example(obs_dim, act_dim, args.num_heads)
        buffer = TreeBuffer.from_example(buffer_example, size=int(1e6), seed=buffer_seed, remove_batch_dim=False)
    else:
        buffer = TreeBuffer.from_experience(obs_dim, act_dim, size=int(1e6), seed=buffer_seed)

    def mish(x: jax.Array):
        return x * jnp.tanh(jax.nn.softplus(x))

    print(f"BootFlow | K={args.num_heads} | exploration={args.exploration} | "
          f"mask_mode={args.mask_mode} | bootstrap_prob={args.bootstrap_prob} | "
          f"prior_scale={args.prior_scale} | disagree_coeff={args.disagree_coeff} | "
          f"independent_backbones={args.independent_backbones}")

    agent, params = create_bootflow_net(
        init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
        num_timesteps=args.diffusion_steps,
        num_particles=args.num_particles,
        noise_scale=args.noise_scale,
        beta_schedule_scale=args.beta_schedule_scale,
        num_heads=args.num_heads,
        prior_scale=args.prior_scale,
        independent_backbones=args.independent_backbones,
        flow_matching=args.flow_matching,
    )

    algorithm = BootflowSDAC(
        agent, params,
        lr=args.lr,
        alpha_lr=args.alpha_lr,
        delay_alpha_update=args.delay_alpha_update,
        lr_schedule_end=args.lr_schedule_end,
        bootstrap_prob=args.bootstrap_prob,
        exploration=args.exploration,
        mask_mode=args.mask_mode,
        disagree_coeff=args.disagree_coeff,
        diversity_coeff=args.diversity_coeff,
        num_envs=1 if _is_metaworld(args.env) else args.num_vec_envs,
        ucb_beta=args.ucb_beta,
        timestep_disagree=args.timestep_disagree,
        adaptive_mask=args.adaptive_mask,
        ot_diversity_coeff=args.ot_diversity_coeff,
    )

    if args.cluster:
        PROJECT_ROOT = Path('/n/netscratch/nali_lab_seas/Lab/haitongma/sdac_logs')

    # Build experiment name
    parts = ["bootflow", time.strftime("%Y-%m-%d_%H-%M-%S"), f"s{args.seed}",
             f"K{args.num_heads}", args.exploration,
             f"p{args.bootstrap_prob}", f"mask-{args.mask_mode}",
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
        update_per_iteration=args.updates_per_step,
        upi_decay=args.upi_decay,
        evaluate_env=eval_env,
        save_policy_every=int(args.total_step / 20),
        warmup_with="random",
        log_path=exp_dir,
        update_log_n_step=1 if args.debug else 1000,
        wandb_project=wandb_project,
        env_name=args.env,
    )

    # Use appropriate experience example for warmup
    if args.mask_mode == "stored":
        trainer.setup(BootflowExperience.create_example(obs_dim, act_dim, args.num_heads, trainer.batch_size))
    else:
        trainer.setup(Experience.create_example(obs_dim, act_dim, trainer.batch_size))

    log_git_details(log_file=os.path.join(exp_dir, 'dacer.diff'))

    args_dict = vars(args)
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as yaml_file:
        yaml.dump(args_dict, yaml_file)
    trainer.run(train_key)
