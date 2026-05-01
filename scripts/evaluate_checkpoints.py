"""
Evaluate all checkpoints in a log directory across multiple seeds.
Reports return and success rate for each checkpoint.

Usage:
    python scripts/evaluate_checkpoints.py \
        --log_dir logs/push-v3/<run-name> \
        --env push-v3 \
        --seeds 0 42 100 \
        --num_episodes 5
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import pickle
import csv
from pathlib import Path

import numpy as np
import jax
import metaworld

from relax.utils.persistence import PersistFunction


def create_eval_env(name, seed):
    mt = metaworld.MT1(name, seed=seed)
    env = mt.train_classes[name]()
    env.set_task(mt.train_tasks[0])
    return env


def evaluate(env, policy_fn, policy_params, num_episodes):
    returns = []
    successes = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        obs = obs.astype(np.float32)
        ep_ret = 0.0
        ep_success = False
        for _ in range(500):
            act = np.asarray(policy_fn(policy_params, obs))
            act = np.clip(act, -1.0, 1.0)
            obs, reward, terminated, truncated, info = env.step(act)
            obs = obs.astype(np.float32)
            ep_ret += reward
            if info.get("success", 0.0) == 1.0:
                ep_success = True
            if terminated or truncated:
                break
        returns.append(ep_ret)
        successes.append(ep_success)
    return returns, successes


def main():
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints across seeds")
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--env", type=str, default=None, help="If not set, reads from config.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 100])
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        from relax.utils.fs import PROJECT_ROOT
        log_dir = PROJECT_ROOT / log_dir

    if args.env is None:
        import yaml
        with open(log_dir / "config.yaml") as f:
            config = yaml.safe_load(f)
        env_name = config["env"]
    else:
        env_name = args.env

    # Load policy structure
    policy_structure = PersistFunction.load(log_dir / "deterministic.pkl")

    @jax.jit
    def policy_fn(policy_params, obs):
        return policy_structure(policy_params, obs).clip(-1, 1)

    # Find all checkpoints sorted by step
    checkpoints = sorted(log_dir.glob("policy-*.pkl"), key=lambda p: int(p.stem.split("-")[1]))

    print(f"Environment: {env_name}")
    print(f"Seeds: {args.seeds}")
    print(f"Episodes per seed: {args.num_episodes}")
    print(f"Checkpoints: {len(checkpoints)}")
    print()

    # Header
    seed_cols = [f"ret_s{s}" for s in args.seeds] + [f"suc_s{s}" for s in args.seeds]
    header = ["checkpoint", "step"] + seed_cols + ["mean_ret", "success_rate"]
    print(f"{'checkpoint':<35} {'step':>8}", end="")
    for s in args.seeds:
        print(f"  {'ret_s'+str(s):>10}", end="")
    for s in args.seeds:
        print(f"  {'suc_s'+str(s):>10}", end="")
    print(f"  {'mean_ret':>10}  {'success%':>8}")
    print("-" * (60 + 22 * len(args.seeds)))

    results = []

    for ckpt_path in checkpoints:
        step = int(ckpt_path.stem.split("-")[1])

        with open(ckpt_path, "rb") as f:
            params = pickle.load(f)

        all_returns = []
        all_successes = []
        seed_returns = []
        seed_success_rates = []

        for seed in args.seeds:
            env = create_eval_env(env_name, seed)
            rets, succs = evaluate(env, policy_fn, params, args.num_episodes)
            env.close()
            mean_ret = np.mean(rets)
            suc_rate = np.mean(succs)
            seed_returns.append(mean_ret)
            seed_success_rates.append(suc_rate)
            all_returns.extend(rets)
            all_successes.extend(succs)

        overall_ret = np.mean(all_returns)
        overall_suc = np.mean(all_successes)

        print(f"{ckpt_path.name:<35} {step:>8}", end="")
        for r in seed_returns:
            print(f"  {r:>10.1f}", end="")
        for s in seed_success_rates:
            print(f"  {s:>10.0%}", end="")
        print(f"  {overall_ret:>10.1f}  {overall_suc:>8.0%}")

        results.append({
            "checkpoint": ckpt_path.name,
            "step": step,
            "mean_ret": overall_ret,
            "success_rate": overall_suc,
            **{f"ret_s{s}": r for s, r in zip(args.seeds, seed_returns)},
            **{f"suc_s{s}": sr for s, sr in zip(args.seeds, seed_success_rates)},
        })

    # Save CSV
    output_path = args.output or (log_dir / "eval_checkpoints.csv")
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint", "step", "mean_ret", "success_rate"]
                                + [f"ret_s{s}" for s in args.seeds]
                                + [f"suc_s{s}" for s in args.seeds])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved results to {output_path}")

    # Find best checkpoint
    best = max(results, key=lambda r: (r["success_rate"], r["mean_ret"]))
    print(f"\nBest checkpoint: {best['checkpoint']} (step {best['step']})")
    print(f"  Mean return: {best['mean_ret']:.1f}")
    print(f"  Success rate: {best['success_rate']:.0%}")


if __name__ == "__main__":
    main()
