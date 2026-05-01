"""
Record a video of a trained policy on a MetaWorld manipulation environment.
Uses offscreen rendering (no GUI needed).

Usage:
    python scripts/record_video_metaworld.py \
        --log_dir logs/push-v3/bootflow_2026-03-27_s100_K1_sdac_baseline \
        --num_episodes 3 \
        --output_path videos/push_v3_sdac.mp4
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["MUJOCO_GL"] = "egl"

import argparse
import pickle
from pathlib import Path

import numpy as np
import jax
import mujoco
import metaworld

from relax.utils.persistence import PersistFunction


def create_metaworld_env_for_render(name: str, seed: int = 0):
    """Create a MetaWorld env (no RelaxWrapper, no render_mode needed for manual rendering)."""
    mt = metaworld.MT1(name, seed=seed)
    env = mt.train_classes[name]()
    task = mt.train_tasks[0]
    env.set_task(task)
    return env


def record_metaworld_video(env_name, policy_fn, policy_params, num_episodes, output_path,
                           fps=20, width=480, height=480, camera_name=None):
    """Record episodes of a MetaWorld task to mp4."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = create_metaworld_env_for_render(env_name)
    model = env.model
    data = env.data

    frames = []
    ep_returns = []
    ep_successes = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        obs = obs.astype(np.float32)
        ep_ret = 0.0
        ep_success = False

        renderer = mujoco.Renderer(model, height=height, width=width)

        if camera_name:
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            cam.fixedcamid = cam_id
        else:
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.distance = 1.2
            cam.elevation = -25.0
            cam.azimuth = 145.0
            cam.lookat[:] = [0.0, 0.6, 0.2]

        step = 0
        done = False
        while not done:
            renderer.update_scene(data, camera=cam)
            frame = renderer.render()
            frames.append(frame)

            act = np.asarray(policy_fn(policy_params, obs))
            act = np.clip(act, -1.0, 1.0)

            # Rescale if action space is not [-1, 1]
            low = env.action_space.low
            high = env.action_space.high
            if np.any(low != -1.0) or np.any(high != 1.0):
                act = low + (act + 1.0) * 0.5 * (high - low)

            obs, reward, terminated, truncated, info = env.step(act)
            obs = obs.astype(np.float32)
            ep_ret += reward
            if info.get("success", 0.0) == 1.0:
                ep_success = True
            done = terminated or truncated
            step += 1

        renderer.close()
        ep_returns.append(ep_ret)
        ep_successes.append(ep_success)
        print(f"Episode {ep + 1}/{num_episodes}: return = {ep_ret:.2f}, success = {ep_success}, steps = {step}")

    env.close()

    try:
        import imageio
    except ImportError:
        print("imageio not found, installing...")
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio[ffmpeg]"])
        import imageio

    writer = imageio.get_writer(str(output_path), fps=fps, codec='libx264', quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    success_rate = sum(ep_successes) / len(ep_successes)
    print(f"\nSaved video to {output_path}")
    print(f"Total frames: {len(frames)}")
    print(f"Mean return: {np.mean(ep_returns):.2f} +/- {np.std(ep_returns):.2f}")
    print(f"Success rate: {success_rate:.0%} ({sum(ep_successes)}/{len(ep_successes)})")
    return ep_returns


def main():
    parser = argparse.ArgumentParser(description="Record video of trained policy on MetaWorld")
    parser.add_argument("--log_dir", type=str, required=True,
                        help="Path to the training log directory")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint filename (e.g. policy-500000-100000.pkl). "
                             "If not specified, uses the latest checkpoint.")
    parser.add_argument("--num_episodes", type=int, default=3,
                        help="Number of episodes to record")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output video path. Defaults to <log_dir>/video.mp4")
    parser.add_argument("--fps", type=int, default=20,
                        help="Video frames per second")
    parser.add_argument("--env", type=str, default=None,
                        help="MetaWorld task name (e.g. push-v3). If not specified, reads from config.yaml")
    parser.add_argument("--camera", type=str, default=None,
                        help="Camera name: corner, corner2, corner3, topview, behindGripper, gripperPOV. "
                             "Default: custom free camera with good default angle")
    parser.add_argument("--seed", type=int, default=0,
                        help="Environment seed for task randomization")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        from relax.utils.fs import PROJECT_ROOT
        log_dir = PROJECT_ROOT / log_dir

    # Read env name from config if not specified
    if args.env is None:
        import yaml
        with open(log_dir / "config.yaml") as f:
            config = yaml.safe_load(f)
        env_name = config["env"]
    else:
        env_name = args.env

    # Find checkpoint
    if args.checkpoint is None:
        policy_files = sorted(log_dir.glob("policy-*.pkl"))
        if not policy_files:
            raise FileNotFoundError(f"No policy checkpoints found in {log_dir}")
        checkpoint_path = policy_files[-1]
        print(f"Using latest checkpoint: {checkpoint_path.name}")
    else:
        checkpoint_path = log_dir / args.checkpoint

    # Load persisted policy structure (deterministic — same as locomotion record_video.py)
    policy_structure = PersistFunction.load(log_dir / "deterministic.pkl")

    @jax.jit
    def policy_fn(policy_params, obs):
        return policy_structure(policy_params, obs).clip(-1, 1)

    # Load checkpoint params
    with open(checkpoint_path, "rb") as f:
        policy_params = pickle.load(f)

    print(f"Environment: {env_name}")
    print(f"Checkpoint: {checkpoint_path.name}")

    output_path = args.output_path
    if output_path is None:
        output_path = log_dir / "video.mp4"

    record_metaworld_video(
        env_name, policy_fn, policy_params,
        args.num_episodes, output_path, args.fps,
        camera_name=args.camera,
    )


if __name__ == "__main__":
    main()
