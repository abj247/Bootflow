"""
Record a video of a trained SDAC policy on a MuJoCo environment.
Uses offscreen rendering (no GUI needed).

Usage:
    python scripts/record_video.py \
        --log_dir logs/HalfCheetah-v4/sdac_2026-02-28_10-34-03_s100_test_use_atp1 \
        --checkpoint policy-900000-180000.pkl \
        --num_episodes 3 \
        --output_path videos/halfcheetah_sdac.mp4
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["MUJOCO_GL"] = "egl"  # Use EGL for headless rendering

import argparse
import pickle
from pathlib import Path

import numpy as np
import jax
import gymnasium as gym

from relax.utils.persistence import PersistFunction


def record_video(env_name, policy_fn, policy_params, num_episodes, output_path, fps=30, width=480, height=480):
    """Record episodes to an mp4 video with a tracking camera."""
    import mujoco

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create env (no render_mode needed - we render manually with tracking camera)
    env = gym.make(env_name)

    # Make the floor plane infinite and set up a clear B&W checkerboard.
    # MuJoCo renders a plane as infinite when size[0] and size[1] are 0.
    model = env.unwrapped.model
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    if floor_id >= 0:
        model.geom_size[floor_id] = [0.0, 0.0, 1.5]

        # Replace the floor texture with a high-contrast B&W checkerboard
        mat_id = model.geom_matid[floor_id]
        if mat_id >= 0:
            tex_id = model.mat_texid[mat_id]
            if tex_id >= 0:
                tex_h = model.tex_height[tex_id]
                tex_w = model.tex_width[tex_id]
                adr = model.tex_adr[tex_id]
                tex = model.tex_rgb[adr:adr + tex_h * tex_w * 3].reshape(tex_h, tex_w, 3)
                half_h, half_w = tex_h // 2, tex_w // 2
                tex[:half_h, :half_w] = [230, 230, 230]
                tex[half_h:, half_w:] = [230, 230, 230]
                tex[:half_h, half_w:] = [50, 50, 50]
                tex[half_h:, :half_w] = [50, 50, 50]
                model.mat_texrepeat[mat_id] = [1.0, 1.0]

    frames = []
    ep_returns = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        obs = obs.astype(np.float32)
        ep_ret = 0.0
        done = False

        # Setup offscreen renderer
        data = env.unwrapped.data
        renderer = mujoco.Renderer(model, height=height, width=width)

        # Use FREE camera so we can manually lock lookat to the torso
        # every frame. TRACKING mode can lag behind fast-moving agents.
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE

        # Env-specific camera settings
        if 'Ant' in env_name or 'Humanoid' in env_name:
            cam.distance = 6.0
            cam.elevation = -30.0
            cam.azimuth = 45.0
        else:
            cam.distance = 5.0
            cam.elevation = -15.0
            cam.azimuth = 90.0

        torso_id = model.body('torso').id

        while not done:
            # Lock camera center on the torso position every frame
            cam.lookat[:] = data.xpos[torso_id]

            renderer.update_scene(data, camera=cam)
            frame = renderer.render()
            frames.append(frame)

            act = np.asarray(policy_fn(policy_params, obs))
            act = np.clip(act, -1.0, 1.0)
            obs, reward, terminated, truncated, _ = env.step(act)
            obs = obs.astype(np.float32)
            ep_ret += reward
            done = terminated or truncated

        renderer.close()
        ep_returns.append(ep_ret)
        print(f"Episode {ep + 1}/{num_episodes}: return = {ep_ret:.1f}")

    env.close()

    # Save video using imageio
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

    print(f"\nSaved video to {output_path}")
    print(f"Total frames: {len(frames)}")
    print(f"Mean return: {np.mean(ep_returns):.1f} +/- {np.std(ep_returns):.1f}")
    return ep_returns


def main():
    parser = argparse.ArgumentParser(description="Record video of trained SDAC policy")
    parser.add_argument("--log_dir", type=str, required=True,
                        help="Path to the training log directory")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint filename (e.g. policy-900000-180000.pkl). "
                             "If not specified, uses the latest checkpoint.")
    parser.add_argument("--num_episodes", type=int, default=3,
                        help="Number of episodes to record")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output video path. Defaults to <log_dir>/video.mp4")
    parser.add_argument("--fps", type=int, default=30,
                        help="Video frames per second")
    parser.add_argument("--env", type=str, default=None,
                        help="Environment name. If not specified, reads from config.yaml")
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
        # Find latest policy checkpoint
        policy_files = sorted(log_dir.glob("policy-*.pkl"))
        if not policy_files:
            raise FileNotFoundError(f"No policy checkpoints found in {log_dir}")
        checkpoint_path = policy_files[-1]
        print(f"Using latest checkpoint: {checkpoint_path.name}")
    else:
        checkpoint_path = log_dir / args.checkpoint

    # Load persisted policy structure
    policy_structure = PersistFunction.load(log_dir / "deterministic.pkl")

    @jax.jit
    def policy_fn(policy_params, obs):
        return policy_structure(policy_params, obs).clip(-1, 1)

    # Load checkpoint params
    with open(checkpoint_path, "rb") as f:
        policy_params = pickle.load(f)

    print(f"Environment: {env_name}")
    print(f"Checkpoint: {checkpoint_path.name}")

    # Output path
    output_path = args.output_path
    if output_path is None:
        output_path = log_dir / "video.mp4"

    record_video(env_name, policy_fn, policy_params, args.num_episodes, output_path, args.fps)


if __name__ == "__main__":
    main()
