from pathlib import Path
import subprocess
import sys
from typing import Callable, Optional, Tuple

import jax
import numpy as np
from gymnasium import Env
from tqdm import tqdm
from tensorboardX import SummaryWriter
from tensorboardX.summary import hparams
import wandb

from relax.algorithm import Algorithm
from relax.buffer import ExperienceBuffer
from relax.env.vector import VectorEnv
from relax.trainer.accumulator import SampleLog, VectorSampleLog, UpdateLog, Interval
from relax.utils.experience import Experience


class OffPolicyTrainer:
    def __init__(
        self,
        env: Env,
        algorithm: Algorithm,
        buffer: ExperienceBuffer,
        log_path: Path,
        batch_size: int = 256,
        start_step: int = 1000,
        total_step: int = int(1e6),
        sample_per_iteration: int = 1,
        update_per_iteration: int = 1,
        evaluate_env: Optional[Env] = None,
        evaluate_every: int = 10000,
        evaluate_n_episode: int = 20,
        sample_log_n_episode: int = 10,
        update_log_n_step: int = 1000,
        done_info_keys: Tuple[str, ...] = (),
        save_policy_every: int = 10000,
        save_value: bool = True,
        hparams: Optional[dict] = None,
        policy_pkl_template: str = "policy-{sample_step}-{update_step}.pkl",
        warmup_with: str = "random",  # "policy" or "random"
        wandb_project: str = "diffusion_online_rl",
        env_name: Optional[str] = None,
        upi_decay: bool = False,
    ):
        self.env = env
        self.algorithm = algorithm
        self.buffer = buffer
        self.batch_size = batch_size
        self.start_step = start_step
        self.total_step = total_step
        self.sample_per_iteration = sample_per_iteration
        self.update_per_iteration = update_per_iteration
        self.upi_decay = upi_decay
        self.log_path = log_path
        self.policy_pkl_template = policy_pkl_template
        self.evaluate_env = evaluate_env
        self.evaluate_every = evaluate_every
        self.evaluate_n_episode = evaluate_n_episode
        self.sample_log_n_episode = sample_log_n_episode
        self.update_log_n_step = update_log_n_step
        self.done_info_keys = done_info_keys
        self.save_policy_every = save_policy_every
        self.hparams = hparams
        self.warmup_with = warmup_with
        self.save_value = save_value
        # TODO: make EpisodeLog and Experience configurable
        # TODO: re-add done_info_keys support
        # TODO: re-add evaluation support

        if isinstance(self.env.unwrapped, VectorEnv):
            self.is_vec = True
            self.sample_log = VectorSampleLog(self.env.unwrapped.num_envs)
        else:
            self.is_vec = False
            self.sample_log = SampleLog()
        self.update_log = UpdateLog()
        self.last_metrics = {}
        self._wandb_batch = {}
        self._last_sample_step = 0
        # The following two depends on sample_step, which may not update by one only
        self.sample_log_interval = Interval(self.sample_log_n_episode)
        self.save_policy_interval = Interval(self.save_policy_every)
        # self.eval_interval = Interval()
        self.env_id = env_name or (env.spec.id if env.spec is not None else env.unwrapped.__class__.__name__)
        wandb.init(project=wandb_project,
                   name=f"{self.env_id}_{log_path.name}",
                   dir=log_path,
                   group=self.env_id)
        wandb.define_metric("sample/*", step_metric="sample/step")
        wandb.define_metric("update/*", step_metric="update/step")
        wandb.define_metric("evaluate/*", step_metric="evaluate/step")
        wandb.define_metric("episode/*", step_metric="episode/count")

    def setup(self, dummy_data: Experience):
        self.algorithm.warmup(dummy_data)

        # Setup logger
        self.logger = SummaryWriter(str(self.log_path))
        self.progress = tqdm(total=self.total_step, desc="Sample Step", disable=None, dynamic_ncols=True)

        self.algorithm.save_policy_structure(self.log_path, dummy_data.obs[0])
        if self.save_value:
            self.algorithm.save_q_structure(self.log_path, dummy_obs=dummy_data.obs[0], dummy_action=dummy_data.action[0])
        self.evaluator = subprocess.Popen(
            [
                sys.executable,
                "-m", "relax.trainer.evaluator",
                str(self.log_path),
                "--env", self.env_id,
                "--num_episodes", str(self.evaluate_n_episode),
                "--seed", str(0),
            ],
            stdin=subprocess.PIPE,
            bufsize=0,
        )

    def warmup(self, key: jax.Array, obs: np.ndarray):
        step = 0
        key_fn = jax.jit(lambda step: jax.random.fold_in(key, step))
        while len(self.buffer) < self.start_step:
            step += 1
            if self.warmup_with == "random":
                action = self.env.action_space.sample()
            elif self.warmup_with == "policy":
                action = self.algorithm.get_action(key_fn(step), obs)
            else:
                raise ValueError(f"Invalid warmup_with {self.warmup_with}!")
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            experience = Experience.create(obs, action, reward, terminated, truncated, next_obs, info)
            if self.is_vec:
                self.buffer.add_batch(experience)
            else:
                self.buffer.add(experience)

            if np.any(terminated) or np.any(truncated):
                obs, _ = self.env.reset()
            else:
                obs = next_obs
        return obs

    def sample(self, sample_key: jax.Array, obs: np.ndarray):
        sl = self.sample_log

        action = self.algorithm.get_action(sample_key, obs)
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        experience = Experience.create(obs, action, reward, terminated, truncated, next_obs, info)
        if self.is_vec:
            self.buffer.add_batch(experience)
        else:
            self.buffer.add(experience)

        any_done = sl.add(reward, terminated, truncated, info)

        if any_done:
            if self.is_vec:
                for ret in sl.last_episode_returns:
                    wandb.log({"episode/return": ret, "episode/count": sl.sample_episode})
            else:
                wandb.log({"episode/return": sl.last_episode_return, "episode/count": sl.sample_episode})
            if self.sample_log_interval.check(sl.sample_episode):
                sl.log(self.add_scalar)
                self.flush_scalars(use_global_step=True)
            self.progress.update(sl.sample_step - self.progress.n)

            # Thompson sampling: select new head at episode boundary (per-env)
            if hasattr(self.algorithm, 'select_new_head'):
                head_key = jax.random.fold_in(sample_key, sl.sample_episode)
                done_mask = (np.asarray(terminated) | np.asarray(truncated))
                self.algorithm.select_new_head(head_key, done_mask=done_mask)

            obs, _ = self.env.reset()
        else:
            obs = next_obs

        return obs

    def update(self, update_key: jax.Array):
        ul = self.update_log
        data = self.buffer.sample(self.batch_size)
        info, dist_info = self.algorithm.update(update_key, data)

        ul.add(info)

        if ul.update_step % self.update_log_n_step == 0:
            self.add_hist(dist_info, ul.update_step * 5)
            ul.log(self.add_scalar)
            self.flush_scalars()

    def train(self, key: jax.Array):
        key, warmup_key = jax.random.split(key)

        obs, _ = self.env.reset()
        obs = self.warmup(warmup_key, obs)

        # Thompson sampling: select initial head before first episode
        if hasattr(self.algorithm, 'select_new_head'):
            self.algorithm.select_new_head(jax.random.fold_in(key, 0))

        iter_key_fn = create_iter_key_fn(key, self.sample_per_iteration, self.update_per_iteration)
        sl, ul = self.sample_log, self.update_log

        self.progress.unpause()
        while sl.sample_step <= self.total_step:
            sample_keys, update_keys = iter_key_fn(sl.sample_step)

            for i in range(self.sample_per_iteration):
                obs = self.sample(sample_keys[i], obs)

            if self.upi_decay and self.update_per_iteration > 1:
                progress = sl.sample_step / self.total_step
                current_upi = max(1, round(self.update_per_iteration - (self.update_per_iteration - 1) * progress))
            else:
                current_upi = self.update_per_iteration
            for i in range(current_upi):
                self.update(update_keys[i])

            if self.save_policy_interval.check(sl.sample_step):
                policy_pkl_name = self.policy_pkl_template.format(
                    sample_step=sl.sample_step,
                    update_step=ul.update_step,
                )
                self.algorithm.save_policy(self.log_path / policy_pkl_name)
                
                if self.save_value:
                    self.algorithm.save_q(self.log_path / policy_pkl_name.replace('policy', 'value'))
                

                command = f"{sl.sample_step},{self.log_path / policy_pkl_name}\n"
                self.evaluator.stdin.write(command.encode())

    def add_scalar(self, tag: str, value: float, step: int):
        self.last_metrics[tag] = value
        prefix = tag.split("/")[0]
        self._wandb_batch[tag] = value
        self._wandb_batch[f"{prefix}/step"] = step
        if prefix == "sample":
            self._last_sample_step = step
        self.logger.add_scalar(tag, value, step)

    def flush_scalars(self, use_global_step=False):
        if self._wandb_batch:
            if use_global_step:
                wandb.log(dict(self._wandb_batch), step=self._last_sample_step)
            else:
                wandb.log(dict(self._wandb_batch))
            self._wandb_batch.clear()
        self.logger.flush()
        
    def add_hist(self, info_hist, step):
        for tag, value in info_hist.items():
            self.logger.add_histogram(tag, np.array(value), step)
            prefix = tag.split("/")[0]
            wandb.log({tag: wandb.Histogram(np.array(value)), f"{prefix}/step": step}, commit=False)
        self.logger.flush()

    def run(self, key: jax.Array):
        try:
            self.train(key)
        except KeyboardInterrupt:
            pass
        finally:
            self.finish()

    def finish(self):
        self.env.close()
        self.algorithm.save(self.log_path / "state.pkl")
        if self.hparams is not None and len(self.last_metrics) > 0:
            exp, ssi, sei = hparams(self.hparams, self.last_metrics)
            self.logger.file_writer.add_summary(exp)
            self.logger.file_writer.add_summary(ssi)
            self.logger.file_writer.add_summary(sei)
        self.logger.close()
        self.progress.close()
        self.evaluator.stdin.close()
        self.evaluator.wait()

def create_iter_key_fn(key: jax.Array, sample_per_iteration: int, update_per_iteration: int) -> Callable[[int], Tuple[jax.Array, jax.Array]]:
    def iter_key_fn(step: int):
        iter_key = jax.random.fold_in(key, step)
        sample_key, update_key = jax.random.split(iter_key)
        if sample_per_iteration > 1:
            sample_key = jax.random.split(sample_key, sample_per_iteration)
        else:
            sample_key = (sample_key,)
        if update_per_iteration > 1:
            update_key = jax.random.split(update_key, update_per_iteration)
        else:
            update_key = (update_key,)
        return sample_key, update_key

    iter_key_fn = jax.jit(iter_key_fn)
    iter_key_fn(0)  # Warm up
    return iter_key_fn
