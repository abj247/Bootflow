"""
FullBootFlow network: K independent flow policy heads + K independent Q-networks.

Combines IndependentBootflowPolicyNet (from BootFlow) with K independent
Q-networks (from AdaFlow). Each head k trains against its own Q_k's targets,
enabling genuine posterior diversity for Thompson sampling.

When num_heads=1, this reduces to vanilla SDAC.
"""

from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple, Union

import jax, jax.numpy as jnp
import haiku as hk
import math

from relax.network.blocks import Activation, QNet, scaled_sinusoidal_encoding, mlp
from relax.network.diffv2_bootflow import IndependentBootflowPolicyNet
from relax.utils.diffusion import GaussianDiffusion, FlowMatching
from relax.utils.jax_utils import random_key_from_data


class FullBootflowParams(NamedTuple):
    q_params: tuple           # K independent hk.Params
    target_q_params: tuple    # K target Q-params
    policy: hk.Params         # IndependentBootflowPolicyNet params (K independent backbones)
    target_policy: hk.Params
    log_alpha: jax.Array


@dataclass
class FullBootflowNet:
    """FullBootFlow network: K independent policy heads + K independent Q-networks.

    Each (head_k, Q_k) pair forms an independent actor-critic, enabling
    genuine posterior diversity for Thompson sampling. UCB pools candidates
    from all K heads and scores them using the full Q-ensemble.
    """
    q: Callable[[hk.Params, jax.Array, jax.Array], jax.Array]
    policy: Callable[[hk.Params, jax.Array, jax.Array, jax.Array, int], jax.Array]
    num_timesteps: int
    act_dim: int
    num_particles: int
    target_entropy: float
    noise_scale: float
    beta_schedule_scale: float
    num_heads: int = 5
    beta_schedule_type: str = 'linear'
    flow_matching: bool = False
    # Number of denoising-time probe points for the TIDE disagreement bonus:
    # 4 -> [0, T/4, T/2, 3T/4] (paper default); 1 -> [0] (terminal-only ablation)
    idfm_num_levels: int = 4

    @property
    def diffusion(self):
        if self.flow_matching:
            return FlowMatching(self.num_timesteps)
        return GaussianDiffusion(self.num_timesteps,
                                 self.beta_schedule_scale,
                                 self.beta_schedule_type)

    def _sample_action_pessimistic(self, key: jax.Array, policy_params_raw, log_alpha,
                                    q_params_a, q_params_b, obs, head_idx) -> jax.Array:
        """Sample action using a specific head + min(Q_a, Q_b) for pessimistic scoring."""
        def model_fn(t, x):
            return self.policy(policy_params_raw, obs, x, t, head_idx)

        def sample(key: jax.Array) -> Tuple[jax.Array, jax.Array]:
            act = self.diffusion.p_sample(key, model_fn, (*obs.shape[:-1], self.act_dim))
            q_a = self.q(q_params_a, obs, act)
            q_b = self.q(q_params_b, obs, act)
            q = jnp.minimum(q_a, q_b)  # twin-Q pessimism
            return act.clip(-1, 1), q

        key, noise_key = jax.random.split(key)
        if self.num_particles == 1:
            act, _ = sample(key)
        else:
            keys = jax.random.split(key, self.num_particles)
            acts, qs = jax.vmap(sample)(keys)
            q_best_ind = jnp.argmax(qs, axis=0, keepdims=True)
            act = jnp.take_along_axis(acts, q_best_ind[..., None], axis=0).squeeze(axis=0)
        act = act + jax.random.normal(noise_key, act.shape) * jnp.exp(log_alpha) * self.noise_scale
        return act

    def get_action_thompson(self, key: jax.Array, policy_params: hk.Params,
                            obs: jax.Array, head_idx: int) -> jax.Array:
        """Thompson: use head_k + min(Q_k, Q_{k+1}) for pessimistic scoring."""
        policy_params_raw, log_alpha, q_params_tuple = policy_params
        k = head_idx
        k_next = (k + 1) % len(q_params_tuple)
        return self._sample_action_pessimistic(key, policy_params_raw, log_alpha,
                                                q_params_tuple[k], q_params_tuple[k_next],
                                                obs, head_idx)

    def _sample_action_ucb(self, key: jax.Array, policy_params_raw, log_alpha,
                           q_params_tuple, obs, ucb_beta: float) -> jax.Array:
        """UCB: pool K*N candidates, score by mean(Q_all) + beta*std(Q_all)."""
        key, noise_key = jax.random.split(key)
        num_heads = self.num_heads

        # Generate candidates from each head
        all_acts_list = []
        for k in range(num_heads):
            def _make_model_fn(head_k):
                def model_fn(t, x):
                    return self.policy(policy_params_raw, obs, x, t, head_k)
                return model_fn

            def sample_k(sample_key):
                act = self.diffusion.p_sample(sample_key, _make_model_fn(k),
                                              (*obs.shape[:-1], self.act_dim))
                return act.clip(-1, 1)

            head_keys = jax.random.split(jax.random.fold_in(key, k), self.num_particles)
            acts_k = jax.vmap(sample_k)(head_keys)  # (N, *obs_batch, act_dim)
            all_acts_list.append(acts_k)

        # Pool K*N candidates
        pooled_acts = jnp.concatenate(all_acts_list, axis=0)  # (K*N, *obs_batch, act_dim)

        # For each candidate, evaluate ALL K Q-networks
        obs_expanded = jnp.broadcast_to(obs, pooled_acts.shape[:-1] + obs.shape[-1:])
        all_q_vals = jnp.stack([
            self.q(q_params_tuple[i], obs_expanded, pooled_acts)
            for i in range(num_heads)
        ], axis=0)  # (K, K*N, *obs_batch)

        q_mean = jnp.mean(all_q_vals, axis=0)  # (K*N, *obs_batch)
        q_std = jnp.std(all_q_vals, axis=0)    # (K*N, *obs_batch)

        if ucb_beta > 0 and num_heads > 1:
            scores = q_mean + ucb_beta * q_std
        else:
            scores = q_mean

        # Select best candidate
        best_idx = jnp.argmax(scores, axis=0, keepdims=True)
        act = jnp.take_along_axis(pooled_acts, best_idx[..., None], axis=0).squeeze(axis=0)

        # Add exploration noise
        act = act + jax.random.normal(noise_key, act.shape) * jnp.exp(log_alpha) * self.noise_scale
        return act

    def get_action_ucb(self, key: jax.Array, policy_params: hk.Params,
                       obs: jax.Array, ucb_beta: float) -> jax.Array:
        """UCB action selection using all K heads and all K Q-networks."""
        policy_params_raw, log_alpha, q_params_tuple = policy_params
        return self._sample_action_ucb(key, policy_params_raw, log_alpha,
                                       q_params_tuple, obs, ucb_beta)

    def _sample_action_idfm(self, key: jax.Array, policy_params_raw, log_alpha,
                             q_params_tuple, obs, idfm_beta: float,
                             idfm_mode: str = 'raw',
                             running_disagree: float = 0.0,
                             running_disagree_std: float = 1.0) -> jax.Array:
        """IDFM: pool K*N candidates, score by Q_mean + calibrated velocity disagreement.

        idfm_mode controls how the velocity disagreement bonus is calibrated:
          'raw': no calibration (original, works on hard envs, fails on easy)
          'qgate': multiply by Q_std / Q_std_max (Fix 3: Q-weighted gating)
          'floor': subtract running EMA of disagreement (Fix 1: noise floor subtraction)
          'corr': scale beta by correlation between disagree and Q_std (Fix 5)
        """
        num_heads = self.num_heads
        key, noise_key = jax.random.split(key)

        # Step 1: Generate candidates from all K heads
        all_acts_list = []
        for k in range(num_heads):
            def _make_model_fn(head_k):
                def model_fn(t, x):
                    return self.policy(policy_params_raw, obs, x, t, head_k)
                return model_fn

            def sample_k(sample_key):
                act = self.diffusion.p_sample(sample_key, _make_model_fn(k),
                                              (*obs.shape[:-1], self.act_dim))
                return act.clip(-1, 1)

            head_keys = jax.random.split(jax.random.fold_in(key, k), self.num_particles)
            acts_k = jax.vmap(sample_k)(head_keys)
            all_acts_list.append(acts_k)

        pooled_acts = jnp.concatenate(all_acts_list, axis=0)

        # Step 2: Q-values across ensemble
        obs_expanded = jnp.broadcast_to(obs, pooled_acts.shape[:-1] + obs.shape[-1:])
        all_q_vals = jnp.stack([
            self.q(q_params_tuple[i], obs_expanded, pooled_acts)
            for i in range(num_heads)
        ], axis=0)
        q_mean = jnp.mean(all_q_vals, axis=0)
        q_std = jnp.std(all_q_vals, axis=0)

        # Step 3: Integrated velocity disagreement across denoising timesteps
        T = self.num_timesteps
        timestep_levels = [0, T // 4, T // 2, 3 * T // 4][: self.idfm_num_levels]
        total_disagree = jnp.zeros_like(q_mean)

        for t_level in timestep_levels:
            t_arr = jnp.full(pooled_acts.shape[:-1], t_level, dtype=jnp.int32)
            head_velocities = []
            for k in range(num_heads):
                v_k = self.policy(policy_params_raw, obs_expanded, pooled_acts, t_arr, k)
                head_velocities.append(v_k)
            v_stack = jnp.stack(head_velocities, axis=0)
            disagree_t = jnp.mean(jnp.std(v_stack, axis=0), axis=-1)
            total_disagree = total_disagree + disagree_t

        info_gain = total_disagree / len(timestep_levels)

        # Step 4: Calibrate the bonus based on idfm_mode
        if idfm_mode == 'qgate':
            # Fix 3: Gate by Q-std (suppress when Q-networks agree)
            q_std_norm = q_std / (jnp.max(q_std) + 1e-8)
            scores = q_mean + idfm_beta * info_gain * q_std_norm
        elif idfm_mode == 'floor':
            # Fix 1: Subtract noise floor (running EMA of disagreement)
            calibrated = jnp.maximum(0.0, info_gain - running_disagree)
            scores = q_mean + idfm_beta * calibrated
        elif idfm_mode == 'corr':
            # Fix 5: Scale beta by per-step correlation between disagree and Q_std
            ig_flat = info_gain.flatten()
            qs_flat = q_std.flatten()
            ig_centered = ig_flat - jnp.mean(ig_flat)
            qs_centered = qs_flat - jnp.mean(qs_flat)
            corr = jnp.sum(ig_centered * qs_centered) / (
                jnp.sqrt(jnp.sum(ig_centered**2) * jnp.sum(qs_centered**2)) + 1e-8)
            effective_beta = idfm_beta * jnp.maximum(0.0, corr)
            scores = q_mean + effective_beta * info_gain
        elif idfm_mode == 'ema_corr':
            # Fix 5b: EMA-smoothed correlation (stable version of corr)
            # Compute per-step correlation, but use EMA from algorithm for stability
            ig_flat = info_gain.flatten()
            qs_flat = q_std.flatten()
            ig_centered = ig_flat - jnp.mean(ig_flat)
            qs_centered = qs_flat - jnp.mean(qs_flat)
            current_corr = jnp.sum(ig_centered * qs_centered) / (
                jnp.sqrt(jnp.sum(ig_centered**2) * jnp.sum(qs_centered**2)) + 1e-8)
            # Blend current correlation with EMA from algorithm
            # running_disagree carries the EMA correlation value
            ema_corr = 0.99 * running_disagree + 0.01 * current_corr
            effective_beta = idfm_beta * jnp.maximum(0.0, ema_corr)
            scores = q_mean + effective_beta * info_gain
        else:
            # 'raw': no calibration (original behavior)
            scores = q_mean + idfm_beta * info_gain

        # Step 5: Select best candidate
        best_idx = jnp.argmax(scores, axis=0, keepdims=True)
        act = jnp.take_along_axis(pooled_acts, best_idx[..., None], axis=0).squeeze(axis=0)

        act = act + jax.random.normal(noise_key, act.shape) * jnp.exp(log_alpha) * self.noise_scale
        return act

    def get_action_idfm(self, key: jax.Array, policy_params: hk.Params,
                        obs: jax.Array, idfm_beta: float,
                        idfm_mode: str = 'raw',
                        running_disagree: float = 0.0,
                        running_disagree_std: float = 1.0) -> jax.Array:
        """IDFM action selection with calibrated velocity disagreement."""
        policy_params_raw, log_alpha, q_params_tuple = policy_params
        return self._sample_action_idfm(key, policy_params_raw, log_alpha,
                                         q_params_tuple, obs, idfm_beta,
                                         idfm_mode, running_disagree, running_disagree_std)

    def get_action(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        """Default: head 0 + min(Q_0, Q_1) for pessimistic action scoring.

        Used inside JIT for Bellman target next_action and policy loss anchor.
        min(Q_0, Q_1) matches BootFlow's twin-Q pessimism — prevents overoptimistic
        action selection that causes Q-learning death spiral.
        """
        policy_params_raw, log_alpha, q_params_tuple = policy_params
        return self._sample_action_pessimistic(key, policy_params_raw, log_alpha,
                                                q_params_tuple[0],
                                                q_params_tuple[1 % len(q_params_tuple)],
                                                obs, head_idx=0)

    def get_deterministic_action(self, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        """Evaluation: pool candidates from ALL K heads, score by mean(Q) across all K Q-networks.

        This is UCB-style evaluation without exploration noise — uses all heads
        to generate K*num_particles candidates, scores by mean Q across ensemble.
        Much better than head-0-only for MetaWorld where heads may specialize.
        """
        key = random_key_from_data(obs)
        policy_params_raw, log_alpha, all_q_params_tuple = policy_params
        num_heads = self.num_heads

        all_acts_list = []
        for k in range(num_heads):
            def _make_model_fn(head_k):
                def model_fn(t, x):
                    return self.policy(policy_params_raw, obs, x, t, head_k)
                return model_fn

            def sample_k(sample_key):
                act = self.diffusion.p_sample(sample_key, _make_model_fn(k),
                                              (*obs.shape[:-1], self.act_dim))
                return act.clip(-1, 1)

            head_keys = jax.random.split(jax.random.fold_in(key, k), self.num_particles)
            acts_k = jax.vmap(sample_k)(head_keys)
            all_acts_list.append(acts_k)

        pooled_acts = jnp.concatenate(all_acts_list, axis=0)

        obs_expanded = jnp.broadcast_to(obs, pooled_acts.shape[:-1] + obs.shape[-1:])
        all_q_vals = jnp.stack([
            self.q(all_q_params_tuple[i], obs_expanded, pooled_acts)
            for i in range(num_heads)
        ], axis=0)

        scores = jnp.mean(all_q_vals, axis=0)

        best_idx = jnp.argmax(scores, axis=0, keepdims=True)
        act = jnp.take_along_axis(pooled_acts, best_idx[..., None], axis=0).squeeze(axis=0)
        return act

    def get_deterministic_action_head0(self, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        """Evaluation using head 0 only (original behavior, for comparison)."""
        key = random_key_from_data(obs)
        policy_params_raw, log_alpha, all_q_params_tuple = policy_params

        def model_fn(t, x):
            return self.policy(policy_params_raw, obs, x, t, 0)

        def sample(key: jax.Array) -> Tuple[jax.Array, jax.Array]:
            act = self.diffusion.p_sample(key, model_fn, (*obs.shape[:-1], self.act_dim))
            all_qs = jnp.stack([self.q(all_q_params_tuple[i], obs, act)
                                for i in range(self.num_heads)])
            q = jnp.min(all_qs, axis=0)
            return act.clip(-1, 1), q

        if self.num_particles == 1:
            act, _ = sample(key)
        else:
            keys = jax.random.split(key, self.num_particles)
            acts, qs = jax.vmap(sample)(keys)
            q_best_ind = jnp.argmax(qs, axis=0, keepdims=True)
            act = jnp.take_along_axis(acts, q_best_ind[..., None], axis=0).squeeze(axis=0)
        return act

    def get_all_head_actions(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        """Get actions from all K heads for uncertainty/disagreement estimation.
        Returns: (K, *obs_batch, act_dim)
        """
        keys = jax.random.split(key, self.num_heads)
        actions = []
        for k in range(self.num_heads):
            act = self.get_action_thompson(keys[k], policy_params, obs, head_idx=k)
            actions.append(act)
        return jnp.stack(actions, axis=0)

    def q_evaluate(
        self, key: jax.Array, q_params: hk.Params, obs: jax.Array, act: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        q_mean, q_std = self.q(q_params, obs, act)
        z = jax.random.normal(key, q_mean.shape)
        z = jnp.clip(z, -3.0, 3.0)
        q_value = q_mean + q_std * z
        return q_mean, q_std, q_value


def create_fullbootflow_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int],
    diffusion_hidden_sizes: Sequence[int],
    activation: Activation = jax.nn.relu,
    num_timesteps: int = 20,
    num_particles: int = 32,
    noise_scale: float = 0.05,
    target_entropy_scale: float = 0.9,
    beta_schedule_scale: float = 0.3,
    num_heads: int = 5,
    flow_matching: bool = False,
    idfm_num_levels: int = 4,
) -> Tuple[FullBootflowNet, FullBootflowParams]:

    q = hk.without_apply_rng(hk.transform(
        lambda obs, act: QNet(hidden_sizes, activation)(obs, act)))

    policy = hk.without_apply_rng(hk.transform(
        lambda obs, act, t, head_idx: IndependentBootflowPolicyNet(
            diffusion_hidden_sizes, activation, num_heads=num_heads
        )(obs, act, t, head_idx)))

    @jax.jit
    def init(key, obs, act):
        keys = jax.random.split(key, num_heads + 1)
        # K independent Q-networks with different keys
        q_params = tuple(q.init(keys[i], obs, act) for i in range(num_heads))
        target_q_params = tuple(q_params)
        # Single policy init (IndependentBootflowPolicyNet handles K backbones internally)
        policy_params = policy.init(keys[num_heads], obs, act, 0, None)
        target_policy_params = policy_params
        log_alpha = jnp.array(math.log(5), dtype=jnp.float32)
        return FullBootflowParams(q_params, target_q_params,
                                  policy_params, target_policy_params, log_alpha)

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    params = init(key, sample_obs, sample_act)

    net = FullBootflowNet(
        q=q.apply,
        policy=policy.apply,
        num_timesteps=num_timesteps,
        act_dim=act_dim,
        target_entropy=-act_dim * target_entropy_scale,
        num_particles=num_particles,
        noise_scale=noise_scale,
        beta_schedule_scale=beta_schedule_scale,
        num_heads=num_heads,
        flow_matching=flow_matching,
        idfm_num_levels=idfm_num_levels,
    )
    return net, params
