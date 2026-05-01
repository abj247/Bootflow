"""
Q-Weighted Head Voting network.

Same architecture as BootFlow (K heads + shared backbone), but action selection
always uses ALL K heads simultaneously via Q-weighted voting.
No Thompson sampling — every step pools K×N candidates.
"""

from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple, Union

import jax, jax.numpy as jnp
import haiku as hk
import math

from relax.network.blocks import Activation, QNet, scaled_sinusoidal_encoding, mlp
from relax.utils.diffusion import GaussianDiffusion, FlowMatching
from relax.utils.jax_utils import random_key_from_data

# Reuse BootflowPolicyNet and BootflowParams directly
from relax.network.diffv2_bootflow import BootflowPolicyNet, BootflowParams


@dataclass
class QVotingNet:
    """Q-Weighted Voting network.

    Action selection pools candidates from ALL K heads and selects by
    Q-value + optional disagreement bonus. No Thompson sampling.
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

    @property
    def diffusion(self):
        if self.flow_matching:
            return FlowMatching(self.num_timesteps)
        return GaussianDiffusion(self.num_timesteps,
                                 self.beta_schedule_scale,
                                 self.beta_schedule_type)

    def _sample_action_voting(self, key: jax.Array, policy_params_raw, log_alpha,
                               q1_params, q2_params, obs,
                               voting_beta: float) -> jax.Array:
        """Pool candidates from all K heads, select by Q + β×disagreement."""
        key, noise_key = jax.random.split(key)

        all_acts_list = []
        all_qs_list = []
        for k in range(self.num_heads):
            def _make_model_fn(head_k):
                def model_fn(t, x):
                    return self.policy(policy_params_raw, obs, x, t, head_k)
                return model_fn

            def sample_k(sample_key):
                act = self.diffusion.p_sample(sample_key, _make_model_fn(k),
                                               (*obs.shape[:-1], self.act_dim))
                act = act.clip(-1, 1)
                q1 = self.q(q1_params, obs, act)
                q2 = self.q(q2_params, obs, act)
                q = jnp.minimum(q1, q2)
                return act, q

            head_keys = jax.random.split(jax.random.fold_in(key, k), self.num_particles)
            acts_k, qs_k = jax.vmap(sample_k)(head_keys)
            all_acts_list.append(acts_k)
            all_qs_list.append(qs_k)

        pooled_acts = jnp.concatenate(all_acts_list, axis=0)  # (K*N, *obs_batch, act_dim)
        pooled_qs = jnp.concatenate(all_qs_list, axis=0)      # (K*N, *obs_batch)

        if voting_beta > 0 and self.num_heads > 1:
            # Disagreement bonus
            t_zero = jnp.zeros(pooled_acts.shape[:1], dtype=jnp.int32)
            obs_expanded = jnp.broadcast_to(obs, pooled_acts.shape[:-1] + obs.shape[-1:])
            head_preds = jnp.stack([
                self.policy(policy_params_raw, obs_expanded, pooled_acts, t_zero, k)
                for k in range(self.num_heads)
            ], axis=0)
            disagreement = jnp.mean(jnp.std(head_preds, axis=0), axis=-1)
            scores = pooled_qs + voting_beta * disagreement
        else:
            scores = pooled_qs

        best_idx = jnp.argmax(scores, axis=0, keepdims=True)
        act = jnp.take_along_axis(pooled_acts, best_idx[..., None], axis=0).squeeze(axis=0)
        act = act + jax.random.normal(noise_key, act.shape) * jnp.exp(log_alpha) * self.noise_scale
        return act

    def get_action(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array,
                   voting_beta: float = 0.0) -> jax.Array:
        """Action selection via Q-weighted voting across all heads."""
        policy_params_raw, log_alpha, q1_params, q2_params = policy_params
        return self._sample_action_voting(key, policy_params_raw, log_alpha,
                                           q1_params, q2_params, obs, voting_beta)

    def get_action_single_head(self, key: jax.Array, policy_params: hk.Params,
                                obs: jax.Array, head_idx: int = 0) -> jax.Array:
        """Single-head action (for Q-target computation inside JIT)."""
        policy_params_raw, log_alpha, q1_params, q2_params = policy_params

        def model_fn(t, x):
            return self.policy(policy_params_raw, obs, x, t, head_idx)

        def sample(sample_key):
            act = self.diffusion.p_sample(sample_key, model_fn, (*obs.shape[:-1], self.act_dim))
            act = act.clip(-1, 1)
            q1 = self.q(q1_params, obs, act)
            q2 = self.q(q2_params, obs, act)
            q = jnp.minimum(q1, q2)
            return act, q

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

    def get_deterministic_action(self, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        """Deterministic action using head 0 with no noise (evaluation)."""
        key = random_key_from_data(obs)
        policy_params_raw, log_alpha, q1_params, q2_params = policy_params
        log_alpha = -jnp.inf
        # Use single-head for deterministic (stable)
        def model_fn(t, x):
            return self.policy(policy_params_raw, obs, x, t, 0)
        def sample(sample_key):
            act = self.diffusion.p_sample(sample_key, model_fn, (*obs.shape[:-1], self.act_dim))
            act = act.clip(-1, 1)
            q1 = self.q(q1_params, obs, act)
            q2 = self.q(q2_params, obs, act)
            return act, jnp.minimum(q1, q2)
        keys = jax.random.split(key, self.num_particles)
        acts, qs = jax.vmap(sample)(keys)
        q_best_ind = jnp.argmax(qs, axis=0, keepdims=True)
        act = jnp.take_along_axis(acts, q_best_ind[..., None], axis=0).squeeze(axis=0)
        return act


def create_qvoting_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int],
    diffusion_hidden_sizes: Sequence[int],
    activation: Activation = jax.nn.relu,
    num_timesteps: int = 20,
    num_particles: int = 4,
    noise_scale: float = 0.05,
    target_entropy_scale: float = 0.9,
    beta_schedule_scale: float = 0.3,
    num_heads: int = 5,
    prior_scale: float = 1.0,
    flow_matching: bool = False,
) -> Tuple[QVotingNet, BootflowParams]:

    q = hk.without_apply_rng(hk.transform(
        lambda obs, act: QNet(hidden_sizes, activation)(obs, act)))

    policy = hk.without_apply_rng(hk.transform(
        lambda obs, act, t, head_idx: BootflowPolicyNet(
            diffusion_hidden_sizes, activation, num_heads=num_heads,
            prior_scale=prior_scale
        )(obs, act, t, head_idx)))

    @jax.jit
    def init(key, obs, act):
        q1_key, q2_key, policy_key = jax.random.split(key, 3)
        q1_params = q.init(q1_key, obs, act)
        q2_params = q.init(q2_key, obs, act)
        target_q1_params = q1_params
        target_q2_params = q2_params
        policy_params = policy.init(policy_key, obs, act, 0, None)
        target_policy_params = policy_params
        log_alpha = jnp.array(math.log(5), dtype=jnp.float32)
        return BootflowParams(
            q1_params, q2_params, target_q1_params, target_q2_params,
            policy_params, target_policy_params, log_alpha
        )

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    params = init(key, sample_obs, sample_act)

    net = QVotingNet(
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
    )
    return net, params
