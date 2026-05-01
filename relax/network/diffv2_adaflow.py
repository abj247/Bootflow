"""
AdaFlow network: Single flow policy with K Q-network ensemble (REDQ-style).

Where BootFlow ensembles the policy (K velocity heads), AdaFlow keeps a
single velocity network but expands twin-Q to K independent Q-networks.
Thompson Sampling selects one Q per episode for action evaluation.
Adaptive best-of-N scales candidate count with Q-ensemble disagreement.

When num_q_heads=2 and adaptive_n=False, this reduces to vanilla SDAC.
"""

from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple, Union

import jax, jax.numpy as jnp
import haiku as hk
import math

from relax.network.blocks import Activation, QNet, DACERPolicyNet
from relax.utils.diffusion import GaussianDiffusion, FlowMatching
from relax.utils.jax_utils import random_key_from_data


class AdaflowParams(NamedTuple):
    q_params: tuple          # Tuple of K hk.Params
    target_q_params: tuple   # Tuple of K target hk.Params
    policy: hk.Params        # Single DACERPolicyNet (NOT multi-head)
    target_policy: hk.Params
    log_alpha: jax.Array


@dataclass
class AdaflowNet:
    """AdaFlow network: single flow policy + K Q-ensemble.

    Action selection uses Thompson-selected Q (one Q per episode) and
    optionally adaptive best-of-N candidate selection driven by Q-ensemble
    disagreement.
    """
    q: Callable[[hk.Params, jax.Array, jax.Array], jax.Array]
    policy: Callable[[hk.Params, jax.Array, jax.Array, jax.Array], jax.Array]
    num_timesteps: int
    act_dim: int
    num_particles: int       # fixed N when adaptive_n=False, n_max when True
    target_entropy: float
    noise_scale: float
    beta_schedule_scale: float
    num_q_heads: int = 5
    n_min: int = 4
    n_max: int = 64
    adaptive_n: bool = False
    beta_schedule_type: str = 'linear'
    flow_matching: bool = False

    @property
    def diffusion(self):
        if self.flow_matching:
            return FlowMatching(self.num_timesteps)
        return GaussianDiffusion(self.num_timesteps,
                                 self.beta_schedule_scale,
                                 self.beta_schedule_type)

    def get_action(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        """Sample action using Thompson-selected Q with optional adaptive N.

        policy_params is a tuple of:
            (policy_params_raw, log_alpha, active_q_params, q_std_normalized)
        where active_q_params is the Thompson-selected single Q-network params.
        """
        policy_params_raw, log_alpha, active_q_params, q_std_normalized = policy_params

        def model_fn(t, x):
            return self.policy(policy_params_raw, obs, x, t)

        def sample(key: jax.Array) -> Tuple[jax.Array, jax.Array]:
            act = self.diffusion.p_sample(key, model_fn, (*obs.shape[:-1], self.act_dim))
            q = self.q(active_q_params, obs, act)
            return act.clip(-1, 1), q

        key, noise_key = jax.random.split(key)

        if self.adaptive_n:
            # Always generate n_max candidates (JIT-friendly)
            keys = jax.random.split(key, self.n_max)
            acts, qs = jax.vmap(sample)(keys)

            # Scale N by Q-ensemble disagreement
            frac = jnp.clip(q_std_normalized, 0.0, 1.0)
            effective_n = self.n_min + frac * (self.n_max - self.n_min)
            effective_n = jnp.round(effective_n).astype(jnp.int32)

            # Mask candidates beyond effective_n
            indices = jnp.arange(self.n_max)
            mask = indices < effective_n
            qs = jnp.where(mask, qs, -jnp.inf)

            q_best_ind = jnp.argmax(qs, axis=0, keepdims=True)
            act = jnp.take_along_axis(acts, q_best_ind[..., None], axis=0).squeeze(axis=0)
        elif self.num_particles == 1:
            act, _ = sample(key)
        else:
            keys = jax.random.split(key, self.num_particles)
            acts, qs = jax.vmap(sample)(keys)
            q_best_ind = jnp.argmax(qs, axis=0, keepdims=True)
            act = jnp.take_along_axis(acts, q_best_ind[..., None], axis=0).squeeze(axis=0)

        act = act + jax.random.normal(noise_key, act.shape) * jnp.exp(log_alpha) * self.noise_scale
        return act

    def get_deterministic_action(self, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        """Deterministic action using min of ALL K Q-networks (conservative).

        policy_params is a tuple of:
            (policy_params_raw, log_alpha, all_q_params_tuple)
        where all_q_params_tuple is the full tuple of K Q-network params.
        """
        key = random_key_from_data(obs)
        policy_params_raw, log_alpha, all_q_params_tuple = policy_params

        def model_fn(t, x):
            return self.policy(policy_params_raw, obs, x, t)

        def sample(key: jax.Array) -> Tuple[jax.Array, jax.Array]:
            act = self.diffusion.p_sample(key, model_fn, (*obs.shape[:-1], self.act_dim))
            # Min of all K Q-networks for conservative evaluation
            all_qs = jnp.stack([self.q(all_q_params_tuple[i], obs, act)
                                for i in range(self.num_q_heads)])
            q = jnp.min(all_qs, axis=0)
            return act.clip(-1, 1), q

        n = self.n_max if self.adaptive_n else self.num_particles
        if n == 1:
            act, _ = sample(key)
        else:
            keys = jax.random.split(key, n)
            acts, qs = jax.vmap(sample)(keys)
            q_best_ind = jnp.argmax(qs, axis=0, keepdims=True)
            act = jnp.take_along_axis(acts, q_best_ind[..., None], axis=0).squeeze(axis=0)
        return act

    def q_evaluate(
        self, key: jax.Array, q_params: hk.Params, obs: jax.Array, act: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        q_mean, q_std = self.q(q_params, obs, act)
        z = jax.random.normal(key, q_mean.shape)
        z = jnp.clip(z, -3.0, 3.0)
        q_value = q_mean + q_std * z
        return q_mean, q_std, q_value


def create_adaflow_net(
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
    num_q_heads: int = 5,
    n_min: int = 4,
    n_max: int = 64,
    adaptive_n: bool = False,
    flow_matching: bool = False,
) -> Tuple[AdaflowNet, AdaflowParams]:

    q = hk.without_apply_rng(hk.transform(
        lambda obs, act: QNet(hidden_sizes, activation)(obs, act)))
    policy = hk.without_apply_rng(hk.transform(
        lambda obs, act, t: DACERPolicyNet(diffusion_hidden_sizes, activation)(obs, act, t)))

    @jax.jit
    def init(key, obs, act):
        keys = jax.random.split(key, num_q_heads + 1)
        q_params = tuple(q.init(keys[i], obs, act) for i in range(num_q_heads))
        target_q_params = tuple(q_params)
        policy_params = policy.init(keys[num_q_heads], obs, act, 0)
        target_policy_params = policy_params
        log_alpha = jnp.array(math.log(5), dtype=jnp.float32)
        return AdaflowParams(q_params, target_q_params,
                             policy_params, target_policy_params, log_alpha)

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    params = init(key, sample_obs, sample_act)

    net = AdaflowNet(
        q=q.apply,
        policy=policy.apply,
        num_timesteps=num_timesteps,
        act_dim=act_dim,
        target_entropy=-act_dim * target_entropy_scale,
        num_particles=num_particles,
        noise_scale=noise_scale,
        beta_schedule_scale=beta_schedule_scale,
        num_q_heads=num_q_heads,
        n_min=n_min,
        n_max=n_max,
        adaptive_n=adaptive_n,
        flow_matching=flow_matching,
    )
    return net, params
