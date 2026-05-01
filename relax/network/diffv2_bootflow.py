"""
BootFlow network: Diffv2Net with K independent output heads sharing a
common backbone. Each head predicts noise independently, enabling
epistemic uncertainty estimation via disagreement.

When num_heads=1, this reduces to vanilla SDAC (single head, no ensembling).
"""

from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple, Union

import jax, jax.numpy as jnp
import haiku as hk
import math

from relax.network.blocks import Activation, QNet, scaled_sinusoidal_encoding, mlp
from relax.utils.diffusion import GaussianDiffusion, FlowMatching
from relax.utils.jax_utils import random_key_from_data


class BootflowParams(NamedTuple):
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    policy: hk.Params          # single set of params containing backbone + all K heads
    target_policy: hk.Params
    log_alpha: jax.Array


@dataclass
class BootflowPolicyNet(hk.Module):
    """Diffusion policy with K output heads sharing a backbone.

    Architecture:
        backbone: obs + act + time_emb -> hidden_sizes MLP -> features
        head_k:   features -> act_dim   (for k = 0..num_heads-1)
        prior_k:  stop_grad(features) -> act_dim  (fixed random, never trained)
        output_k: head_k + prior_scale * stop_grad(prior_k)

    Randomized priors (Osband et al. 2018) maintain head diversity even when
    trainable heads converge due to shared backbone and overlapping data.

    When num_heads=1, this is architecturally equivalent to DACERPolicyNet.
    """
    hidden_sizes: Sequence[int]
    activation: Activation
    num_heads: int = 5
    prior_scale: float = 1.0
    time_dim: int = 16
    name: str = None

    def __call__(self, obs: jax.Array, act: jax.Array, t: jax.Array, head_idx = None) -> jax.Array:
        act_dim = act.shape[-1]

        # Time embedding (same as DACERPolicyNet)
        te = scaled_sinusoidal_encoding(t, dim=self.time_dim, batch_shape=obs.shape[:-1])
        te = hk.Linear(self.time_dim * 2)(te)
        te = self.activation(te)
        te = hk.Linear(self.time_dim)(te)

        # Shared backbone
        x = jnp.concatenate((obs, act, te), axis=-1)
        for i, hidden_size in enumerate(self.hidden_sizes):
            x = hk.Linear(hidden_size, name=f"backbone_linear_{i}")(x)
            x = self.activation(x)
        features = x  # shared features

        # Always compute all heads (so Haiku names are concrete Python strings)
        all_outputs = []
        for k in range(self.num_heads):
            trainable_k = hk.Linear(act_dim, name=f"head_{k}")(features)
            if self.num_heads > 1 and self.prior_scale > 0:
                # Randomized prior: fixed random network, never updated
                prior_k = hk.Linear(act_dim, name=f"prior_{k}")(
                    jax.lax.stop_gradient(features))
                v_k = trainable_k + self.prior_scale * jax.lax.stop_gradient(prior_k)
            else:
                v_k = trainable_k
            all_outputs.append(v_k)
        all_v = jnp.stack(all_outputs, axis=0)  # (K, *batch, act_dim)

        if head_idx is None:
            return all_v
        else:
            # Select one head — works with both concrete and traced head_idx
            return all_v[head_idx]


@dataclass
class IndependentBootflowPolicyNet(hk.Module):
    """K fully independent policy networks (no shared backbone).

    Each head k has its own time embedding, backbone MLP, and output layer.
    This enables genuine head diversity through independent training —
    Thompson sampling can then provide real exploration benefit.

    Same __call__ interface as BootflowPolicyNet for drop-in replacement.
    ~K× parameters and compute compared to shared backbone.
    """
    hidden_sizes: Sequence[int]
    activation: Activation
    num_heads: int = 5
    time_dim: int = 16
    name: str = None

    def __call__(self, obs: jax.Array, act: jax.Array, t: jax.Array, head_idx = None) -> jax.Array:
        act_dim = act.shape[-1]

        all_outputs = []
        for k in range(self.num_heads):
            # Independent time embedding for head k
            te = scaled_sinusoidal_encoding(t, dim=self.time_dim, batch_shape=obs.shape[:-1])
            te = hk.Linear(self.time_dim * 2, name=f"head_{k}_te_1")(te)
            te = self.activation(te)
            te = hk.Linear(self.time_dim, name=f"head_{k}_te_2")(te)

            # Independent backbone for head k
            x = jnp.concatenate((obs, act, te), axis=-1)
            for i, hidden_size in enumerate(self.hidden_sizes):
                x = hk.Linear(hidden_size, name=f"head_{k}_backbone_{i}")(x)
                x = self.activation(x)

            # Independent output for head k
            v_k = hk.Linear(act_dim, name=f"head_{k}_output")(x)
            all_outputs.append(v_k)

        all_v = jnp.stack(all_outputs, axis=0)  # (K, *batch, act_dim)

        if head_idx is None:
            return all_v
        else:
            return all_v[head_idx]


@dataclass
class BootflowNet:
    """Unified BootFlow network.

    Supports:
    - get_action: random head per step (for random_per_step exploration)
    - get_action_with_head: specific head (for Thompson sampling)
    - get_deterministic_action: head 0, no noise (for evaluation)
    - get_all_head_actions: all K heads (for disagreement computation)

    When num_heads=1, all methods use head 0 — equivalent to vanilla Diffv2Net.
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

    def _sample_action(self, key: jax.Array, policy_params_raw, log_alpha,
                       q1_params, q2_params, obs, head_idx) -> jax.Array:
        """Core action sampling logic shared by all get_action variants."""
        def model_fn(t, x):
            return self.policy(policy_params_raw, obs, x, t, head_idx)

        def sample(key: jax.Array) -> Tuple[jax.Array, jax.Array]:
            act = self.diffusion.p_sample(key, model_fn, (*obs.shape[:-1], self.act_dim))
            q1 = self.q(q1_params, obs, act)
            q2 = self.q(q2_params, obs, act)
            q = jnp.minimum(q1, q2)
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

    def _sample_action_ucb(self, key: jax.Array, policy_params_raw, log_alpha,
                           q1_params, q2_params, obs, ucb_beta: float) -> jax.Array:
        """UCB action selection: pool candidates from ALL K heads, score with Q + β×disagreement."""
        key, noise_key = jax.random.split(key)

        # Generate candidates from each head
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
            acts_k, qs_k = jax.vmap(sample_k)(head_keys)  # (N, *obs_batch, act_dim), (N, *obs_batch)
            all_acts_list.append(acts_k)
            all_qs_list.append(qs_k)

        # Pool K×N candidates
        pooled_acts = jnp.concatenate(all_acts_list, axis=0)  # (K*N, *obs_batch, act_dim)
        pooled_qs = jnp.concatenate(all_qs_list, axis=0)      # (K*N, *obs_batch)

        if ucb_beta > 0 and self.num_heads > 1:
            # Disagreement bonus: run each candidate through all K heads at t=0
            # and measure std of noise predictions
            t_zero = jnp.zeros(pooled_acts.shape[:-1], dtype=jnp.int32)
            obs_expanded = jnp.broadcast_to(obs, pooled_acts.shape[:-1] + obs.shape[-1:])
            head_preds = jnp.stack([
                self.policy(policy_params_raw, obs_expanded, pooled_acts, t_zero, k)
                for k in range(self.num_heads)
            ], axis=0)  # (K, K*N, *obs_batch, act_dim)
            disagreement = jnp.mean(jnp.std(head_preds, axis=0), axis=-1)  # (K*N, *obs_batch)
            scores = pooled_qs + ucb_beta * disagreement
        else:
            scores = pooled_qs

        # Select best candidate
        best_idx = jnp.argmax(scores, axis=0, keepdims=True)
        act = jnp.take_along_axis(pooled_acts, best_idx[..., None], axis=0).squeeze(axis=0)

        # Add exploration noise
        act = act + jax.random.normal(noise_key, act.shape) * jnp.exp(log_alpha) * self.noise_scale
        return act

    def get_action_ucb(self, key: jax.Array, policy_params: hk.Params,
                       obs: jax.Array, ucb_beta: float) -> jax.Array:
        """UCB action selection using all K heads."""
        policy_params_raw, log_alpha, q1_params, q2_params = policy_params
        return self._sample_action_ucb(key, policy_params_raw, log_alpha,
                                        q1_params, q2_params, obs, ucb_beta)

    def get_action(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        """Sample action using head 0 (used inside JIT for Q-targets and policy loss anchor).

        Thompson sampling head selection happens in BootflowSDAC.get_action()
        during data collection, not here. Using head 0 consistently makes
        value estimation stable (identical to vanilla SDAC behavior).
        """
        policy_params_raw, log_alpha, q1_params, q2_params = policy_params
        return self._sample_action(key, policy_params_raw, log_alpha, q1_params, q2_params, obs, head_idx=0)

    def get_action_with_head(self, key: jax.Array, policy_params: hk.Params,
                             obs: jax.Array, head_idx: int) -> jax.Array:
        """Sample action using a specific head (for Thompson sampling)."""
        policy_params_raw, log_alpha, q1_params, q2_params = policy_params
        return self._sample_action(key, policy_params_raw, log_alpha, q1_params, q2_params, obs, head_idx)

    def get_deterministic_action(self, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        """Deterministic action: pool candidates from ALL K heads, score by min(Q1, Q2), no noise."""
        key = random_key_from_data(obs)
        policy_params_raw, log_alpha, q1_params, q2_params = policy_params
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
        q1 = self.q(q1_params, obs_expanded, pooled_acts)
        q2 = self.q(q2_params, obs_expanded, pooled_acts)
        scores = jnp.minimum(q1, q2)

        best_idx = jnp.argmax(scores, axis=0, keepdims=True)
        act = jnp.take_along_axis(pooled_acts, best_idx[..., None], axis=0).squeeze(axis=0)
        return act

    def get_deterministic_action_head0(self, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        """Deterministic action using head 0 only (original behavior, for comparison)."""
        key = random_key_from_data(obs)
        policy_params_raw, log_alpha, q1_params, q2_params = policy_params
        log_alpha = -jnp.inf
        return self._sample_action(key, policy_params_raw, log_alpha, q1_params, q2_params, obs, head_idx=0)

    def get_all_head_actions(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        """Get actions from all K heads for uncertainty/disagreement estimation.
        Returns: (K, *obs_batch, act_dim)
        """
        keys = jax.random.split(key, self.num_heads)
        actions = []
        for k in range(self.num_heads):
            act = self.get_action_with_head(keys[k], policy_params, obs, head_idx=k)
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


def create_bootflow_net(
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
    independent_backbones: bool = False,
    flow_matching: bool = False,
) -> Tuple[BootflowNet, BootflowParams]:

    q = hk.without_apply_rng(hk.transform(
        lambda obs, act: QNet(hidden_sizes, activation)(obs, act)))

    if independent_backbones and num_heads > 1:
        policy = hk.without_apply_rng(hk.transform(
            lambda obs, act, t, head_idx: IndependentBootflowPolicyNet(
                diffusion_hidden_sizes, activation, num_heads=num_heads
            )(obs, act, t, head_idx)))
    else:
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
        # Init policy with head_idx=None so all heads get initialized
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

    net = BootflowNet(
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
