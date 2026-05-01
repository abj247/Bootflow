"""
BootFlow SDAC: Unified algorithm supporting:
- Multi-head velocity network (K heads, K=1 reduces to vanilla SDAC)
- Thompson Sampling (per-episode head selection) or random per-step
- Bootstrap masks: stored in buffer or generated on-the-fly
- Head disagreement metric for epistemic uncertainty logging
"""

from typing import NamedTuple, Tuple

import jax, jax.numpy as jnp
import numpy as np
import optax
import haiku as hk
import pickle
import os
from pathlib import Path

from relax.algorithm.base import Algorithm
from relax.network.diffv2_bootflow import BootflowNet, BootflowParams
from relax.utils.experience import Experience
from relax.utils.typing_utils import Metric
from relax.utils.persistence import make_persist


class BootflowOptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    policy: optax.OptState
    log_alpha: optax.OptState


class BootflowTrainState(NamedTuple):
    params: BootflowParams
    opt_state: BootflowOptStates
    step: int
    entropy: float
    running_mean: float
    running_std: float


class BootflowSDAC(Algorithm):

    def __init__(
        self,
        agent: BootflowNet,
        params: BootflowParams,
        *,
        gamma: float = 0.99,
        lr: float = 1e-4,
        alpha_lr: float = 3e-2,
        lr_schedule_end: float = 5e-5,
        tau: float = 0.005,
        delay_alpha_update: int = 250,
        delay_update: int = 2,
        reward_scale: float = 0.2,
        num_samples: int = 200,
        bootstrap_prob: float = 0.8,
        exploration: str = 'thompson',   # 'thompson' or 'random_per_step'
        mask_mode: str = 'random',       # 'random' or 'stored'
        disagree_coeff: float = 0.0,     # 0.0 = original BootFlow, >0 = disagreement exploration bonus
        diversity_coeff: float = 0.0,    # 0.0 = off, >0 = diversity loss pushing heads apart (MED-RL)
        num_envs: int = 1,               # number of vectorized envs for per-env Thompson sampling
        ucb_beta: float = 1.0,           # UCB exploration bonus scale (only used when exploration='ucb')
        timestep_disagree: bool = False,
        adaptive_mask: bool = False,
        ot_diversity_coeff: float = 0.0,
    ):
        self.agent = agent
        self.gamma = gamma
        self.tau = tau
        self.delay_alpha_update = delay_alpha_update
        self.delay_update = delay_update
        self.reward_scale = reward_scale
        self.num_samples = num_samples
        self.bootstrap_prob = bootstrap_prob
        self.exploration = exploration
        self.mask_mode = mask_mode
        self.disagree_coeff = disagree_coeff
        self.diversity_coeff = diversity_coeff
        self.num_envs = num_envs
        self.ucb_beta = ucb_beta
        self.timestep_disagree = timestep_disagree
        self.adaptive_mask = adaptive_mask
        self.ot_diversity_coeff = ot_diversity_coeff

        # Per-env Thompson sampling state (Python-level, not JIT state)
        self.active_heads = np.zeros(num_envs, dtype=int)

        self.optim = optax.adam(lr)
        lr_schedule = optax.schedules.linear_schedule(
            init_value=lr,
            end_value=lr_schedule_end,
            transition_steps=int(5e4),
            transition_begin=int(2.5e4),
        )
        self.policy_optim = optax.adam(learning_rate=lr_schedule)
        self.alpha_optim = optax.adam(alpha_lr)

        self.state = BootflowTrainState(
            params=params,
            opt_state=BootflowOptStates(
                q1=self.optim.init(params.q1),
                q2=self.optim.init(params.q2),
                policy=self.policy_optim.init(params.policy),
                log_alpha=self.alpha_optim.init(params.log_alpha),
            ),
            step=jnp.int32(0),
            entropy=jnp.float32(0.0),
            running_mean=jnp.float32(0.0),
            running_std=jnp.float32(1.0),
        )

        # Pre-compile the JIT-ed update for the specific num_heads
        num_heads = agent.num_heads

        @jax.jit
        def stateless_update(
            key: jax.Array, state: BootflowTrainState, data
        ) -> Tuple[BootflowTrainState, Metric]:
            obs, action, reward, next_obs, done = data.obs, data.action, data.reward, data.next_obs, data.done
            q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params, log_alpha = state.params
            q1_opt_state, q2_opt_state, policy_opt_state, log_alpha_opt_state = state.opt_state
            step = state.step
            running_mean = state.running_mean
            running_std = state.running_std

            # Match SDAC's 7-way key split so first keys are identical for K=1
            (next_eval_key, new_eval_key, bootstrap_key, disagree_key,
             _unused_key, diffusion_time_key, diffusion_noise_key) = jax.random.split(key, 7)

            reward *= self.reward_scale

            # --- Disagreement exploration bonus (Pathak et al. ICML 2019) ---
            # When disagree_coeff > 0, add head disagreement as intrinsic reward.
            # This incentivizes visiting states where heads disagree (high uncertainty).
            # disagree_coeff=0.0 disables this (original BootFlow behavior).
            disagree_bonus = jnp.float32(0.0)
            if num_heads > 1 and self.disagree_coeff > 0:
                bonus_t = jnp.zeros(obs.shape[0], dtype=jnp.int32)
                bonus_noise = jax.random.normal(disagree_key, action.shape)
                bonus_xt = jax.vmap(self.agent.diffusion.q_sample)(bonus_t, action, bonus_noise)
                bonus_preds = []
                for k in range(num_heads):
                    pred_k = self.agent.policy(policy_params, obs, bonus_xt, bonus_t, k)
                    bonus_preds.append(pred_k)
                bonus_preds_stack = jnp.stack(bonus_preds, axis=0)  # (K, batch, act_dim)
                disagree_bonus = jnp.mean(jnp.std(bonus_preds_stack, axis=0), axis=-1)  # (batch,)
                reward = reward + self.disagree_coeff * disagree_bonus

            def get_min_q(s, a):
                q1 = self.agent.q(q1_params, s, a)
                q2 = self.agent.q(q2_params, s, a)
                return jnp.minimum(q1, q2)

            # --- Compute target Q ---
            next_action = self.agent.get_action(
                next_eval_key,
                (policy_params, log_alpha, q1_params, q2_params),
                next_obs)
            q1_target = self.agent.q(target_q1_params, next_obs, next_action)
            q2_target = self.agent.q(target_q2_params, next_obs, next_action)
            q_target = jnp.minimum(q1_target, q2_target)
            q_backup = reward + (1 - done) * self.gamma * q_target

            # --- Q-function losses ---
            def q_loss_fn(q_params: hk.Params) -> jax.Array:
                q = self.agent.q(q_params, obs, action)
                q_loss = jnp.mean((q - q_backup) ** 2)
                return q_loss, q

            (q1_loss, q1), q1_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q1_params)
            (q2_loss, q2), q2_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q2_params)

            # --- Policy loss ---
            new_action = self.agent.get_action(
                new_eval_key,
                (policy_params, log_alpha, q1_params, q2_params),
                obs)

            diff_key1, diff_key2 = jax.random.split(diffusion_noise_key, 2)
            t = jax.random.randint(diffusion_time_key, (next_obs.shape[0],), 0, self.agent.num_timesteps)
            noise1 = jax.random.normal(diff_key1, action.shape)
            tilde_at = jax.vmap(self.agent.diffusion.q_sample)(t, new_action, noise1)

            reverse_mc_num = 64
            tilde_at = jnp.repeat(tilde_at, reverse_mc_num, axis=0)
            t = jnp.repeat(t, reverse_mc_num, axis=0)
            wide_obs = jnp.repeat(obs, reverse_mc_num, axis=0)
            wide_new_action = jnp.repeat(new_action, reverse_mc_num, axis=0)

            batch_size = obs.shape[0]

            # Bootstrap mask: either from data, adaptive (Idea 2), or fixed
            adaptive_p = None
            if self.mask_mode == 'stored' and hasattr(data, 'bootstrap_mask'):
                bootstrap_mask = data.bootstrap_mask.T
            elif self.adaptive_mask and num_heads > 1:
                # Idea 2: Adaptive mask using twin-Q disagreement
                q1_vals = self.agent.q(q1_params, obs, action)
                q2_vals = self.agent.q(q2_params, obs, action)
                q_disagree = jnp.abs(q1_vals - q2_vals)
                q_disagree_norm = q_disagree / (jnp.max(q_disagree) + 1e-8)
                p_min = 0.5
                adaptive_p = self.bootstrap_prob - (self.bootstrap_prob - p_min) * q_disagree_norm
                bootstrap_mask = jax.random.bernoulli(
                    bootstrap_key, p=adaptive_p[None, :],
                    shape=(num_heads, batch_size)
                ).astype(jnp.float32)
                any_selected = bootstrap_mask.sum(axis=0) > 0
                bootstrap_mask = bootstrap_mask.at[0].set(
                    jnp.where(any_selected, bootstrap_mask[0], 1.0))
            else:
                bootstrap_mask = jax.random.bernoulli(
                    bootstrap_key,
                    p=self.bootstrap_prob,
                    shape=(num_heads, batch_size)
                ).astype(jnp.float32)
                any_selected = bootstrap_mask.sum(axis=0) > 0
                bootstrap_mask = bootstrap_mask.at[0].set(
                    jnp.where(any_selected, bootstrap_mask[0], 1.0)
                )

            def policy_loss_fn(policy_params) -> jax.Array:
                total_loss = jnp.float32(0.0)
                per_head_losses = []

                # Shared noise, recon, and Q-weights for all heads — avoids
                # gradient interference in shared backbone from conflicting targets
                noise2 = jax.random.normal(
                    diff_key2,
                    (action.shape[0] * reverse_mc_num, action.shape[1]))
                recon = self.agent.diffusion.get_recon(t, tilde_at, noise2).clip(-1, 1)
                q_min = get_min_q(wide_obs, recon) * 5.0 / jnp.exp(log_alpha)
                q_reshape = q_min.reshape((-1, reverse_mc_num))
                Z = jax.nn.logsumexp(q_reshape, axis=1, keepdims=True)
                q_weights = jnp.exp(q_reshape - Z).flatten()

                for k in range(num_heads):
                    def denoiser_k(t_val, x_val):
                        return self.agent.policy(policy_params, wide_obs, x_val, t_val, k)

                    head_loss = self.agent.diffusion.reverse_samping_weighted_p_loss(
                        noise2, q_weights, denoiser_k, t, tilde_at, x_start=wide_new_action)

                    per_head_losses.append(head_loss)

                    # Apply bootstrap mask for this head
                    mask_ratio = jnp.mean(bootstrap_mask[k])
                    total_loss = total_loss + head_loss * mask_ratio

                # Normalize by K so backbone gradient magnitude matches K=1.
                # Without this, K=5 gets ~4x larger gradients through the shared
                # backbone, destabilizing learning on harder tasks (e.g. Ant-v4).
                if num_heads > 1:
                    total_loss = total_loss / num_heads

                # --- Diversity loss: push heads apart (MED-RL, ICLR 2022) ---
                # Maximize std of head predictions = minimize negative std.
                # This directly counteracts head convergence in every gradient step.
                diversity_loss = jnp.float32(0.0)
                if num_heads > 1 and self.diversity_coeff > 0:
                    div_preds = []
                    for k in range(num_heads):
                        pred_k = self.agent.policy(
                            policy_params, wide_obs, noise2, t, k)
                        div_preds.append(pred_k)
                    div_preds_stack = jnp.stack(div_preds, axis=0)  # (K, batch*mc, act_dim)
                    # Negative mean std = minimize → maximize disagreement
                    diversity_loss = -self.diversity_coeff * jnp.mean(
                        jnp.std(div_preds_stack, axis=0))
                    total_loss = total_loss + diversity_loss

                # --- OT Diversity Loss (Idea 3) ---
                ot_div_loss = jnp.float32(0.0)
                ot_mean_dist = jnp.float32(0.0)
                if num_heads > 1 and self.ot_diversity_coeff > 0:
                    div_preds_ot = []
                    for k in range(num_heads):
                        pred_k = self.agent.policy(policy_params, wide_obs, noise2, t, k)
                        div_preds_ot.append(pred_k)
                    ot_act_dim = action.shape[-1]
                    ot_stack = jnp.stack(div_preds_ot, axis=0)
                    ot_reshaped = ot_stack.reshape(num_heads, -1, reverse_mc_num, ot_act_dim)
                    total_dist = jnp.float32(0.0)
                    n_pairs = 0
                    for k1 in range(num_heads):
                        for k2 in range(k1 + 1, num_heads):
                            diff = ot_reshaped[k1][:, :, None, :] - ot_reshaped[k2][:, None, :, :]
                            cost = jnp.sum(diff ** 2, axis=-1)
                            min_cost = jnp.mean(jnp.min(cost, axis=-1))
                            total_dist = total_dist + min_cost
                            n_pairs += 1
                    ot_mean_dist = total_dist / n_pairs
                    ot_div_loss = -self.ot_diversity_coeff * ot_mean_dist
                    total_loss = total_loss + ot_div_loss

                q_mean = q_min.mean()
                q_std = q_min.std()
                head_losses_stack = jnp.stack(per_head_losses)

                return total_loss, (q_weights, q_min, q_mean, q_std, head_losses_stack, diversity_loss, ot_div_loss, ot_mean_dist)

            (total_loss, (q_weights, scaled_q, q_mean, q_std, head_losses_stack, div_loss, ot_div_loss, ot_mean_dist)), policy_grads = jax.value_and_grad(
                policy_loss_fn, has_aux=True)(policy_params)

            # --- Alpha loss ---
            def log_alpha_loss_fn(log_alpha: jax.Array) -> jax.Array:
                approx_entropy = 0.5 * self.agent.act_dim * jnp.log(
                    2 * jnp.pi * jnp.exp(1) * (0.1 * jnp.exp(log_alpha)) ** 2)
                return -1 * log_alpha * (-1 * jax.lax.stop_gradient(approx_entropy) + self.agent.target_entropy)

            # --- Parameter updates ---
            def param_update(optim, params, grads, opt_state):
                update, new_opt_state = optim.update(grads, opt_state)
                new_params = optax.apply_updates(params, update)
                return new_params, new_opt_state

            def delay_param_update(optim, params, grads, opt_state):
                return jax.lax.cond(
                    step % self.delay_update == 0,
                    lambda params, opt_state: param_update(optim, params, grads, opt_state),
                    lambda params, opt_state: (params, opt_state),
                    params, opt_state)

            def delay_alpha_param_update(optim, params, opt_state):
                return jax.lax.cond(
                    step % self.delay_alpha_update == 0,
                    lambda params, opt_state: param_update(
                        optim, params, jax.grad(log_alpha_loss_fn)(params), opt_state),
                    lambda params, opt_state: (params, opt_state),
                    params, opt_state)

            def delay_target_update(params, target_params, tau):
                return jax.lax.cond(
                    step % self.delay_update == 0,
                    lambda target_params: optax.incremental_update(params, target_params, tau),
                    lambda target_params: target_params,
                    target_params)

            q1_params, q1_opt_state = param_update(self.optim, q1_params, q1_grads, q1_opt_state)
            q2_params, q2_opt_state = param_update(self.optim, q2_params, q2_grads, q2_opt_state)
            policy_params, policy_opt_state = delay_param_update(
                self.policy_optim, policy_params, policy_grads, policy_opt_state)
            log_alpha, log_alpha_opt_state = delay_alpha_param_update(
                self.alpha_optim, log_alpha, log_alpha_opt_state)

            target_q1_params = delay_target_update(q1_params, target_q1_params, self.tau)
            target_q2_params = delay_target_update(q2_params, target_q2_params, self.tau)
            target_policy_params = delay_target_update(policy_params, target_policy_params, self.tau)

            new_running_mean = running_mean + 0.001 * (q_mean - running_mean)
            new_running_std = running_std + 0.001 * (q_std - running_std)

            # --- Head disagreement metrics ---
            act_dim = action.shape[-1]
            strategic_disagree = jnp.float32(0.0)
            execution_disagree = jnp.float32(0.0)
            strategic_ratio = jnp.float32(1.0)
            if num_heads > 1:
                disagree_noise = jax.random.normal(disagree_key, action.shape)

                if self.timestep_disagree:
                    # Idea 1: Timestep-stratified disagreement
                    T = self.agent.num_timesteps
                    timestep_levels = [0, T // 4, T // 2, 3 * T // 4]
                    per_level_disagree = []
                    for t_level in timestep_levels:
                        disagree_t = jnp.full(obs.shape[0], t_level, dtype=jnp.int32)
                        disagree_xt = jax.vmap(self.agent.diffusion.q_sample)(disagree_t, action, disagree_noise)
                        head_preds = []
                        for k in range(num_heads):
                            pred_k = self.agent.policy(policy_params, obs, disagree_xt, disagree_t, k)
                            head_preds.append(pred_k)
                        head_preds_stack = jnp.stack(head_preds, axis=0)
                        level_disagree = jnp.mean(jnp.std(head_preds_stack, axis=0), axis=-1)
                        per_level_disagree.append(jnp.mean(level_disagree))
                    strategic_disagree = per_level_disagree[-1]
                    execution_disagree = per_level_disagree[0]
                    strategic_ratio = strategic_disagree / (execution_disagree + 1e-8)
                    head_disagreement = jnp.mean(jnp.array(per_level_disagree))
                    per_sample_disagree = jnp.zeros(obs.shape[0])  # placeholder
                else:
                    disagree_t = jnp.zeros(obs.shape[0], dtype=jnp.int32)
                    disagree_xt = jax.vmap(self.agent.diffusion.q_sample)(disagree_t, action, disagree_noise)
                    head_preds = []
                    for k in range(num_heads):
                        pred_k = self.agent.policy(policy_params, obs, disagree_xt, disagree_t, k)
                        head_preds.append(pred_k)
                    head_preds_stack = jnp.stack(head_preds, axis=0)
                    per_sample_disagree = jnp.mean(jnp.std(head_preds_stack, axis=0), axis=-1)
                    head_disagreement = jnp.mean(per_sample_disagree)

                head_disagreement_max = jnp.max(per_sample_disagree) if not self.timestep_disagree else strategic_disagree
                head_disagreement_min = jnp.min(per_sample_disagree) if not self.timestep_disagree else execution_disagree
                per_dim_disagree = jnp.zeros(act_dim)  # simplified
            else:
                head_disagreement = jnp.float32(0.0)
                head_disagreement_max = jnp.float32(0.0)
                head_disagreement_min = jnp.float32(0.0)
                per_dim_disagree = jnp.zeros(act_dim)
                per_sample_disagree = jnp.zeros(obs.shape[0])

            state = BootflowTrainState(
                params=BootflowParams(
                    q1_params, q2_params, target_q1_params, target_q2_params,
                    policy_params, target_policy_params, log_alpha),
                opt_state=BootflowOptStates(
                    q1=q1_opt_state, q2=q2_opt_state,
                    policy=policy_opt_state, log_alpha=log_alpha_opt_state),
                step=step + 1,
                entropy=jnp.float32(0.0),
                running_mean=new_running_mean,
                running_std=new_running_std,
            )
            info = {
                "q1_loss": q1_loss,
                "q1_mean": jnp.mean(q1),
                "q1_max": jnp.max(q1),
                "q1_min": jnp.min(q1),
                "q2_loss": q2_loss,
                "policy_loss": total_loss,
                "alpha": jnp.exp(log_alpha),
                "q_weights_std": jnp.std(q_weights),
                "q_weights_mean": jnp.mean(q_weights),
                "q_weights_min": jnp.min(q_weights),
                "q_weights_max": jnp.max(q_weights),
                "hist_q_weights": q_weights,
                "hist_t": t,
                "scale_q_mean": jnp.mean(scaled_q),
                "scale_q_std": jnp.std(scaled_q),
                "running_q_mean": new_running_mean,
                "running_q_std": new_running_std,
                "entropy_approx": 0.5 * self.agent.act_dim * jnp.log(
                    2 * jnp.pi * jnp.exp(1) * (0.1 * jnp.exp(log_alpha)) ** 2),
                "head_disagreement": head_disagreement,
                "disagree_bonus_mean": jnp.mean(disagree_bonus) if num_heads > 1 else jnp.float32(0.0),
                "diversity_loss": div_loss,
                # --- New analysis metrics ---
                "head_disagreement_max": head_disagreement_max,
                "head_disagreement_min": head_disagreement_min,
                "q1_q2_gap": jnp.mean(jnp.abs(q1 - q2)),
                "q_target_mean": jnp.mean(q_backup),
                "hist_head_disagreement": per_sample_disagree,
            }
            # Per-action-dim disagreement
            for i in range(act_dim):
                info[f"head_disagree_dim_{i}"] = per_dim_disagree[i]
            # Per-head policy losses
            for k in range(num_heads):
                info[f"head_loss_{k}"] = head_losses_stack[k]
            # Per-head bootstrap mask ratios
            for k in range(num_heads):
                info[f"head_mask_ratio_{k}"] = jnp.mean(bootstrap_mask[k])
            # Idea 1: Timestep-stratified disagreement metrics
            if self.timestep_disagree:
                info["strategic_disagree"] = strategic_disagree
                info["execution_disagree"] = execution_disagree
                info["strategic_ratio"] = strategic_ratio
            # Idea 2: Adaptive mask metrics
            if self.adaptive_mask and adaptive_p is not None:
                info["adaptive_mask_p_mean"] = jnp.mean(adaptive_p)
                info["adaptive_mask_p_std"] = jnp.std(adaptive_p)
            # Idea 3: OT diversity metrics
            if self.ot_diversity_coeff > 0:
                info["ot_diversity_loss"] = ot_div_loss
                info["ot_mean_distance"] = ot_mean_dist
            return state, info

        # Wire up common behavior
        # For Thompson mode, we JIT a separate get_action per head to avoid
        # tracing head_idx (Haiku needs concrete ints for module names).
        # We store K JIT-compiled functions and dispatch by self.active_head.
        if exploration == 'thompson' and num_heads > 1:
            self._head_action_fns = []
            for k in range(num_heads):
                def _make_fn(head_k):
                    def fn(key, params, obs):
                        return self.agent.get_action_with_head(key, params, obs, head_k)
                    return fn
                self._head_action_fns.append(jax.jit(_make_fn(k)))

            self._implement_common_behavior(
                stateless_update,
                self.agent.get_action,  # fallback, overridden by get_action below
                self.agent.get_deterministic_action,
                stateless_get_value=self.agent.q)
        elif exploration == 'ucb' and num_heads > 1:
            # UCB: JIT a single function that uses all K heads
            _ucb_beta = self.ucb_beta
            def _ucb_action_fn(key, params, obs):
                return self.agent.get_action_ucb(key, params, obs, _ucb_beta)
            self._ucb_action_fn = jax.jit(_ucb_action_fn)

            self._implement_common_behavior(
                stateless_update,
                self.agent.get_action,  # for Q-targets (uses head 0)
                self.agent.get_deterministic_action,
                stateless_get_value=self.agent.q)
        else:
            self._implement_common_behavior(
                stateless_update,
                self.agent.get_action,
                self.agent.get_deterministic_action,
                stateless_get_value=self.agent.q)

    # --- Thompson Sampling ---

    def select_new_head(self, key, done_mask=None):
        """Called by trainer at episode boundaries for Thompson sampling.

        Args:
            key: PRNG key.
            done_mask: Optional bool array (num_envs,) — which envs just finished.
                       None means reset all heads (used at initialization).
        """
        if self.exploration == 'thompson' and self.agent.num_heads > 1:
            new_heads = np.array(
                jax.random.randint(key, (self.num_envs,), 0, self.agent.num_heads)
            )
            if done_mask is None:
                self.active_heads = new_heads
            else:
                self.active_heads = np.where(done_mask, new_heads, self.active_heads)

    def get_action(self, key: jax.Array, obs: np.ndarray) -> np.ndarray:
        if self.exploration == 'thompson' and self.agent.num_heads > 1:
            # Per-env head selection: run all K heads, select per-env
            params = self.get_policy_params_to_save()
            unique_heads = np.unique(self.active_heads)
            if len(unique_heads) == 1:
                # All envs use same head — single forward pass
                action = self._head_action_fns[unique_heads[0]](key, params, obs)
            else:
                # Multiple heads active — run each unique head, assemble per-env
                all_actions = np.stack([
                    np.asarray(self._head_action_fns[k](key, params, obs))
                    for k in range(self.agent.num_heads)
                ])  # (K, num_envs, act_dim)
                action = all_actions[self.active_heads, np.arange(self.num_envs)]
        elif self.exploration == 'ucb' and self.agent.num_heads > 1:
            action = self._ucb_action_fn(key, self.get_policy_params_to_save(), obs)
        else:
            action = self._get_action(key, self.get_policy_params_to_save(), obs)
        return np.asarray(action)

    # --- Params ---

    def get_policy_params(self):
        return (self.state.params.policy, self.state.params.log_alpha,
                self.state.params.q1, self.state.params.q2)

    def get_policy_params_to_save(self):
        return (self.state.params.target_policy, self.state.params.log_alpha,
                self.state.params.q1, self.state.params.q2)

    def save_policy(self, path: str) -> None:
        policy = jax.device_get(self.get_policy_params_to_save())
        with open(path, "wb") as f:
            pickle.dump(policy, f)

    def save_q_structure(self, root: os.PathLike, dummy_obs: jax.Array, dummy_action: jax.Array) -> None:
        root = Path(root)
        deterministic = make_persist(self._get_value._fun)(
            self.get_value_params()[0], dummy_obs, dummy_action)
        deterministic.save(root / "q_func.pkl")
        deterministic.save_info(root / "q_func.txt")

    def get_value_params(self):
        return self.state.params.q1, self.state.params.q2

    def save_q(self, path: str) -> None:
        value = jax.device_get(self.get_value_params())
        with open(path, "wb") as f:
            pickle.dump(value, f)
