"""
FullBootFlow SDAC: K independent policy heads + K independent Q-networks.

Combines:
- BootFlow's per-head policy loss and Thompson sampling
- AdaFlow's K Q-network ensemble with REDQ-style targets

Key innovation: each policy head k trains against its OWN Q_k's targets,
enabling genuine posterior diversity for Thompson sampling.

Features:
- K independent Q-networks with REDQ target (sample M from K, take min)
- K independent policy heads, each using its own Q_k for q_weights
- Thompson Sampling: per-env (head_k, Q_k) pair selection
- UCB: pool K*N candidates, score by mean(Q_all) + beta*std(Q_all)
- Bootstrap masking per head
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
from relax.network.diffv2_fullbootflow import FullBootflowNet, FullBootflowParams
from relax.utils.experience import Experience
from relax.utils.typing_utils import Metric
from relax.utils.persistence import make_persist


class FullBootflowOptStates(NamedTuple):
    q_opt_states: tuple      # K optax.OptState
    policy: optax.OptState
    log_alpha: optax.OptState


class FullBootflowTrainState(NamedTuple):
    params: FullBootflowParams
    opt_state: FullBootflowOptStates
    step: int
    entropy: float
    running_mean: float
    running_std: float
    max_q_disagreement: float


class FullBootflowSDAC(Algorithm):

    def __init__(
        self,
        agent: FullBootflowNet,
        params: FullBootflowParams,
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
        redq_m: int = 2,
        bootstrap_prob: float = 0.8,
        exploration: str = 'thompson',   # 'thompson', 'ucb', or 'idfm'
        ucb_beta: float = 1.0,
        num_envs: int = 1,
        timestep_disagree: bool = False,
        adaptive_mask: bool = False,
        ot_diversity_coeff: float = 0.0,
        svgd_coeff: float = 0.0,
        idfm_beta: float = 1.0,
        idfm_mode: str = 'raw',
    ):
        self.agent = agent
        self.gamma = gamma
        self.tau = tau
        self.delay_alpha_update = delay_alpha_update
        self.delay_update = delay_update
        self.reward_scale = reward_scale
        self.num_samples = num_samples
        self.redq_m = redq_m
        self.bootstrap_prob = bootstrap_prob
        self.exploration = exploration
        self.ucb_beta = ucb_beta
        self.num_envs = num_envs
        self.timestep_disagree = timestep_disagree
        self.adaptive_mask = adaptive_mask
        self.ot_diversity_coeff = ot_diversity_coeff
        self.svgd_coeff = svgd_coeff
        self.idfm_beta = idfm_beta
        self.idfm_mode = idfm_mode

        # Per-env Thompson sampling state (Python-level, not JIT state)
        self.active_heads = np.zeros(num_envs, dtype=int)
        # EMA correlation for ema_corr IDFM mode (Python-level, updated after each action)
        self._ema_corr = 0.5  # Start neutral

        self.optim = optax.adam(lr)
        lr_schedule = optax.schedules.linear_schedule(
            init_value=lr,
            end_value=lr_schedule_end,
            transition_steps=int(5e4),
            transition_begin=int(2.5e4),
        )
        self.policy_optim = optax.adam(learning_rate=lr_schedule)
        self.alpha_optim = optax.adam(alpha_lr)

        num_heads = agent.num_heads

        self.state = FullBootflowTrainState(
            params=params,
            opt_state=FullBootflowOptStates(
                q_opt_states=tuple(self.optim.init(params.q_params[i]) for i in range(num_heads)),
                policy=self.policy_optim.init(params.policy),
                log_alpha=self.alpha_optim.init(params.log_alpha),
            ),
            step=jnp.int32(0),
            entropy=jnp.float32(0.0),
            running_mean=jnp.float32(0.0),
            running_std=jnp.float32(1.0),
            max_q_disagreement=jnp.float32(1.0),
        )

        @jax.jit
        def stateless_update(
            key: jax.Array, state: FullBootflowTrainState, data
        ) -> Tuple[FullBootflowTrainState, Metric]:
            obs, action, reward, next_obs, done = data.obs, data.action, data.reward, data.next_obs, data.done
            q_params = state.params.q_params
            target_q_params = state.params.target_q_params
            policy_params = state.params.policy
            target_policy_params = state.params.target_policy
            log_alpha = state.params.log_alpha
            q_opt_states = state.opt_state.q_opt_states
            policy_opt_state = state.opt_state.policy
            log_alpha_opt_state = state.opt_state.log_alpha
            step = state.step
            running_mean = state.running_mean
            running_std = state.running_std
            max_q_disagreement = state.max_q_disagreement

            (next_eval_key, new_eval_key, bootstrap_key,
             diffusion_time_key, diffusion_noise_key, redq_key) = jax.random.split(key, 6)

            reward *= self.reward_scale

            # --- Compute REDQ target ---
            # Use head 0 + Q_0 for next action selection (stability)
            next_action = self.agent.get_action(
                next_eval_key,
                (policy_params, log_alpha, q_params),
                next_obs)

            # Stack all K target Q-values
            all_target_qs = jnp.stack([
                self.agent.q(target_q_params[i], next_obs, next_action)
                for i in range(num_heads)
            ])  # (K, batch)

            # REDQ: sample M from K, take min
            redq_indices = jax.random.choice(redq_key, num_heads,
                                             shape=(self.redq_m,), replace=False)
            selected_qs = all_target_qs[redq_indices]  # (M, batch)
            q_target = jnp.min(selected_qs, axis=0)
            q_backup = reward + (1 - done) * self.gamma * q_target

            # --- K Q-network updates ---
            new_q_params_list = []
            new_q_opt_states_list = []
            q_losses = []
            q_values_first = None

            for i in range(num_heads):
                def q_loss_fn(qp: hk.Params) -> jax.Array:
                    q = self.agent.q(qp, obs, action)
                    q_loss = jnp.mean((q - q_backup) ** 2)
                    return q_loss, q

                (q_loss_i, q_i), q_grads_i = jax.value_and_grad(q_loss_fn, has_aux=True)(q_params[i])
                update_i, new_qos_i = self.optim.update(q_grads_i, q_opt_states[i])
                new_qp_i = optax.apply_updates(q_params[i], update_i)
                new_q_params_list.append(new_qp_i)
                new_q_opt_states_list.append(new_qos_i)
                q_losses.append(q_loss_i)
                if i == 0:
                    q_values_first = q_i

            new_q_params = tuple(new_q_params_list)
            new_q_opt_states = tuple(new_q_opt_states_list)

            # --- Q-ensemble disagreement ---
            all_qs_on_batch = jnp.stack([
                self.agent.q(q_params[i], obs, action)
                for i in range(num_heads)
            ])  # (K, batch)
            q_ensemble_std = jnp.mean(jnp.std(all_qs_on_batch, axis=0))
            new_max_q_disagreement = jnp.maximum(max_q_disagreement, q_ensemble_std)
            q_std_norm = q_ensemble_std / jnp.maximum(new_max_q_disagreement, 1e-8)

            # --- Policy loss: per-head with per-head Q_k ---
            new_action = self.agent.get_action(
                new_eval_key,
                (policy_params, log_alpha, q_params),
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

            # Bootstrap mask: adaptive (Idea 2) or fixed
            if self.adaptive_mask and num_heads > 1:
                all_q_on_batch = jnp.stack([
                    self.agent.q(q_params[i], obs, action) for i in range(num_heads)
                ])
                q_disagree = jnp.std(all_q_on_batch, axis=0)
                q_disagree_norm = q_disagree / (jnp.max(q_disagree) + 1e-8)
                p_min = 0.5
                adaptive_p = self.bootstrap_prob - (self.bootstrap_prob - p_min) * q_disagree_norm
                bootstrap_mask = jax.random.bernoulli(
                    bootstrap_key,
                    p=adaptive_p[None, :],
                    shape=(num_heads, batch_size)
                ).astype(jnp.float32)
            else:
                adaptive_p = None
                bootstrap_mask = jax.random.bernoulli(
                    bootstrap_key,
                    p=self.bootstrap_prob,
                    shape=(num_heads, batch_size)
                ).astype(jnp.float32)
            # Ensure each sample used by at least one head
            any_selected = bootstrap_mask.sum(axis=0) > 0
            bootstrap_mask = bootstrap_mask.at[0].set(
                jnp.where(any_selected, bootstrap_mask[0], 1.0)
            )

            def policy_loss_fn(policy_params) -> jax.Array:
                total_loss = jnp.float32(0.0)
                per_head_losses = []

                noise2 = jax.random.normal(
                    diff_key2,
                    (action.shape[0] * reverse_mc_num, action.shape[1]))
                recon = self.agent.diffusion.get_recon(t, tilde_at, noise2).clip(-1, 1)

                # Compute all K Q-values on recon for REDQ-style pessimism
                all_q_on_recon = jnp.stack([
                    self.agent.q(q_params[i], wide_obs, recon)
                    for i in range(num_heads)
                ])  # (K, batch*mc)

                for k in range(num_heads):
                    # REDQ pessimism: for head k, use min of Q_k + one random other Q
                    # This gives each head a slightly different perspective while
                    # maintaining twin-Q-level pessimism (min of 2)
                    other_idx = (k + 1) % num_heads  # deterministic pairing avoids extra RNG in JIT
                    q_min_k = jnp.minimum(all_q_on_recon[k], all_q_on_recon[other_idx])
                    q_min_k = q_min_k * 5.0 / jnp.exp(log_alpha)
                    q_reshape_k = q_min_k.reshape((-1, reverse_mc_num))
                    Z_k = jax.nn.logsumexp(q_reshape_k, axis=1, keepdims=True)
                    q_weights_k = jnp.exp(q_reshape_k - Z_k).flatten()

                    def _make_denoiser_k(head_k):
                        def denoiser_k(t_val, x_val):
                            return self.agent.policy(policy_params, wide_obs, x_val, t_val, head_k)
                        return denoiser_k

                    head_loss_k = self.agent.diffusion.reverse_samping_weighted_p_loss(
                        noise2, q_weights_k, _make_denoiser_k(k), t, tilde_at, x_start=wide_new_action)

                    per_head_losses.append(head_loss_k)

                    # Apply bootstrap mask for this head
                    mask_ratio_k = jnp.mean(bootstrap_mask[k])
                    total_loss = total_loss + head_loss_k * mask_ratio_k

                # Normalize by K
                if num_heads > 1:
                    total_loss = total_loss / num_heads

                # --- OT Diversity Loss (Idea 3) ---
                ot_div_loss = jnp.float32(0.0)
                ot_mean_dist = jnp.float32(0.0)
                if num_heads > 1 and self.ot_diversity_coeff > 0:
                    # Use noise predictions at current timestep as proxy for action distributions
                    div_preds = []
                    for k in range(num_heads):
                        pred_k = self.agent.policy(policy_params, wide_obs, noise2, t, k)
                        div_preds.append(pred_k)
                    # Reshape to (K, batch, mc, act_dim) for pairwise comparison
                    act_dim = action.shape[-1]
                    div_stack = jnp.stack(div_preds, axis=0)  # (K, batch*mc, act_dim)
                    div_reshaped = div_stack.reshape(num_heads, -1, reverse_mc_num, act_dim)  # (K, batch, mc, act_dim)
                    # Pairwise greedy Wasserstein between heads
                    total_dist = jnp.float32(0.0)
                    n_pairs = 0
                    for k1 in range(num_heads):
                        for k2 in range(k1 + 1, num_heads):
                            diff = div_reshaped[k1][:, :, None, :] - div_reshaped[k2][:, None, :, :]
                            cost = jnp.sum(diff ** 2, axis=-1)  # (batch, mc, mc)
                            min_cost = jnp.mean(jnp.min(cost, axis=-1))
                            total_dist = total_dist + min_cost
                            n_pairs += 1
                    ot_mean_dist = total_dist / n_pairs
                    ot_div_loss = -self.ot_diversity_coeff * ot_mean_dist
                    total_loss = total_loss + ot_div_loss

                # --- SVGD Repulsive Loss ---
                svgd_loss = jnp.float32(0.0)
                svgd_mean_dist = jnp.float32(0.0)
                if num_heads > 1 and self.svgd_coeff > 0:
                    svgd_preds = []
                    n_svgd = min(64, obs.shape[0])
                    svgd_t = jnp.zeros(n_svgd, dtype=jnp.int32)
                    svgd_noise = jax.random.normal(jax.random.fold_in(diff_key2, 777), (n_svgd, action.shape[-1]))
                    svgd_xt = jax.vmap(self.agent.diffusion.q_sample)(svgd_t, action[:n_svgd], svgd_noise)
                    for k in range(num_heads):
                        pred_k = self.agent.policy(policy_params, obs[:n_svgd], svgd_xt, svgd_t, k)
                        svgd_preds.append(pred_k)
                    svgd_stack = jnp.stack(svgd_preds, axis=0)  # (K, n_svgd, act_dim)
                    svgd_flat = svgd_stack.reshape(num_heads, -1)  # (K, n_svgd*act_dim)
                    pairwise_dists = jnp.sum((svgd_flat[:, None] - svgd_flat[None, :]) ** 2, axis=-1)
                    mask_diag = 1.0 - jnp.eye(num_heads)
                    svgd_mean_dist = jnp.sum(pairwise_dists * mask_diag) / (num_heads * (num_heads - 1))
                    svgd_loss = -self.svgd_coeff * svgd_mean_dist
                    total_loss = total_loss + svgd_loss

                # Use head 0's pessimistic Q for logging
                q_0 = jnp.minimum(all_q_on_recon[0], all_q_on_recon[1]) * 5.0 / jnp.exp(log_alpha)
                q_mean = q_0.mean()
                q_std = q_0.std()
                q_reshape_0 = q_0.reshape((-1, reverse_mc_num))
                Z_0 = jax.nn.logsumexp(q_reshape_0, axis=1, keepdims=True)
                q_weights_0 = jnp.exp(q_reshape_0 - Z_0).flatten()
                head_losses_stack = jnp.stack(per_head_losses)

                return total_loss, (q_weights_0, q_0, q_mean, q_std, head_losses_stack, ot_div_loss, ot_mean_dist, svgd_loss, svgd_mean_dist)

            (total_loss, (q_weights, scaled_q, q_mean, q_std, head_losses_stack, ot_div_loss, ot_mean_dist, svgd_loss, svgd_mean_dist)), policy_grads = jax.value_and_grad(
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

            # Update policy and alpha
            policy_params, policy_opt_state = delay_param_update(
                self.policy_optim, policy_params, policy_grads, policy_opt_state)
            log_alpha, log_alpha_opt_state = delay_alpha_param_update(
                self.alpha_optim, log_alpha, log_alpha_opt_state)

            # Target updates for all K Q-networks and policy
            new_target_q_params = tuple(
                delay_target_update(new_q_params[i], target_q_params[i], self.tau)
                for i in range(num_heads)
            )
            target_policy_params = delay_target_update(policy_params, target_policy_params, self.tau)

            new_running_mean = running_mean + 0.001 * (q_mean - running_mean)
            new_running_std = running_std + 0.001 * (q_std - running_std)

            # --- Head disagreement metrics ---
            strategic_disagree = jnp.float32(0.0)
            execution_disagree = jnp.float32(0.0)
            strategic_ratio = jnp.float32(1.0)
            if num_heads > 1:
                disagree_key = jax.random.fold_in(key, 999)
                disagree_noise = jax.random.normal(disagree_key, action.shape)

                if self.timestep_disagree:
                    # Idea 1: Compute disagreement at multiple timesteps
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
            else:
                head_disagreement = jnp.float32(0.0)

            state = FullBootflowTrainState(
                params=FullBootflowParams(
                    new_q_params, new_target_q_params,
                    policy_params, target_policy_params, log_alpha),
                opt_state=FullBootflowOptStates(
                    q_opt_states=new_q_opt_states,
                    policy=policy_opt_state,
                    log_alpha=log_alpha_opt_state),
                step=step + 1,
                entropy=jnp.float32(0.0),
                running_mean=new_running_mean,
                running_std=new_running_std,
                max_q_disagreement=new_max_q_disagreement,
            )

            # --- Per-Q-network value stats (track Q-network divergence) ---
            all_q_values_on_batch = jnp.stack([
                self.agent.q(new_q_params[i], obs, action)
                for i in range(num_heads)
            ])  # (K, batch)
            per_q_means = jnp.mean(all_q_values_on_batch, axis=1)  # (K,) mean per Q-net
            q_network_divergence = jnp.std(per_q_means)  # how much Q-networks disagree on avg value
            q_network_max_spread = jnp.max(per_q_means) - jnp.min(per_q_means)  # max-min across Q-nets

            q_losses_stacked = jnp.stack(q_losses)
            info = {
                # --- Q-network metrics ---
                "q1_loss": q_losses[0],
                "q1_mean": jnp.mean(q_values_first),
                "q1_max": jnp.max(q_values_first),
                "q1_min": jnp.min(q_values_first),
                "q_ensemble_mean_loss": jnp.mean(q_losses_stacked),
                "q_ensemble_std": q_ensemble_std,           # per-sample Q disagreement (mean of std across K)
                "q_std_normalized": q_std_norm,
                "q_network_divergence": q_network_divergence,  # std of per-Q-net means (are Q-nets learning different values?)
                "q_network_max_spread": q_network_max_spread,  # max-min of per-Q-net means
                # --- Policy metrics ---
                "policy_loss": total_loss,
                "alpha": jnp.exp(log_alpha),
                "head_disagreement": head_disagreement,     # policy head noise prediction disagreement
                # --- Q-weight metrics (from policy loss) ---
                "q_weights_std": jnp.std(q_weights),
                "q_weights_mean": jnp.mean(q_weights),
                "q_weights_min": jnp.min(q_weights),
                "q_weights_max": jnp.max(q_weights),
                "hist_q_weights": q_weights,
                "hist_t": t,
                # --- Scale/running metrics ---
                "scale_q_mean": jnp.mean(scaled_q),
                "scale_q_std": jnp.std(scaled_q),
                "running_q_mean": new_running_mean,
                "running_q_std": new_running_std,
                "entropy_approx": 0.5 * self.agent.act_dim * jnp.log(
                    2 * jnp.pi * jnp.exp(1) * (0.1 * jnp.exp(log_alpha)) ** 2),
            }
            # Per-head policy losses
            for k in range(num_heads):
                info[f"head_loss_{k}"] = head_losses_stack[k]
            # Per-head bootstrap mask ratios
            for k in range(num_heads):
                info[f"head_mask_ratio_{k}"] = jnp.mean(bootstrap_mask[k])
            # Per-Q-network losses and mean values
            for i in range(num_heads):
                info[f"q_loss_{i}"] = q_losses[i]
                info[f"q_mean_{i}"] = per_q_means[i]
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
            # SVGD repulsion metrics
            if self.svgd_coeff > 0:
                info["svgd_loss"] = svgd_loss
                info["svgd_mean_dist"] = svgd_mean_dist
            # For ema_corr mode: compute correlation between head_disagreement and Q-std
            if self.idfm_mode == 'ema_corr':
                info["idfm_corr_estimate"] = head_disagreement / (q_std + 1e-8)
            return state, info

        # Wire up: Thompson, UCB, or default
        if exploration == 'thompson' and num_heads > 1:
            self._head_action_fns = []
            for k in range(num_heads):
                def _make_fn(head_k):
                    def fn(key, params, obs):
                        return self.agent.get_action_thompson(key, params, obs, head_k)
                    return fn
                self._head_action_fns.append(jax.jit(_make_fn(k)))

            self._implement_common_behavior(
                stateless_update,
                self.agent.get_action,
                self.agent.get_deterministic_action,
                stateless_get_value=self.agent.q)
        elif exploration == 'ucb' and num_heads > 1:
            _ucb_beta = self.ucb_beta
            def _ucb_action_fn(key, params, obs):
                return self.agent.get_action_ucb(key, params, obs, _ucb_beta)
            self._ucb_action_fn = jax.jit(_ucb_action_fn)

            self._implement_common_behavior(
                stateless_update,
                self.agent.get_action,
                self.agent.get_deterministic_action,
                stateless_get_value=self.agent.q)
        elif exploration == 'idfm' and num_heads > 1:
            _idfm_beta = self.idfm_beta
            _idfm_mode = self.idfm_mode
            if _idfm_mode == 'ema_corr':
                # For ema_corr, pass running_disagree as the EMA correlation value
                def _idfm_action_fn(key, params, obs, ema_val):
                    return self.agent.get_action_idfm(key, params, obs, _idfm_beta, _idfm_mode, ema_val)
                self._idfm_action_fn = jax.jit(_idfm_action_fn)
            else:
                def _idfm_action_fn(key, params, obs):
                    return self.agent.get_action_idfm(key, params, obs, _idfm_beta, _idfm_mode)
                self._idfm_action_fn = jax.jit(_idfm_action_fn)

            self._implement_common_behavior(
                stateless_update,
                self.agent.get_action,
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
            params = self.get_policy_params_to_save()
            unique_heads = np.unique(self.active_heads)
            if len(unique_heads) == 1:
                action = self._head_action_fns[unique_heads[0]](key, params, obs)
            else:
                all_actions = np.stack([
                    np.asarray(self._head_action_fns[k](key, params, obs))
                    for k in range(self.agent.num_heads)
                ])  # (K, num_envs, act_dim)
                action = all_actions[self.active_heads, np.arange(self.num_envs)]
        elif self.exploration == 'ucb' and self.agent.num_heads > 1:
            action = self._ucb_action_fn(key, self.get_policy_params_to_save(), obs)
        elif self.exploration == 'idfm' and self.agent.num_heads > 1:
            if self.idfm_mode == 'ema_corr':
                action = self._idfm_action_fn(key, self.get_policy_params_to_save(), obs,
                                               jnp.float32(self._ema_corr))
            else:
                action = self._idfm_action_fn(key, self.get_policy_params_to_save(), obs)
        else:
            action = self._get_action(key, self.get_policy_params_to_save(), obs)
        return np.asarray(action)

    # --- Params ---

    def get_policy_params(self):
        """Params for get_action inside JIT (head 0 + all Q_params)."""
        return (self.state.params.policy, self.state.params.log_alpha,
                self.state.params.q_params)

    def get_policy_params_to_save(self):
        """Params for data collection (target policy + all Q_params for Thompson/UCB)."""
        return (self.state.params.target_policy, self.state.params.log_alpha,
                self.state.params.q_params)

    def get_deterministic_params(self):
        """Params for deterministic action (evaluation): uses all K Q-networks."""
        return (self.state.params.target_policy, self.state.params.log_alpha,
                self.state.params.q_params)

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
        return self.state.params.q_params[0], (
            self.state.params.q_params[1] if self.agent.num_heads > 1
            else self.state.params.q_params[0])

    def save_q(self, path: str) -> None:
        value = jax.device_get(self.get_value_params())
        with open(path, "wb") as f:
            pickle.dump(value, f)
