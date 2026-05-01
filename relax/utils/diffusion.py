from typing import Protocol, Tuple
from dataclasses import dataclass

import numpy as np
import jax, jax.numpy as jnp
import optax

class DiffusionModel(Protocol):
    def __call__(self, t: jax.Array, x: jax.Array) -> jax.Array:
        ...

@dataclass(frozen=True)
class BetaScheduleCoefficients:
    betas: jax.Array
    alphas: jax.Array
    alphas_cumprod: jax.Array
    alphas_cumprod_prev: jax.Array
    sqrt_alphas_cumprod: jax.Array
    sqrt_one_minus_alphas_cumprod: jax.Array
    log_one_minus_alphas_cumprod: jax.Array
    sqrt_recip_alphas_cumprod: jax.Array
    sqrt_recipm1_alphas_cumprod: jax.Array
    posterior_variance: jax.Array
    posterior_log_variance_clipped: jax.Array
    posterior_mean_coef1: jax.Array
    posterior_mean_coef2: jax.Array

    @staticmethod
    def from_beta(betas: np.ndarray):
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1. - alphas_cumprod)
        log_one_minus_alphas_cumprod = np.log(1. - alphas_cumprod)
        sqrt_recip_alphas_cumprod = np.sqrt(1. / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1. / alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_log_variance_clipped = np.log(np.maximum(posterior_variance, 1e-20))
        posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_mean_coef2 = (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)

        return BetaScheduleCoefficients(
            *jax.device_put((
                betas, alphas, alphas_cumprod, alphas_cumprod_prev,
                sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, log_one_minus_alphas_cumprod,
                sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod,
                posterior_variance, posterior_log_variance_clipped, posterior_mean_coef1, posterior_mean_coef2
            ))
        )

    @staticmethod
    def vp_beta_schedule(timesteps: int):
        t = np.arange(1, timesteps + 1)
        T = timesteps
        b_max = 10.
        b_min = 0.1
        alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
        betas = 1 - alpha
        return betas

    @staticmethod
    def cosine_beta_schedule(timesteps: int):
        s = 0.008
        t = np.arange(0, timesteps + 1) / timesteps
        alphas_cumprod = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
        alphas_cumprod /= alphas_cumprod[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = np.clip(betas, 0, 0.999)
        return betas
    
    @staticmethod
    def linear_beta_schedule(timesteps: int, beta_start=1e-4, beta_end=0.999):
        return np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)

@dataclass(frozen=True)
class GaussianDiffusion:
    num_timesteps: int
    beta_schedule_scale: float = 0.3
    beta_schedule_type: str = 'linear'

    def beta_schedule(self):
        with jax.ensure_compile_time_eval():
            if self.beta_schedule_type == 'linear':
                betas = self.beta_schedule_scale * BetaScheduleCoefficients.linear_beta_schedule(self.num_timesteps)
            elif self.beta_schedule_type == 'cosine':
                betas = self.beta_schedule_scale * BetaScheduleCoefficients.cosine_beta_schedule(self.num_timesteps)
            return BetaScheduleCoefficients.from_beta(betas)

    def p_mean_variance(self, t: int, x: jax.Array, noise_pred: jax.Array):
        B = self.beta_schedule()
        x_recon = x * B.sqrt_recip_alphas_cumprod[t] - noise_pred * B.sqrt_recipm1_alphas_cumprod[t]
        x_recon = jnp.clip(x_recon, -1, 1)
        model_mean = x_recon * B.posterior_mean_coef1[t] + x * B.posterior_mean_coef2[t]
        model_log_variance = B.posterior_log_variance_clipped[t]
        return model_mean, model_log_variance
    
    def get_recon(self, t: int, x: jax.Array, noise: jax.Array):
        B = self.beta_schedule()
        x_recon = x * B.sqrt_recip_alphas_cumprod[t][:, jnp.newaxis] - noise * B.sqrt_recipm1_alphas_cumprod[t][:, jnp.newaxis]
        return x_recon

    def p_sample(self, key: jax.Array, model: DiffusionModel, shape: Tuple[int, ...]) -> jax.Array:
        x_key, noise_key = jax.random.split(key)
        x = 0.5 * jax.random.normal(x_key, shape)
        noise = jax.random.normal(noise_key, (self.num_timesteps, *shape))

        def body_fn(x, input):
            t, noise = input
            noise_pred = model(t, x)
            model_mean, model_log_variance = self.p_mean_variance(t, x, noise_pred)
            x = model_mean + (t > 0) * jnp.exp(0.5 * model_log_variance) * noise
            return x, None

        t = jnp.arange(self.num_timesteps)[::-1]
        x, _ = jax.lax.scan(body_fn, x, (t, noise))
        return x

    def q_sample(self, t: int, x_start: jax.Array, noise: jax.Array):
        B = self.beta_schedule()
        return B.sqrt_alphas_cumprod[t] * x_start + B.sqrt_one_minus_alphas_cumprod[t] * noise

    def p_loss(self, key: jax.Array, model: DiffusionModel, t: jax.Array, x_start: jax.Array):
        assert t.ndim == 1 and t.shape[0] == x_start.shape[0]

        noise = jax.random.normal(key, x_start.shape)
        x_noisy = jax.vmap(self.q_sample)(t, x_start, noise)
        noise_pred = model(t, x_noisy)
        loss = optax.l2_loss(noise_pred, noise)
        return loss.mean()

    def weighted_p_loss(self, key: jax.Array, weights: jax.Array, model: DiffusionModel, t: jax.Array,
                        x_start: jax.Array):
        if len(weights.shape) == 1:
            weights = weights.reshape(-1, 1)
        assert t.ndim == 1 and t.shape[0] == x_start.shape[0]
        noise = jax.random.normal(key, x_start.shape)
        x_noisy = jax.vmap(self.q_sample)(t, x_start, noise)
        noise_pred = model(t, x_noisy)
        loss = weights * optax.squared_error(noise_pred, noise)
        return loss.mean()
    
    def reverse_samping_weighted_p_loss(self, noise: jax.Array, weights: jax.Array, model: DiffusionModel, t: jax.Array,
                        x_t: jax.Array, x_start: jax.Array = None):
        if len(weights.shape) == 1:
            weights = weights.reshape(-1, 1)
        assert t.ndim == 1 and t.shape[0] == x_t.shape[0]
        noise_pred = model(t, x_t)
        loss = weights * optax.squared_error(noise_pred, noise)
        return loss.mean()
    

@dataclass(frozen=True)
class FlowMatching:
    """Conditional Flow Matching (Lipman et al. 2023, FPO convention) as drop-in replacement for GaussianDiffusion.

    Convention (matches FPO and SDAC's t ordering):
      t=0 → clean data (x_0), t=1 → pure noise
      Forward:  x_t = (1 - t) * x_0 + t * noise
      Network predicts: v = noise - x_0  (velocity from data toward noise)
      Loss: ||v_pred - (noise - x_0)||^2
      Reverse:  Euler ODE integration from t=1 (noise) backward to t=0 (data)
                x_{t-dt} = x_t - v_pred * dt
    """
    num_timesteps: int
    beta_schedule_scale: float = 0.3  # unused, kept for interface compatibility
    beta_schedule_type: str = 'linear'  # unused, kept for interface compatibility

    def _t_continuous(self, t: jax.Array) -> jax.Array:
        """Convert discrete timestep index (0..T-1) to continuous time [~0, ~1].
        t=0 maps to ~0 (near clean data), t=T-1 maps to ~1 (near pure noise).
        Matches SDAC convention: high t = more noisy."""
        return jnp.clip((t + 1).astype(jnp.float32) / self.num_timesteps, 1e-5, 1.0 - 1e-5)

    def q_sample(self, t: int, x_start: jax.Array, noise: jax.Array):
        """Forward process: interpolate between data and noise.
        x_t = (1 - t_c) * x_start + t_c * noise
        t=0 → x_start (clean), t=1 → noise (pure noise)
        """
        t_c = self._t_continuous(t)
        if t_c.ndim == 0:
            return (1 - t_c) * x_start + t_c * noise
        return (1 - t_c[:, jnp.newaxis]) * x_start + t_c[:, jnp.newaxis] * noise

    def get_recon(self, t: int, x_t: jax.Array, velocity_pred: jax.Array):
        """Reconstruct x_0 from x_t and predicted velocity.
        v = noise - x_0, and x_t = (1-t)*x_0 + t*noise
        x_0 = x_t - t * v
        """
        t_c = self._t_continuous(t)
        return x_t - t_c[:, jnp.newaxis] * velocity_pred

    def p_sample(self, key: jax.Array, model: DiffusionModel, shape: Tuple[int, ...]) -> jax.Array:
        """Reverse process: Euler ODE integration from noise (t=1) backward to data (t=0).
        x_{t-dt} = x_t - v_theta(x_t, t) * dt
        """
        x = jax.random.normal(key, shape)

        dt = 1.0 / self.num_timesteps

        def body_fn(x, t_idx):
            # t_idx goes T-1, T-2, ..., 0 (from noisy to clean)
            v_pred = model(t_idx, x)
            x = x - v_pred * dt
            return x, None

        t_indices = jnp.arange(self.num_timesteps)[::-1]  # T-1 down to 0
        x, _ = jax.lax.scan(body_fn, x, t_indices)
        return x

    def p_loss(self, key: jax.Array, model: DiffusionModel, t: jax.Array, x_start: jax.Array):
        """Standard flow matching loss: ||v_pred - (noise - x_0)||^2"""
        assert t.ndim == 1 and t.shape[0] == x_start.shape[0]
        noise = jax.random.normal(key, x_start.shape)
        x_t = jax.vmap(self.q_sample)(t, x_start, noise)
        v_pred = model(t, x_t)
        v_target = noise - x_start
        loss = optax.l2_loss(v_pred, v_target)
        return loss.mean()

    def weighted_p_loss(self, key: jax.Array, weights: jax.Array, model: DiffusionModel, t: jax.Array,
                        x_start: jax.Array):
        if len(weights.shape) == 1:
            weights = weights.reshape(-1, 1)
        assert t.ndim == 1 and t.shape[0] == x_start.shape[0]
        noise = jax.random.normal(key, x_start.shape)
        x_t = jax.vmap(self.q_sample)(t, x_start, noise)
        v_pred = model(t, x_t)
        v_target = noise - x_start
        loss = weights * optax.squared_error(v_pred, v_target)
        return loss.mean()

    def reverse_samping_weighted_p_loss(self, noise: jax.Array, weights: jax.Array, model: DiffusionModel,
                                        t: jax.Array, x_t: jax.Array, x_start: jax.Array = None):
        """Weighted flow matching loss for policy training.

        For flow matching, the velocity target is v = noise - x_0.
        x_start (x_0, the clean action) must be provided.
        If x_start is None, falls back to using noise as target (for backward compat).
        """
        if len(weights.shape) == 1:
            weights = weights.reshape(-1, 1)
        assert t.ndim == 1 and t.shape[0] == x_t.shape[0]
        t_c = self._t_continuous(t)
        if x_start is not None:
            # Proper flow matching: v_target = noise - x_0
            # We recover noise from x_t and x_0: noise = (x_t - (1-t)*x_0) / t
            noise_recovered = (x_t - (1 - t_c[:, jnp.newaxis]) * x_start) / jnp.maximum(t_c[:, jnp.newaxis], 1e-5)
            v_target = noise_recovered - x_start
        else:
            # Fallback: use noise argument directly as velocity target
            v_target = noise
        v_pred = model(t, x_t)
        loss = weights * optax.squared_error(v_pred, v_target)
        return loss.mean()


if __name__ == '__main__':
    diffusion = GaussianDiffusion(20)
    beta_schedule = diffusion.beta_schedule(scale=0.3)
    print("betas", beta_schedule.betas)
    print("sqrt 1 - bar alpha", beta_schedule.sqrt_one_minus_alphas_cumprod)
    print("sqrt 1 over bar alpha", beta_schedule.sqrt_recip_alphas_cumprod)
    print("sqrt 1 - bar alpha over bar alpha", beta_schedule.sqrt_recipm1_alphas_cumprod)

