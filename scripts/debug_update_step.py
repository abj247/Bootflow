"""
Diagnostic: Run one update step for both SDAC and K=1 BootFlow,
compare the resulting loss, gradients, and state.
"""
import jax
import jax.numpy as jnp
import numpy as np
from relax.utils.random_utils import seeding
from relax.utils.experience import Experience

def mish(x):
    return x * jnp.tanh(jax.nn.softplus(x))

def main():
    obs_dim, act_dim = 27, 8  # Ant-v4
    hidden_sizes = [256, 256, 256]
    diffusion_hidden_sizes = [256, 256, 256]
    seed = 100
    batch_size = 256

    # Replicate the seed derivation from training scripts
    master_rng, _ = seeding(seed)
    env_seed, env_action_seed, eval_env_seed, buffer_seed, init_network_seed, train_seed = map(
        int, master_rng.integers(0, 2**32 - 1, 6))
    init_network_key = jax.random.key(init_network_seed)
    train_key = jax.random.key(train_seed)

    # Create both networks
    from relax.network.diffv2 import create_diffv2_net
    from relax.network.diffv2_bootflow import create_bootflow_net
    from relax.algorithm.sdac import SDAC
    from relax.algorithm.sdac_bootflow import BootflowSDAC

    sdac_net, sdac_params = create_diffv2_net(
        init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
        num_timesteps=20, num_particles=32, noise_scale=0.1, beta_schedule_scale=0.8)

    bf_net, bf_params = create_bootflow_net(
        init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
        num_timesteps=20, num_particles=32, noise_scale=0.1, beta_schedule_scale=0.8,
        num_heads=1, prior_scale=0.0)

    # Create algorithms
    sdac_alg = SDAC(sdac_net, sdac_params, lr=3e-4, alpha_lr=7e-3,
                    delay_alpha_update=250, lr_schedule_end=3e-5)
    bf_alg = BootflowSDAC(bf_net, bf_params, lr=3e-4, alpha_lr=7e-3,
                          delay_alpha_update=250, lr_schedule_end=3e-5,
                          bootstrap_prob=1.0, exploration='random_per_step',
                          mask_mode='random')

    # Create fake batch data
    data_key = jax.random.key(42)
    k1, k2, k3, k4, k5 = jax.random.split(data_key, 5)
    fake_data = Experience(
        obs=jax.random.normal(k1, (batch_size, obs_dim)),
        action=jax.random.uniform(k2, (batch_size, act_dim), minval=-1, maxval=1),
        reward=jax.random.normal(k3, (batch_size,)),
        done=jnp.zeros((batch_size,)),
        next_obs=jax.random.normal(k4, (batch_size, obs_dim)),
    )

    # Compare initial states
    print("=" * 60)
    print("INITIAL STATE COMPARISON")
    print("=" * 60)

    def param_norm(params):
        return float(jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree.leaves(params))))
    def param_sum(params):
        return float(sum(jnp.sum(x) for x in jax.tree.leaves(params)))

    print(f"SDAC policy norm:    {param_norm(sdac_alg.state.params.policy):.6f}")
    print(f"BootFlow policy norm: {param_norm(bf_alg.state.params.policy):.6f}")
    print(f"SDAC policy sum:    {param_sum(sdac_alg.state.params.policy):.6f}")
    print(f"BootFlow policy sum: {param_sum(bf_alg.state.params.policy):.6f}")

    print(f"SDAC Q1 norm:    {param_norm(sdac_alg.state.params.q1):.6f}")
    print(f"BootFlow Q1 norm: {param_norm(bf_alg.state.params.q1):.6f}")

    print(f"SDAC log_alpha: {sdac_alg.state.params.log_alpha}")
    print(f"BootFlow log_alpha: {bf_alg.state.params.log_alpha}")

    # Run warmup (JIT compilation)
    print("\n" + "=" * 60)
    print("WARMUP (JIT COMPILATION)")
    print("=" * 60)
    sdac_alg.warmup(fake_data)
    bf_alg.warmup(fake_data)
    print("Both warmed up successfully.")

    # Run one update step with the same key and data
    print("\n" + "=" * 60)
    print("ONE UPDATE STEP")
    print("=" * 60)

    update_key = jax.random.key(999)

    sdac_info, sdac_hist = sdac_alg.update(update_key, fake_data)
    bf_info, bf_hist = bf_alg.update(update_key, fake_data)

    print("\nSDCA info:")
    for k, v in sorted(sdac_info.items()):
        print(f"  {k}: {v:.6f}")

    print("\nBootFlow info:")
    for k, v in sorted(bf_info.items()):
        print(f"  {k}: {v:.6f}")

    # Compare specific metrics
    print("\n" + "=" * 60)
    print("METRIC COMPARISON")
    print("=" * 60)

    common_keys = set(sdac_info.keys()) & set(bf_info.keys())
    for k in sorted(common_keys):
        s, b = sdac_info[k], bf_info[k]
        match = abs(s - b) < 1e-5
        status = "✓" if match else f"DIFF: {abs(s-b):.6f}"
        print(f"  {k}: SDAC={s:.6f}, BF={b:.6f} {status}")

    bf_only = set(bf_info.keys()) - set(sdac_info.keys())
    if bf_only:
        print(f"\nBootFlow-only keys: {sorted(bf_only)}")

    # Compare policy params after update
    print("\n" + "=" * 60)
    print("PARAMS AFTER UPDATE")
    print("=" * 60)

    print(f"SDAC policy norm after:    {param_norm(sdac_alg.state.params.policy):.6f}")
    print(f"BootFlow policy norm after: {param_norm(bf_alg.state.params.policy):.6f}")
    print(f"SDAC policy sum after:    {param_sum(sdac_alg.state.params.policy):.6f}")
    print(f"BootFlow policy sum after: {param_sum(bf_alg.state.params.policy):.6f}")
    norm_match = abs(param_norm(sdac_alg.state.params.policy) - param_norm(bf_alg.state.params.policy)) < 1e-4
    print(f"Policy norms match after update: {norm_match}")

    # Compare get_action outputs after update
    print("\n" + "=" * 60)
    print("ACTIONS AFTER UPDATE")
    print("=" * 60)

    test_obs = jax.random.normal(jax.random.key(77), (5, obs_dim))
    action_key = jax.random.key(88)

    sdac_act = sdac_alg.get_action(action_key, test_obs)
    bf_act = bf_alg.get_action(action_key, test_obs)

    act_match = np.allclose(sdac_act, bf_act, atol=1e-5)
    if not act_match:
        print(f"Actions MISMATCH after update!")
        print(f"Max diff: {np.abs(sdac_act - bf_act).max():.8f}")
        print(f"SDAC: {sdac_act[0]}")
        print(f"BF:   {bf_act[0]}")
    else:
        print(f"Actions match after update: {act_match}")

if __name__ == "__main__":
    main()
