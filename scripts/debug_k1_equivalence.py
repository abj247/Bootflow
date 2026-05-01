"""
Diagnostic: Compare K=1 BootFlow vs vanilla SDAC.

Checks parameter counts, output distributions, and loss values
to confirm K=1 BootFlow is architecturally equivalent to vanilla SDAC.

Usage:
    python scripts/debug_k1_equivalence.py
"""
import jax
import jax.numpy as jnp
import numpy as np

def mish(x):
    return x * jnp.tanh(jax.nn.softplus(x))

def main():
    obs_dim, act_dim = 27, 8  # Ant-v4
    hidden_sizes = [256, 256, 256]
    diffusion_hidden_sizes = [256, 256, 256]
    seed = 100

    key = jax.random.key(seed)

    # --- Create both networks with the same key ---
    from relax.network.diffv2 import create_diffv2_net
    from relax.network.diffv2_bootflow import create_bootflow_net

    sdac_net, sdac_params = create_diffv2_net(
        key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
        num_timesteps=20, num_particles=32, noise_scale=0.1,
        beta_schedule_scale=0.8)

    bf_net, bf_params = create_bootflow_net(
        key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
        num_timesteps=20, num_particles=32, noise_scale=0.1,
        beta_schedule_scale=0.8, num_heads=1, prior_scale=0.0)

    # --- Compare parameter counts ---
    def count_params(params):
        return sum(x.size for x in jax.tree.leaves(params))

    sdac_policy_count = count_params(sdac_params.policy)
    bf_policy_count = count_params(bf_params.policy)
    sdac_q_count = count_params(sdac_params.q1)
    bf_q_count = count_params(bf_params.q1)

    print("=" * 60)
    print("PARAMETER COMPARISON")
    print("=" * 60)
    print(f"SDAC policy params:    {sdac_policy_count:,}")
    print(f"BootFlow policy params: {bf_policy_count:,}")
    print(f"Match: {sdac_policy_count == bf_policy_count}")
    print(f"\nSDAC Q params:    {sdac_q_count:,}")
    print(f"BootFlow Q params: {bf_q_count:,}")
    print(f"Match: {sdac_q_count == bf_q_count}")

    # --- Compare parameter tree structure ---
    print("\n" + "=" * 60)
    print("PARAMETER TREE STRUCTURE")
    print("=" * 60)
    print("\nSDAC policy keys:")
    for k, v in jax.tree_util.tree_leaves_with_path(sdac_params.policy):
        path = "/".join(str(p) for p in k)
        print(f"  {path}: {v.shape}")

    print("\nBootFlow policy keys:")
    for k, v in jax.tree_util.tree_leaves_with_path(bf_params.policy):
        path = "/".join(str(p) for p in k)
        print(f"  {path}: {v.shape}")

    # --- Compare action outputs ---
    print("\n" + "=" * 60)
    print("ACTION OUTPUT COMPARISON")
    print("=" * 60)

    test_obs = jax.random.normal(jax.random.key(42), (5, obs_dim))
    action_key = jax.random.key(99)

    sdac_policy_tuple = (sdac_params.policy, sdac_params.log_alpha,
                         sdac_params.q1, sdac_params.q2)
    bf_policy_tuple = (bf_params.policy, bf_params.log_alpha,
                       bf_params.q1, bf_params.q2)

    sdac_actions = sdac_net.get_action(action_key, sdac_policy_tuple, test_obs)
    bf_actions = bf_net.get_action(action_key, bf_policy_tuple, test_obs)

    print(f"SDAC actions shape: {sdac_actions.shape}")
    print(f"BootFlow actions shape: {bf_actions.shape}")
    print(f"\nSDAC actions (first 3):\n{sdac_actions[:3]}")
    print(f"\nBootFlow actions (first 3):\n{bf_actions[:3]}")
    print(f"\nSDAC action stats: mean={sdac_actions.mean():.4f}, std={sdac_actions.std():.4f}")
    print(f"BootFlow action stats: mean={bf_actions.mean():.4f}, std={bf_actions.std():.4f}")

    # --- Compare deterministic actions ---
    print("\n" + "=" * 60)
    print("DETERMINISTIC ACTION COMPARISON")
    print("=" * 60)

    # Use target policy params for deterministic (matching save behavior)
    sdac_target_tuple = (sdac_params.target_poicy, sdac_params.log_alpha,
                         sdac_params.q1, sdac_params.q2)
    bf_target_tuple = (bf_params.target_policy, bf_params.log_alpha,
                       bf_params.q1, bf_params.q2)

    sdac_det = sdac_net.get_deterministic_action(sdac_target_tuple, test_obs)
    bf_det = bf_net.get_deterministic_action(bf_target_tuple, test_obs)

    print(f"SDAC det actions shape: {sdac_det.shape}")
    print(f"BootFlow det actions shape: {bf_det.shape}")
    print(f"\nSDAC det stats: mean={sdac_det.mean():.4f}, std={sdac_det.std():.4f}, "
          f"min={sdac_det.min():.4f}, max={sdac_det.max():.4f}")
    print(f"BootFlow det stats: mean={bf_det.mean():.4f}, std={bf_det.std():.4f}, "
          f"min={bf_det.min():.4f}, max={bf_det.max():.4f}")

    # --- Compare Q-values ---
    print("\n" + "=" * 60)
    print("Q-VALUE COMPARISON")
    print("=" * 60)

    test_act = jax.random.uniform(jax.random.key(7), (5, act_dim), minval=-1, maxval=1)

    sdac_q1 = sdac_net.q(sdac_params.q1, test_obs, test_act)
    bf_q1 = bf_net.q(bf_params.q1, test_obs, test_act)

    print(f"SDAC Q1: {sdac_q1}")
    print(f"BootFlow Q1: {bf_q1}")
    print(f"Q1 match: {jnp.allclose(sdac_q1, bf_q1)}")
    print(f"Q1 diff: {jnp.abs(sdac_q1 - bf_q1).max():.6f}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Policy param count match: {sdac_policy_count == bf_policy_count}")
    print(f"Q param count match: {sdac_q_count == bf_q_count}")
    print(f"Q values match (same init key): {jnp.allclose(sdac_q1, bf_q1)}")
    print(f"Action shapes match: {sdac_actions.shape == bf_actions.shape}")

    if sdac_policy_count != bf_policy_count:
        print(f"\n*** WARNING: Policy param count differs! "
              f"SDAC={sdac_policy_count}, BootFlow={bf_policy_count} ***")
        print("This means the architectures are NOT equivalent for K=1!")

if __name__ == "__main__":
    main()
