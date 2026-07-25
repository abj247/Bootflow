"""
Q-Landscape Modality probe for MetaWorld peg-insert-side-v3 (rebuttal, reviewer Xu8d Q1).

Applies the SAME diagnostics as Appendix E (scripts/probe_q_modality.py) to the
peg-insert-side task, to test the paper's unimodality explanation for the ESAC
regression on this task:

  Row 1 -- distribution of Q over uniformly random actions at reference states
           (SDAC min(Q1,Q2), and ESAC ensemble-mean over K critics)
  Row 2 -- UMAP of action space colored by normalized Q

Prediction under the paper's hypothesis: a single concentrated high-Q region
(HalfCheetah-like), for BOTH the single-policy critic and the ensemble critics.
Also prints an ESAC head-dispersion statistic (inter-critic std on random
actions, normalized by Q range) to check the K heads see the same landscape.

Usage (after the peg_diag runs have checkpoints):
  python scripts/probe_q_modality_metaworld.py                       # auto-finds peg_diag runs
  python scripts/probe_q_modality_metaworld.py --checkpoint_step 200000

Outputs: plots/peg_insert_q_modality.{pdf,png} + printed stats block.
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.10")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # light evals; CPU avoids GPU contention

import argparse
import pickle
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import haiku as hk

from relax.env import create_env
from relax.utils.persistence import PersistFunction
from relax.network.blocks import QNet

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"


def mish(x):
    return x * jnp.tanh(jax.nn.softplus(x))


def find_run(env_name: str, pattern: str) -> Path:
    cands = sorted((LOGS / env_name).glob(pattern))
    if not cands:
        raise RuntimeError(f"no run matching {pattern} under logs/{env_name}")
    return cands[-1]


def load_checkpoint(run_dir: Path, step: int | None):
    pkls = sorted(run_dir.glob("policy-*.pkl"), key=lambda p: int(p.stem.split("-")[1]))
    if not pkls:
        raise RuntimeError(f"no policy-*.pkl checkpoints yet in {run_dir}")
    if step is not None:
        pkls = [p for p in pkls if int(p.stem.split("-")[1]) <= step] or pkls[:1]
    pkl = pkls[-1]
    print(f"  checkpoint: {pkl.name}")
    with open(pkl, "rb") as f:
        return pickle.load(f)


def build_q_fn(hidden_sizes):
    q_transform = hk.without_apply_rng(hk.transform(
        lambda obs, act: QNet(hidden_sizes, mish)(obs, act)))

    @jax.jit
    def q_fn(qp, obs, act):
        return q_transform.apply(qp, obs, act)
    return q_fn


def rollout_ref_states(run_dir: Path, params, env_name: str, n_steps: int,
                       n_ref: int, seed: int = 0):
    """Deterministic rollout of the SDAC policy to collect on-policy reference states."""
    det = PersistFunction.load(run_dir / "deterministic.pkl")

    @jax.jit
    def act_fn(p, obs):
        return det(p, obs).clip(-1, 1)

    env, obs_dim, act_dim = create_env(env_name, seed=seed)
    obs = env.reset()
    obs = obs[0] if isinstance(obs, tuple) else obs
    states = []
    for t in range(n_steps):
        a = np.asarray(act_fn(params, np.asarray(obs, np.float32)))
        states.append(np.asarray(obs, np.float32).copy())
        out = env.step(a)
        obs, done = out[0], (out[2] if len(out) == 4 else (out[2] or out[3]))
        if done:
            obs = env.reset()
            obs = obs[0] if isinstance(obs, tuple) else obs
    states = np.stack(states)
    idx = np.linspace(20, len(states) - 1, n_ref).astype(int)
    return states[idx], act_dim


def q_landscape(q_min_fn, ref_states, n_actions: int, act_dim: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    q_per_state = np.zeros((len(ref_states), n_actions))
    for i, s in enumerate(ref_states):
        acts = rng.uniform(-1, 1, size=(n_actions, act_dim)).astype(np.float32)
        obs = np.broadcast_to(s, (n_actions, s.size))
        q_per_state[i] = np.asarray(q_min_fn(obs, acts))
    # UMAP inputs from the middle reference state
    s = ref_states[len(ref_states) // 2]
    acts_u = rng.uniform(-1, 1, size=(n_actions, act_dim)).astype(np.float32)
    q_u = np.asarray(q_min_fn(np.broadcast_to(s, (n_actions, s.size)), acts_u)).copy()
    q_u = (q_u - q_u.min()) / (q_u.max() - q_u.min() + 1e-8)
    return q_per_state, acts_u, q_u


def umap_2d(X, seed=7):
    try:
        import umap
        return umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.15,
                         random_state=seed).fit_transform(X)
    except Exception:
        Xc = X - X.mean(0, keepdims=True)
        U, S, _ = np.linalg.svd(Xc, full_matrices=False)
        return U[:, :2] * S[:2]


def classify_modality(q_flat):
    try:
        from scipy.stats import gaussian_kde
        from scipy.signal import find_peaks
        kde = gaussian_kde(q_flat, bw_method=0.25)
        x = np.linspace(q_flat.min(), q_flat.max(), 400)
        d = kde(x)
        peaks, _ = find_peaks(d, prominence=d.max() * 0.10)
        return len(peaks), ("multimodal" if len(peaks) >= 2 else "unimodal")
    except Exception:
        return 1, "unimodal"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="peg-insert-side-v3")
    ap.add_argument("--sdac_dir", default=None)
    ap.add_argument("--esac_dir", default=None)
    ap.add_argument("--checkpoint_step", type=int, default=None,
                    help="use latest checkpoint <= this step (default: latest available)")
    ap.add_argument("--n_random_actions", type=int, default=3000)
    ap.add_argument("--n_ref_states", type=int, default=10)
    ap.add_argument("--n_traj_steps", type=int, default=400)
    args = ap.parse_args()

    sdac_dir = Path(args.sdac_dir) if args.sdac_dir else find_run(args.env, "bootflow_*peg_diag_sdac*")
    esac_dir = Path(args.esac_dir) if args.esac_dir else find_run(args.env, "fullbootflow_*peg_diag_tide*")
    print(f"SDAC run: {sdac_dir.name}\nESAC run: {esac_dir.name}")

    cfg = yaml.safe_load(open(sdac_dir / "config.yaml"))
    hidden_sizes = [cfg["hidden_dim"]] * cfg["hidden_num"]
    q_fn = build_q_fn(hidden_sizes)

    # ---- SDAC: (target_policy, log_alpha, q1, q2) ----
    print("[SDAC]")
    sdac_params = load_checkpoint(sdac_dir, args.checkpoint_step)
    _, _, q1p, q2p = sdac_params
    ref_states, act_dim = rollout_ref_states(sdac_dir, sdac_params, args.env,
                                             args.n_traj_steps, args.n_ref_states)

    sdac_qmin = jax.jit(lambda o, a: jnp.minimum(q_fn(q1p, o, a), q_fn(q2p, o, a)))
    q_sdac, acts_u_sdac, qcol_sdac = q_landscape(sdac_qmin, ref_states,
                                                 args.n_random_actions, act_dim)

    # ---- ESAC: (target_policy, log_alpha, (q_1..q_K)) ----
    print("[ESAC]")
    esac_params = load_checkpoint(esac_dir, args.checkpoint_step)
    _, _, qks = esac_params
    esac_qmean = jax.jit(lambda o, a: jnp.mean(
        jnp.stack([q_fn(qp, o, a) for qp in qks]), axis=0))
    q_esac, acts_u_esac, qcol_esac = q_landscape(esac_qmean, ref_states,
                                                 args.n_random_actions, act_dim)

    # head dispersion: inter-critic std / Q-range on random actions at ref states
    esac_qstd = jax.jit(lambda o, a: jnp.std(
        jnp.stack([q_fn(qp, o, a) for qp in qks]), axis=0))
    rng = np.random.default_rng(0)
    disp = []
    for s in ref_states:
        acts = rng.uniform(-1, 1, size=(args.n_random_actions, act_dim)).astype(np.float32)
        obs = np.broadcast_to(s, (args.n_random_actions, s.size))
        qs = np.asarray(esac_qstd(obs, acts))
        qm = np.asarray(esac_qmean(obs, acts))
        disp.append(qs.mean() / (qm.max() - qm.min() + 1e-8))
    disp = float(np.mean(disp))

    print("computing UMAPs (slow step)")
    u_sdac = umap_2d(acts_u_sdac)
    u_esac = umap_2d(acts_u_esac)

    # ---- stats ----
    zs = lambda q: ((q - q.mean(1, keepdims=True)) / (q.std(1, keepdims=True) + 1e-8)).ravel()
    n_peaks_sdac, cls_sdac = classify_modality(zs(q_sdac))
    n_peaks_esac, cls_esac = classify_modality(zs(q_esac))
    print("\n=== RESULTS (peg-insert-side-v3) ===")
    print(f"SDAC  min(Q1,Q2)   : {n_peaks_sdac} KDE peak(s) -> {cls_sdac}")
    print(f"ESAC  mean_K Q     : {n_peaks_esac} KDE peak(s) -> {cls_esac}")
    print(f"ESAC inter-critic dispersion (std/range on random actions): {disp:.4f}")
    print("Paper's hypothesis predicts: unimodal for both; low dispersion.")

    # ---- figure: 2x2 ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for j, (name, qps, U, qcol, cls) in enumerate([
            ("SDAC (min twin-Q)", q_sdac, u_sdac, qcol_sdac, cls_sdac),
            ("ESAC (ensemble-mean Q)", q_esac, u_esac, qcol_esac, cls_esac)]):
        ax = axes[0, j]
        for i in range(qps.shape[0]):
            z = (qps[i] - qps[i].mean()) / (qps[i].std() + 1e-8)
            ax.hist(z, bins=60, density=True, histtype="step", alpha=0.55)
        ax.set_title(f"{name}\nQ over random actions — {cls}")
        ax.set_xlabel("standardized Q(s*, a)")
        ax.set_ylabel("density")
        ax = axes[1, j]
        sc = ax.scatter(U[:, 0], U[:, 1], c=qcol, s=4, cmap="plasma")
        plt.colorbar(sc, ax=ax, label="normalized Q")
        ax.set_title("action-space UMAP colored by Q")
        ax.set_xlabel("UMAP dim 1"); ax.set_ylabel("UMAP dim 2")
    fig.suptitle("Q-landscape modality on peg-insert-side-v3 (Appendix-E protocol)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = ROOT / "plots"
    out.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"peg_insert_q_modality.{ext}")
    print(f"\nwritten: plots/peg_insert_q_modality.pdf/.png")


if __name__ == "__main__":
    main()
