#!/usr/bin/env python3
"""
NPMLE Tests - Parallelized version for HPC cluster
Runs on a node with 120 CPUs and 8 H200 GPUs

This script is converted from NPMLE_py_tests.ipynb with multiprocessing support.
"""

import numpy as np
import math
import pickle
import argparse
import os
from scipy import stats, optimize as opt
from scipy.stats import norm, poisson, gaussian_kde
from scipy.optimize import root_scalar
from collections import Counter
from multiprocessing import Pool, cpu_count
from functools import partial
import time

# Suppress matplotlib GUI backend for headless execution
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    print("Warning: cvxpy not available, some features may not work")


# =============================================================================
# PRIOR GENERATION FUNCTIONS
# =============================================================================

def two_point_adversarial_prior(mu, si_min, si_max, m, n, rng=None):
    """Two-point adversarial prior"""
    assert m < n, f"Assertion failed: m ({m}) is not less than n ({n})"
    if rng is None:
        rng = np.random.default_rng()
    scales = rng.choice([si_min, si_max], size=n, p=[(m/n), 1-(m/n)])
    return rng.normal(mu*np.ones(n), scales)


def two_point_alpha_adversarial_prior(si_min, alpha, n, rng=None):
    """Two-point alpha-mixture prior (Po Ling-Loh)"""
    return two_point_adversarial_prior(1, n**(2*alpha), np.floor(np.log(n)), n, rng)


def sample_scale_mixture(mu, prior_atoms, prior_weights, n=1, rng=None):
    """General scale mixture of Gaussians sampling"""
    scales = np.asarray(prior_atoms)
    weights = np.asarray(prior_weights)
    if rng is None:
        rng = np.random.default_rng()
    comps = rng.choice(len(weights), size=n, p=weights)
    return rng.normal(mu*np.ones(n), scales[comps])


def subset_of_signals_adversarial_prior(mu, si_min, si_max, m, n, rng=None):
    """Subset-of-signals prior"""
    if not (0 <= m/n <= 1):
        raise ValueError("Probability q must be between 0 and 1.")
    if not (si_min <= 1 <= si_max):
        raise ValueError("Condition si_min <= 1 <= K must be met for valid intervals.")
    if rng is None:
        rng = np.random.default_rng()
    scales = np.zeros(n)
    r = rng.random(n)
    idx_interval1 = (r < (m/n))
    idx_interval2 = ~idx_interval1
    scales[idx_interval1] = rng.uniform(si_min, 1, size=np.sum(idx_interval1))
    scales[idx_interval2] = rng.uniform(1, si_max, size=np.sum(idx_interval2))
    return rng.normal(mu*np.ones(n), scales)


def generate_data_from_prior(prior_atoms, prior_weights, mu, n, rng=None):
    """Generate data from a given prior"""
    if rng is None:
        rng = np.random.default_rng()
    sig = rng.choice(prior_atoms, size=n, p=prior_weights)
    return rng.normal(mu*np.ones(n), sig), sig


# =============================================================================
# ESTIMATOR FUNCTIONS
# =============================================================================

def median_estimator(N_obs):
    """Median estimator (standard)"""
    return np.median(N_obs)


def phi_interval(x, a, b):
    """phi(x; [a,b]) = a if x < a; x if a <= x <= b; b if x > b."""
    return np.clip(x, a, b)


def iterative_truncation_estimator(N_obs, B, mu0, mu_star=None, m=1.0, n=None, return_path=False):
    """Mean Estimation via Iterative Truncation (Algorithm 1)."""
    X = np.asarray(N_obs, dtype=float)
    if n is None:
        n = X.shape[0]
    else:
        n = int(n)
        if n != X.shape[0]:
            raise ValueError("Provided n does not match len(X).")

    delta = float(B)
    mu_k0 = float(mu0)

    if delta <= 1.0:
        K = 0
    else:
        K = int(math.ceil(math.log2(delta)))

    T = max(1, int(math.ceil(64.0 * n * math.log(n) / m)))

    history = []
    for k in range(K + 1):
        mu_t = mu_k0
        for _ in range(T + 1):
            a = mu_t - delta
            b = mu_t + delta
            truncated = phi_interval(X, a, b)
            mu_t = truncated.mean()
            if return_path:
                history.append(mu_t)
        mu_k0 = mu_t
        delta = delta / 2.0

    mu_hat = mu_t
    if return_path:
        return mu_hat, history
    return mu_hat


def oracle_estimator(N_obs, N_vars):
    """MLE estimator under oracle knowledge of the full list of variances"""
    N_inv_vars = 1 / N_vars
    dot_prod = np.dot(N_inv_vars, N_obs)
    norm_inv_Y = np.linalg.norm(N_inv_vars)
    return dot_prod / norm_inv_Y


def mle_known_sigma(X, sigma):
    """MLE estimator for mu when the variances sigma_i^2 are known."""
    X = np.asarray(X, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    weights = 1.0 / (sigma ** 2)
    mu_hat = np.sum(X * weights) / np.sum(weights)
    return mu_hat


# =============================================================================
# NPMLE FUNCTIONS
# =============================================================================

def mixture_density(prior_atoms, prior_weights, mu, obs):
    """Compute mixture density"""
    k = len(prior_atoms)
    densities = np.zeros(len(obs))
    for i in range(k):
        densities += prior_weights[i] * norm.pdf(obs, loc=mu, scale=prior_atoms[i])
    return densities


def avg_loglikelihood(prior_atoms, prior_weights, mu, N_obs):
    """Average log-likelihood"""
    densities = mixture_density(prior_atoms, prior_weights, mu, N_obs)
    densities = np.maximum(densities, 1e-300)
    return np.mean(np.log(densities))


def DG_sigma(prior_atoms, prior_weights, sigma, mu, N_obs, denom=None):
    """Directional derivative for atom update"""
    n = len(N_obs)
    if denom is None:
        denom = mixture_density(prior_atoms, prior_weights, mu, N_obs)
    denom = np.maximum(denom, 1e-300)
    numer = norm.pdf(N_obs, loc=mu, scale=sigma)
    return (1.0/n) * np.sum(numer/denom) - 1.0


def grad_DG_sigma(prior_atoms, prior_weights, sigma, mu, N_obs, denom=None):
    """Gradient of directional derivative"""
    n = len(N_obs)
    if denom is None:
        denom = mixture_density(prior_atoms, prior_weights, mu, N_obs)
    denom = np.maximum(denom, 1e-300)

    z = (N_obs - mu) / sigma
    phi = norm.pdf(z)
    dphi_dsigma = phi * (z**2 - 1) / sigma
    numer = dphi_dsigma / sigma - phi / (sigma**2)
    return (1.0/n) * np.sum(numer / denom)


def update_atoms(atoms, weights, mu, N_obs, si_min, si_max):
    """Update atoms using gradient-based optimization"""
    denom = mixture_density(atoms, weights, mu, N_obs)

    def neg_DG(s):
        return -DG_sigma(atoms, weights, s, mu, N_obs, denom=denom)

    def neg_grad_DG(s):
        return -grad_DG_sigma(atoms, weights, s, mu, N_obs, denom=denom)

    try:
        result = opt.minimize_scalar(neg_DG, bounds=(si_min, si_max), method='bounded')
        s_star = result.x
        val = -result.fun
    except:
        s_star = (si_min + si_max) / 2
        val = DG_sigma(atoms, weights, s_star, mu, N_obs, denom=denom)

    return s_star, val


def get_weights_npmle_EM(x, scales, mu=0.0, w_init=None, max_iter=500, tol=1e-8):
    """EM algorithm for NPMLE weights"""
    x = np.asarray(x, dtype=float)
    scales = np.asarray(scales, dtype=float)
    n = x.size
    K = scales.size

    if w_init is None:
        w = np.ones(K) / K
    else:
        w = np.asarray(w_init, dtype=float).copy()
        w = np.maximum(w, 1e-12)
        w /= w.sum()

    phi = np.zeros((n, K))
    for j in range(K):
        phi[:, j] = norm.pdf(x, loc=mu, scale=scales[j])

    for _ in range(max_iter):
        f = phi @ w
        f = np.maximum(f, 1e-300)
        resp = phi * w[None, :]
        resp /= f[:, None]
        w_new = resp.mean(axis=0)
        w_new = np.maximum(w_new, 1e-12)
        w_new /= w_new.sum()

        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new

    return w


def merge_atoms_weights(atoms, weights, atom_tol=1e-3):
    """Merge atoms that are too close"""
    atoms = np.asarray(atoms)
    weights = np.asarray(weights)
    idx = np.argsort(atoms)
    atoms = atoms[idx]
    weights = weights[idx]

    merged_atoms = [atoms[0]]
    merged_weights = [weights[0]]

    for i in range(1, len(atoms)):
        if abs(atoms[i] - merged_atoms[-1]) < atom_tol:
            merged_weights[-1] += weights[i]
        else:
            merged_atoms.append(atoms[i])
            merged_weights.append(weights[i])

    return np.array(merged_atoms), np.array(merged_weights)


def npmle_heteroskedastic(N_obs, mu, si_min, si_max, atoms_init=None, weights_init=None,
                          weight_tol=1e-6, atom_tol=1e-6, progress_tol=1e-6,
                          iter_max=20, verbose=False):
    """NPMLE for heteroskedastic model (fixed mu)"""
    n = len(N_obs)

    if atoms_init is None:
        atoms = np.array([max(si_min, np.std(N_obs))])
    else:
        atoms = np.asarray(atoms_init).copy()

    if weights_init is None:
        weights = np.ones(len(atoms)) / len(atoms)
    else:
        weights = np.asarray(weights_init).copy()

    prev_ll = -np.inf

    for it in range(iter_max):
        # Update weights
        weights = get_weights_npmle_EM(N_obs, atoms, mu=mu, w_init=weights)

        # Prune zero-weight atoms
        mask = weights > weight_tol
        atoms = atoms[mask]
        weights = weights[mask]
        weights /= weights.sum()

        if len(atoms) == 0:
            atoms = np.array([max(si_min, np.std(N_obs))])
            weights = np.array([1.0])

        # Update atoms (add new atom)
        s_star, dg_val = update_atoms(atoms, weights, mu, N_obs, si_min, si_max)

        if dg_val > 1e-6:
            atoms = np.append(atoms, s_star)
            weights = np.append(weights, 0.0)
            weights = get_weights_npmle_EM(N_obs, atoms, mu=mu, w_init=weights)

        # Check convergence
        ll = avg_loglikelihood(atoms, weights, mu, N_obs)
        if abs(ll - prev_ll) < progress_tol:
            break
        prev_ll = ll

    # Merge close atoms
    atoms, weights = merge_atoms_weights(atoms, weights, atom_tol)

    return atoms, weights


def mle_location_scale_mixture(N_obs, atoms, weights, mu_init=None):
    """MLE for location given scale mixture"""
    N_obs = np.asarray(N_obs, dtype=float)

    if mu_init is None:
        mu_init = np.median(N_obs)

    def nll(mu):
        return -avg_loglikelihood(atoms, weights, mu, N_obs)

    def grad_nll(mu):
        denom = mixture_density(atoms, weights, mu, N_obs)
        denom = np.maximum(denom, 1e-300)
        grad = 0.0
        for i, (s, w) in enumerate(zip(atoms, weights)):
            z = (N_obs - mu) / s
            grad += w * np.sum(z * norm.pdf(z) / s / denom)
        return -grad / len(N_obs)

    try:
        result = opt.minimize(nll, mu_init, jac=grad_nll, method='BFGS')
        return result.x[0] if hasattr(result.x, '__len__') else result.x
    except:
        return mu_init


def npmle_alternating_full(N_obs, si_min=0, si_max=np.inf, verbose=False, iter_max=20):
    """Full NPMLE with alternating optimization for mu and scale distribution"""
    N_obs = np.asarray(N_obs, dtype=float)
    n = len(N_obs)

    if si_min <= 0:
        si_min = 0.01
    if si_max == np.inf:
        si_max = 10 * np.std(N_obs)

    mu = np.median(N_obs)
    atoms = np.array([max(si_min, np.std(N_obs))])
    weights = np.array([1.0])

    prev_ll = -np.inf

    for iter_count in range(iter_max):
        # Update scale mixture (fixed mu)
        atoms, weights = npmle_heteroskedastic(
            N_obs, mu, si_min, si_max,
            atoms_init=atoms, weights_init=weights,
            iter_max=10, verbose=False
        )

        # Update mu (fixed scale mixture)
        mu = mle_location_scale_mixture(N_obs, atoms, weights, mu_init=mu)

        # Check convergence
        ll = avg_loglikelihood(atoms, weights, mu, N_obs)
        if verbose:
            print(f"Iteration {iter_count+1}: mu={mu:.6f}, ll={ll:.6f}, atoms={len(atoms)}")

        if abs(ll - prev_ll) < 1e-6:
            break
        prev_ll = ll

    return atoms, weights, mu


# =============================================================================
# PARALLEL SIMULATION FUNCTIONS
# =============================================================================

def run_single_simulation_sos(args):
    """Run a single simulation for Subset-of-Signals experiment"""
    s, n, true_mu, si_min, si_max, B, mu0, seed = args
    rng = np.random.default_rng(seed)

    m = max(1, int(np.sqrt(n * np.log(n))))

    # Generate data
    X = subset_of_signals_adversarial_prior(
        mu=true_mu, si_min=si_min, si_max=si_max, m=m, n=n, rng=rng)

    # 1. Median estimator
    mu_hat_median = median_estimator(X)
    err_median = np.abs(mu_hat_median - true_mu)

    # 2. Iterative truncation estimator
    mu_hat_trunc = iterative_truncation_estimator(
        N_obs=X, B=B, mu0=mu0, mu_star=true_mu, m=m, n=n, return_path=False)
    err_trunc = np.abs(mu_hat_trunc - true_mu)

    # 3. NPMLE estimator
    try:
        _, _, mu_hat_npmle = npmle_alternating_full(
            N_obs=X, si_min=si_min, si_max=si_max, iter_max=8, verbose=False)
        err_npmle = np.abs(mu_hat_npmle - true_mu)
    except Exception as e:
        err_npmle = np.nan

    return {
        'n': n,
        'sim': s,
        'err_median': err_median,
        'err_trunc': err_trunc,
        'err_npmle': err_npmle
    }


def run_single_simulation_2pt(args):
    """Run a single simulation for Two-Point Prior experiment"""
    s, n, true_mu, prior_atoms, prior_weights, si_min, si_max, seed = args
    rng = np.random.default_rng(seed)

    # Generate data from two-point prior
    X, true_sigmas = generate_data_from_prior(prior_atoms, prior_weights, true_mu, n, rng=rng)

    # 1. MLE with known G (known sigmas)
    mu_hat_mle_known = mle_known_sigma(X, true_sigmas)
    err_mle_known = np.abs(mu_hat_mle_known - true_mu)

    # 2. NPMLE estimator (unknown G)
    try:
        _, _, mu_hat_npmle = npmle_alternating_full(
            N_obs=X, si_min=si_min, si_max=si_max, iter_max=8, verbose=False)
        err_npmle = np.abs(mu_hat_npmle - true_mu)
    except Exception as e:
        err_npmle = np.nan

    return {
        'n': n,
        'sim': s,
        'err_mle_known': err_mle_known,
        'err_npmle': err_npmle
    }


def compute_stats(results_dict, n_values):
    """Compute mean and standard error for each estimator at each n"""
    means = []
    stds = []
    for n in n_values:
        vals = np.array(results_dict[n])
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            means.append(np.mean(vals))
            stds.append(np.std(vals) / np.sqrt(len(vals)))
        else:
            means.append(np.nan)
            stds.append(np.nan)
    return np.array(means), np.array(stds)


def run_experiment_sos(n_values, n_sims, true_mu, si_min, si_max, B, mu0, n_workers, output_dir):
    """Run Subset-of-Signals experiment with parallel processing"""
    print("=" * 60)
    print("EXPERIMENT 1: Subset-of-Signals Prior")
    print("=" * 60)
    print(f"n values: {n_values}")
    print(f"Simulations per n: {n_sims}")
    print(f"Using {n_workers} workers")
    print("=" * 60)

    # Prepare all simulation arguments
    all_args = []
    base_seed = 42
    for n in n_values:
        for s in range(n_sims):
            seed = base_seed + n * n_sims + s
            all_args.append((s, n, true_mu, si_min, si_max, B, mu0, seed))

    print(f"Total simulations to run: {len(all_args)}")

    # Run in parallel
    start_time = time.time()
    with Pool(n_workers) as pool:
        results = list(pool.imap_unordered(run_single_simulation_sos, all_args))
    elapsed = time.time() - start_time

    print(f"\nCompleted {len(results)} simulations in {elapsed:.2f} seconds")

    # Organize results
    results_median = {n: [] for n in n_values}
    results_truncation = {n: [] for n in n_values}
    results_npmle = {n: [] for n in n_values}

    for r in results:
        n = r['n']
        results_median[n].append(r['err_median'])
        results_truncation[n].append(r['err_trunc'])
        results_npmle[n].append(r['err_npmle'])

    # Compute statistics
    median_means, median_stds = compute_stats(results_median, n_values)
    trunc_means, trunc_stds = compute_stats(results_truncation, n_values)
    npmle_means, npmle_stds = compute_stats(results_npmle, n_values)

    # Save results
    results_data = {
        'n_values': n_values,
        'results_median': results_median,
        'results_truncation': results_truncation,
        'results_npmle': results_npmle,
        'median_means': median_means,
        'median_stds': median_stds,
        'trunc_means': trunc_means,
        'trunc_stds': trunc_stds,
        'npmle_means': npmle_means,
        'npmle_stds': npmle_stds,
        'params': {
            'true_mu': true_mu,
            'si_min': si_min,
            'si_max': si_max,
            'B': B,
            'mu0': mu0,
            'n_sims': n_sims
        }
    }

    results_file = os.path.join(output_dir, 'experiment1_sos_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results_data, f)
    print(f"Results saved to {results_file}")

    # Create plot
    create_plot_sos(n_values, median_means, median_stds, trunc_means, trunc_stds,
                    npmle_means, npmle_stds, si_min, si_max, output_dir)

    return results_data


def run_experiment_2pt(n_values, n_sims, true_mu, prior_atoms, prior_weights,
                       si_min, si_max, n_workers, output_dir):
    """Run Two-Point Prior experiment with parallel processing"""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Two-Point Prior (MLE vs NPMLE)")
    print("=" * 60)
    print(f"Prior atoms: {prior_atoms}, weights: {prior_weights}")
    print(f"n values: {n_values}")
    print(f"Simulations per n: {n_sims}")
    print(f"Using {n_workers} workers")
    print("=" * 60)

    # Prepare all simulation arguments
    all_args = []
    base_seed = 10000
    for n in n_values:
        for s in range(n_sims):
            seed = base_seed + n * n_sims + s
            all_args.append((s, n, true_mu, prior_atoms, prior_weights, si_min, si_max, seed))

    print(f"Total simulations to run: {len(all_args)}")

    # Run in parallel
    start_time = time.time()
    with Pool(n_workers) as pool:
        results = list(pool.imap_unordered(run_single_simulation_2pt, all_args))
    elapsed = time.time() - start_time

    print(f"\nCompleted {len(results)} simulations in {elapsed:.2f} seconds")

    # Organize results
    results_mle_known = {n: [] for n in n_values}
    results_npmle = {n: [] for n in n_values}

    for r in results:
        n = r['n']
        results_mle_known[n].append(r['err_mle_known'])
        results_npmle[n].append(r['err_npmle'])

    # Compute statistics
    mle_known_means, mle_known_stds = compute_stats(results_mle_known, n_values)
    npmle_means, npmle_stds = compute_stats(results_npmle, n_values)

    # Save results
    results_data = {
        'n_values': n_values,
        'results_mle_known': results_mle_known,
        'results_npmle': results_npmle,
        'mle_known_means': mle_known_means,
        'mle_known_stds': mle_known_stds,
        'npmle_means': npmle_means,
        'npmle_stds': npmle_stds,
        'params': {
            'true_mu': true_mu,
            'prior_atoms': prior_atoms,
            'prior_weights': prior_weights,
            'si_min': si_min,
            'si_max': si_max,
            'n_sims': n_sims
        }
    }

    results_file = os.path.join(output_dir, 'experiment2_2pt_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results_data, f)
    print(f"Results saved to {results_file}")

    # Create plot
    create_plot_2pt(n_values, mle_known_means, mle_known_stds, npmle_means, npmle_stds,
                    prior_atoms, prior_weights, output_dir)

    return results_data


def create_plot_sos(n_values, median_means, median_stds, trunc_means, trunc_stds,
                    npmle_means, npmle_stds, si_min, si_max, output_dir):
    """Create plot for Subset-of-Signals experiment"""
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = {
        'median': '#1f77b4',
        'truncation': '#ff7f0e',
        'npmle': '#2ca02c'
    }

    ax.errorbar(n_values, median_means, yerr=median_stds,
                fmt='o-', color=colors['median'], linewidth=2, markersize=8,
                capsize=5, capthick=2, label='Median', elinewidth=2)

    ax.errorbar(n_values, trunc_means, yerr=trunc_stds,
                fmt='s-', color=colors['truncation'], linewidth=2, markersize=8,
                capsize=5, capthick=2, label='Iterative Truncation', elinewidth=2)

    valid_npmle = ~np.isnan(npmle_means)
    ax.errorbar(n_values[valid_npmle], npmle_means[valid_npmle],
                yerr=npmle_stds[valid_npmle],
                fmt='^-', color=colors['npmle'], linewidth=2, markersize=8,
                capsize=5, capthick=2, label='NPMLE', elinewidth=2)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Sample Size $n$', fontsize=14)
    ax.set_ylabel(r'Absolute Error $|\hat{\mu} - \mu^*|$', fontsize=14)
    ax.set_title(f'Comparison of Estimators: SoS Prior ($\\sigma_{{min}}={si_min}, \\sigma_{{max}}={si_max}$)',
                 fontsize=16)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'experiment1_sos_comparison.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {plot_file}")


def create_plot_2pt(n_values, mle_means, mle_stds, npmle_means, npmle_stds,
                    prior_atoms, prior_weights, output_dir):
    """Create plot for Two-Point Prior experiment"""
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = {
        'mle_known': '#d62728',
        'npmle': '#2ca02c'
    }

    ax.errorbar(n_values, mle_means, yerr=mle_stds,
                fmt='o-', color=colors['mle_known'], linewidth=2, markersize=8,
                capsize=5, capthick=2, label='MLE (known $G$)', elinewidth=2)

    valid_npmle = ~np.isnan(npmle_means)
    ax.errorbar(n_values[valid_npmle], npmle_means[valid_npmle],
                yerr=npmle_stds[valid_npmle],
                fmt='^-', color=colors['npmle'], linewidth=2, markersize=8,
                capsize=5, capthick=2, label='NPMLE (unknown $G$)', elinewidth=2)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Sample Size $n$', fontsize=14)
    ax.set_ylabel(r'Absolute Error $|\hat{\mu} - \mu^*|$', fontsize=14)
    ax.set_title(f'MLE vs NPMLE: Two-Point Prior ($\\sigma \\in {{{prior_atoms[0]}, {prior_atoms[1]}}}$)',
                 fontsize=14)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'experiment2_mle_vs_npmle.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {plot_file}")


def main():
    parser = argparse.ArgumentParser(description='NPMLE Tests - Parallelized')
    parser.add_argument('--n-workers', type=int, default=None,
                        help='Number of parallel workers (default: all available CPUs)')
    parser.add_argument('--n-sims', type=int, default=100,
                        help='Number of simulations per n value (default: 100)')
    parser.add_argument('--n-points', type=int, default=15,
                        help='Number of n values to test (default: 15)')
    parser.add_argument('--n-min', type=float, default=2,
                        help='Log10 of minimum n (default: 2, i.e., n=100)')
    parser.add_argument('--n-max', type=float, default=5,
                        help='Log10 of maximum n (default: 5, i.e., n=100000)')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Output directory for results (default: ./results)')
    parser.add_argument('--experiment', type=str, default='both',
                        choices=['sos', '2pt', 'both'],
                        help='Which experiment to run (default: both)')
    args = parser.parse_args()

    # Set number of workers
    if args.n_workers is None:
        n_workers = cpu_count()
    else:
        n_workers = min(args.n_workers, cpu_count())

    print(f"NPMLE Tests - Parallelized Version")
    print(f"Available CPUs: {cpu_count()}")
    print(f"Using {n_workers} workers")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Define n values
    n_values = np.logspace(args.n_min, args.n_max, args.n_points).astype(int)

    # Common parameters
    true_mu = 2.0

    # Experiment 1: Subset-of-Signals
    if args.experiment in ['sos', 'both']:
        si_min_sos = 0.7
        si_max_sos = 150
        B = 10.0
        mu0 = 1.0

        run_experiment_sos(n_values, args.n_sims, true_mu, si_min_sos, si_max_sos,
                          B, mu0, n_workers, args.output_dir)

    # Experiment 2: Two-Point Prior
    if args.experiment in ['2pt', 'both']:
        prior_atoms = np.array([1.0, 30.0])
        prior_weights = np.array([0.01, 0.99])
        si_min_2pt = 0.5
        si_max_2pt = 100.0

        run_experiment_2pt(n_values, args.n_sims, true_mu, prior_atoms, prior_weights,
                          si_min_2pt, si_max_2pt, n_workers, args.output_dir)

    print("\n" + "=" * 60)
    print("All experiments completed!")
    print(f"Results saved to {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
