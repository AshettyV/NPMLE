#!/usr/bin/env python3
"""
NPMLE Tests - Parallelized version for HPC cluster
=====================================================

This script implements and compares various mean estimation methods for heteroskedastic
data, with a focus on Non-Parametric Maximum Likelihood Estimation (NPMLE).

Overview
--------
The code implements simulations to compare the performance of different estimators when
observations X_i ~ N(μ, σ_i²) where μ is unknown and the variances σ_i² follow an
unknown distribution G. The NPMLE approach estimates both μ and G simultaneously.

Estimators Compared
-------------------
1. **Median Estimator**: Simple robust estimator
2. **Iterative Truncation**: Algorithm 1 from the paper, progressively truncates outliers
3. **Truncated Mean**: Simple clip-and-mean estimator
4. **MLE (Oracle)**: Maximum likelihood with known variances (theoretical lower bound)
5. **NPMLE**: Non-parametric MLE that estimates both location and scale mixture

Experiments
-----------
**Experiment 1 (Subset-of-Signals)**:
- Tests estimators on adversarial two-point prior with subset of signals
- Variances chosen from {σ_min, σ_max} with controlled proportions
- Evaluates robustness to heavy-tailed contamination

**Experiment 2 (Two-Point Prior)**:
- General two-point scale mixture with configurable atoms and weights
- Can simulate dynamic weights (w_1 = n^{-α}) and variances (σ_2 = n^β)
- Compares all estimators including oracle MLE

Implementation
--------------
- Parallelized using Python's multiprocessing module
- Designed for HPC clusters with many CPUs
- Generates publication-quality plots with error bars
- Saves results to pickle files for post-processing

Mathematical Background
-----------------------
For observations X_i ~ N(μ, σ_i²) where σ_i ~ G (unknown), the NPMLE finds:
    (μ̂, Ĝ) = argmax Π_i ∫ φ(X_i; μ, σ²) dG(σ)

where φ is the Gaussian PDF. The scale mixture G is estimated as a discrete distribution
with support on [σ_min, σ_max].

References
----------
Based on research in robust mean estimation under heteroskedasticity.

Author: Converted from NPMLE_py_tests.ipynb
Usage: python npmle_tests_parallel.py --help
"""

import numpy as np
import math
import pickle
import argparse
import os
import sys
import logging
from datetime import datetime
from scipy import stats, optimize as opt
from scipy.stats import norm, poisson, gaussian_kde
from scipy.optimize import root_scalar
from collections import Counter
from multiprocessing import Pool, cpu_count
from functools import partial
import time


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(output_dir, log_to_file=True):
    """
    Setup logging to both console and file.

    Creates a logger that outputs to stdout (console) with simplified timestamps
    and optionally to a file with detailed timestamps for permanent records.

    Parameters
    ----------
    output_dir : str
        Directory where log file will be saved
    log_to_file : bool, optional
        Whether to create a log file (default: True)

    Returns
    -------
    logger : logging.Logger
        Configured logger instance

    Notes
    -----
    - Console format: [HH:MM:SS] message
    - File format: [YYYY-MM-DD HH:MM:SS] message
    - Log file: {output_dir}/experiment_progress.log
    - Clears any existing handlers to avoid duplicates
    """
    # Create logger
    logger = logging.getLogger('npmle')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers

    # Console handler with simpler format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler with detailed format
    if log_to_file and output_dir:
        log_file = os.path.join(output_dir, 'experiment_progress.log')
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        logger.info(f"Logging to: {log_file}")

    return logger


# Global logger (initialized in main)
logger = None


def log(msg):
    """
    Log a message to both console and file.

    Convenience wrapper around the global logger. If logger is not initialized,
    falls back to print with timestamp.

    Parameters
    ----------
    msg : str
        Message to log
    """
    global logger
    if logger:
        logger.info(msg)
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

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
    """
    Generate data from a two-point adversarial prior.

    Creates n observations where m observations have variance σ_min² and
    (n-m) observations have variance σ_max². All observations have mean μ.
    This represents an adversarial setting where a fraction m/n of observations
    have small variance and the rest have large variance.

    Parameters
    ----------
    mu : float
        True mean for all observations
    si_min : float
        Standard deviation for small-variance observations (σ_min)
    si_max : float
        Standard deviation for large-variance observations (σ_max)
    m : int
        Number of observations with variance σ_min² (must be < n)
    n : int
        Total number of observations
    rng : numpy.random.Generator, optional
        Random number generator (default: creates new default_rng)

    Returns
    -------
    X : ndarray of shape (n,)
        Generated observations X_i ~ N(μ, σ_i²)

    Raises
    ------
    AssertionError
        If m >= n

    Notes
    -----
    Probability of σ_min: m/n
    Probability of σ_max: 1 - m/n

    Mathematical model:
        X_i | σ_i ~ N(μ, σ_i²)
        σ_i ~ Discrete({σ_min, σ_max}, {m/n, 1-m/n})
    """
    assert m < n, f"Assertion failed: m ({m}) is not less than n ({n})"
    if rng is None:
        rng = np.random.default_rng()
    scales = rng.choice([si_min, si_max], size=n, p=[(m/n), 1-(m/n)])
    return rng.normal(mu*np.ones(n), scales)


def two_point_alpha_adversarial_prior(si_min, alpha, n, rng=None):
    """
    Generate data from two-point alpha-mixture prior (Po Ling-Loh).

    WARNING: This function has a bug - parameter misalignment in the call to
    two_point_adversarial_prior. Currently unused in the codebase.

    Parameters
    ----------
    si_min : float
        Minimum standard deviation (currently unused due to bug)
    alpha : float
        Power parameter controlling variance scaling
    n : int
        Number of observations
    rng : numpy.random.Generator, optional
        Random number generator

    Returns
    -------
    X : ndarray
        Generated observations (will fail due to bug)

    Notes
    -----
    BUG: Parameters passed to two_point_adversarial_prior are misaligned:
    - Passes n as m, but requires m < n (assertion fails)
    - Passes rng as n (type error)
    - Parameter si_min is unused

    Intended design (based on Po Ling-Loh's work):
        σ_max = n^(2α)
        m = floor(log(n))
    """
    return two_point_adversarial_prior(1, n**(2*alpha), np.floor(np.log(n)), n, rng)


def sample_scale_mixture(mu, prior_atoms, prior_weights, n=1, rng=None):
    """
    Sample from a general scale mixture of Gaussians.

    Generates observations from X_i ~ N(μ, σ_i²) where σ_i is drawn from
    a discrete distribution with specified atoms and weights.

    Parameters
    ----------
    mu : float
        Mean of all Gaussian components
    prior_atoms : array-like
        Possible standard deviation values [σ_1, σ_2, ..., σ_K]
    prior_weights : array-like
        Probability weights [w_1, w_2, ..., w_K] (must sum to 1)
    n : int, optional
        Number of samples to generate (default: 1)
    rng : numpy.random.Generator, optional
        Random number generator

    Returns
    -------
    X : ndarray of shape (n,)
        Generated observations from the scale mixture

    Notes
    -----
    Mathematical model:
        X_i | σ_i ~ N(μ, σ_i²)
        P(σ_i = σ_k) = w_k for k = 1, ..., K

    This is a general framework for heteroskedastic data generation.
    """
    scales = np.asarray(prior_atoms)
    weights = np.asarray(prior_weights)
    if rng is None:
        rng = np.random.default_rng()
    comps = rng.choice(len(weights), size=n, p=weights)
    return rng.normal(mu*np.ones(n), scales[comps])


def subset_of_signals_adversarial_prior(mu, si_min, si_max, m, n, rng=None):
    """
    Generate data from a subset-of-signals adversarial prior.

    Creates observations where a fraction m/n have variances uniformly distributed
    in [σ_min, 1] (small variances) and the rest have variances in [1, σ_max]
    (large variances). This models an adversarial scenario where signal quality
    varies across observations.

    Parameters
    ----------
    mu : float
        True mean for all observations
    si_min : float
        Minimum standard deviation (must satisfy σ_min ≤ 1 ≤ σ_max)
    si_max : float
        Maximum standard deviation (must satisfy σ_min ≤ 1 ≤ σ_max)
    m : int
        Number of observations with σ_i ∈ [σ_min, 1]
    n : int
        Total number of observations
    rng : numpy.random.Generator, optional
        Random number generator

    Returns
    -------
    X : ndarray of shape (n,)
        Generated observations X_i ~ N(μ, σ_i²)

    Raises
    ------
    ValueError
        If m/n not in [0, 1] or if σ_min ≤ 1 ≤ σ_max is violated

    Notes
    -----
    Variance distribution:
        - With probability m/n: σ_i ~ Uniform[σ_min, 1]
        - With probability 1-m/n: σ_i ~ Uniform[1, σ_max]

    The value 1 serves as a threshold separating "good signals" from "bad signals".
    """
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
    """
    Generate data from a given prior distribution on variances.

    Samples variances from a discrete distribution and generates Gaussian
    observations with those variances. Returns both the observations and
    the true variances (useful for oracle estimators).

    Parameters
    ----------
    prior_atoms : array-like
        Possible standard deviation values [σ_1, ..., σ_K]
    prior_weights : array-like
        Probability weights [w_1, ..., w_K] (must sum to 1)
    mu : float
        True mean for all observations
    n : int
        Number of observations to generate
    rng : numpy.random.Generator, optional
        Random number generator

    Returns
    -------
    X : ndarray of shape (n,)
        Generated observations X_i ~ N(μ, σ_i²)
    sig : ndarray of shape (n,)
        True standard deviations σ_i for each observation

    Notes
    -----
    This function is useful when you need to know the ground truth variances,
    for example to compute oracle MLE estimators that assume known σ_i.
    """
    if rng is None:
        rng = np.random.default_rng()
    sig = rng.choice(prior_atoms, size=n, p=prior_weights)
    return rng.normal(mu*np.ones(n), sig), sig


# =============================================================================
# ESTIMATOR FUNCTIONS
# =============================================================================

def median_estimator(N_obs):
    """
    Median estimator for location.

    The median is a robust estimator that achieves sub-optimal but reasonable
    performance under heavy-tailed distributions. It is used as a baseline
    for comparison.

    Parameters
    ----------
    N_obs : array-like
        Observations

    Returns
    -------
    float
        Sample median

    Notes
    -----
    For symmetric distributions, the median estimates the mean.
    Under heteroskedasticity, the median may not be optimal but is robust.
    Convergence rate: O(1/√n) for well-behaved distributions.
    """
    return np.median(N_obs)


def phi_interval(x, a, b):
    """
    Truncation function: clips values to interval [a, b].

    This is the projection operator onto [a, b]:
        φ(x; [a,b]) = a     if x < a
                      x     if a ≤ x ≤ b
                      b     if x > b

    Parameters
    ----------
    x : array-like or float
        Values to truncate
    a : float
        Lower bound
    b : float
        Upper bound

    Returns
    -------
    array-like or float
        Truncated values

    Notes
    -----
    Used in iterative truncation algorithm to progressively remove outliers.
    """
    return np.clip(x, a, b)


def iterative_truncation_estimator(N_obs, B, mu0, mu_star=None, m=1.0, n=None, return_path=False):
    """
    Mean estimation via iterative truncation (Algorithm 1).

    Progressively truncates observations to remove outliers, updating the mean
    estimate iteratively. The truncation window shrinks by half at each stage
    (K stages total), and within each stage, the algorithm runs T iterations
    to converge to a stable estimate.

    Parameters
    ----------
    N_obs : array-like
        Observations X_1, ..., X_n
    B : float
        Initial truncation radius (typically related to max expected deviation)
    mu0 : float
        Initial mean estimate (starting point)
    mu_star : float, optional
        True mean (unused, kept for API compatibility)
    m : float, optional
        Parameter controlling number of iterations T = ⌈64n log(n) / m⌉ (default: 1.0)
    n : int, optional
        Sample size (default: inferred from N_obs)
    return_path : bool, optional
        If True, return history of all intermediate estimates (default: False)

    Returns
    -------
    mu_hat : float
        Final mean estimate
    history : list of float, optional
        If return_path=True, list of all intermediate estimates

    Algorithm
    ---------
    1. Set δ_0 = B, K = ⌈log₂(B)⌉ if B > 1, else K = 0
    2. Set T = max(1, ⌈64n log(n) / m⌉)
    3. For k = 0, ..., K:
        a. For t = 1, ..., T:
            - Truncate: X'_i = φ(X_i; [μ_t - δ_k, μ_t + δ_k])
            - Update: μ_{t+1} = mean(X')
        b. Set δ_{k+1} = δ_k / 2

    Notes
    -----
    The algorithm progressively removes outliers by:
    - Starting with a large truncation window (±B)
    - Iteratively recomputing the mean of truncated values
    - Halving the window at each stage to focus on the center

    The number of inner iterations T ensures convergence at each stage.
    """
    # Convert observations to numpy array
    X = np.asarray(N_obs, dtype=float)
    if n is None:
        n = X.shape[0]
    else:
        n = int(n)
        if n != X.shape[0]:
            raise ValueError("Provided n does not match len(X).")

    # Initialize truncation radius and starting estimate
    delta = float(B)
    mu_k0 = float(mu0)

    # Compute number of stages: reduce radius by half each stage
    if delta <= 1.0:
        K = 0  # Already at fine scale
    else:
        K = int(math.ceil(math.log2(delta)))  # Number of halvings needed

    # Compute iterations per stage (ensures convergence at each scale)
    T = max(1, int(math.ceil(64.0 * n * math.log(n) / m)))

    # Outer loop: progressively shrink truncation window
    history = []
    for k in range(K + 1):
        mu_t = mu_k0
        # Inner loop: converge at current scale
        for _ in range(T + 1):
            # Truncate observations to [μ_t - δ, μ_t + δ]
            a = mu_t - delta
            b = mu_t + delta
            truncated = phi_interval(X, a, b)
            # Update estimate as mean of truncated values
            mu_t = truncated.mean()
            if return_path:
                history.append(mu_t)
        # Use converged estimate as starting point for next stage
        mu_k0 = mu_t
        # Halve the truncation radius
        delta = delta / 2.0

    mu_hat = mu_t
    if return_path:
        return mu_hat, history
    return mu_hat


def oracle_estimator(N_obs, N_vars):
    """
    MLE estimator under oracle knowledge of variances.

    WARNING: This function contains a bug. Use mle_known_sigma() instead.

    When variances σ_i² are known, the MLE for μ is:
        μ̂ = Σ(X_i/σ_i²) / Σ(1/σ_i²)

    BUG: Line using np.linalg.norm() should be np.sum() instead.
    Current code computes: norm = √Σ(1/σ_i⁴) ← WRONG
    Should compute: sum = Σ(1/σ_i²) ← CORRECT

    Parameters
    ----------
    N_obs : array-like
        Observations X_i
    N_vars : array-like
        True variances σ_i²

    Returns
    -------
    float
        Incorrect MLE estimate (due to bug)

    Notes
    -----
    This function is unused in the codebase. Use mle_known_sigma() for correct results.
    """
    N_inv_vars = 1 / N_vars
    dot_prod = np.dot(N_inv_vars, N_obs)
    norm_inv_Y = np.linalg.norm(N_inv_vars)  # BUG: Should be np.sum()
    return dot_prod / norm_inv_Y


def mle_known_sigma(X, sigma):
    """
    MLE estimator for μ when variances σ_i² are known (Oracle estimator).

    This is the optimal estimator when the true variances are known. It provides
    a theoretical lower bound on the estimation error that adaptive estimators
    (like NPMLE) aim to approach.

    Parameters
    ----------
    X : array-like
        Observations X_i ~ N(μ, σ_i²)
    sigma : array-like
        True standard deviations σ_i

    Returns
    -------
    mu_hat : float
        Weighted MLE estimate

    Formula
    -------
    μ̂_MLE = (Σ X_i/σ_i²) / (Σ 1/σ_i²)

    This is the minimum variance unbiased estimator (MVUE) under known variances.

    Notes
    -----
    - Achieves the Cramér-Rao lower bound
    - Convergence rate: O(1/√n) with optimal constant
    - Used as oracle baseline in experiments to measure information loss
      from not knowing the true variances
    """
    X = np.asarray(X, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    weights = 1.0 / (sigma ** 2)
    mu_hat = np.sum(X * weights) / np.sum(weights)
    return mu_hat


# =============================================================================
# NPMLE FUNCTIONS
# =============================================================================

def mixture_density(prior_atoms, prior_weights, mu, obs):
    """
    Compute mixture density for scale mixture of Gaussians.

    Evaluates the marginal density:
        f(x | μ, G) = Σ_k w_k φ(x; μ, σ_k²)

    where φ is the Gaussian PDF and G is the discrete distribution on {σ_k}.

    Parameters
    ----------
    prior_atoms : array-like
        Standard deviation values [σ_1, ..., σ_K]
    prior_weights : array-like
        Mixture weights [w_1, ..., w_K]
    mu : float
        Location parameter
    obs : array-like
        Observations at which to evaluate density

    Returns
    -------
    densities : ndarray
        Mixture density evaluated at each observation

    Notes
    -----
    This is the marginal likelihood of observations under the scale mixture model.
    Used in EM algorithm and likelihood computation.
    """
    k = len(prior_atoms)
    densities = np.zeros(len(obs))
    for i in range(k):
        densities += prior_weights[i] * norm.pdf(obs, loc=mu, scale=prior_atoms[i])
    return densities


def avg_loglikelihood(prior_atoms, prior_weights, mu, N_obs):
    """
    Compute average log-likelihood for scale mixture model.

    Evaluates:
        L(μ, G) = (1/n) Σ_i log f(X_i | μ, G)

    where f is the mixture density.

    Parameters
    ----------
    prior_atoms : array-like
        Standard deviation values [σ_1, ..., σ_K]
    prior_weights : array-like
        Mixture weights [w_1, ..., w_K]
    mu : float
        Location parameter
    N_obs : array-like
        Observations X_1, ..., X_n

    Returns
    -------
    float
        Average log-likelihood

    Notes
    -----
    Used to monitor convergence in NPMLE algorithms and compare model fits.
    Numerical floor of 1e-300 prevents log(0) errors.
    """
    densities = mixture_density(prior_atoms, prior_weights, mu, N_obs)
    densities = np.maximum(densities, 1e-300)
    return np.mean(np.log(densities))


def DG_sigma(prior_atoms, prior_weights, sigma, mu, N_obs, denom=None):
    """
    Compute directional derivative for scale mixture NPMLE.

    This is the key quantity for determining whether to add a new atom σ
    to the support of the estimated distribution G. If DG(σ) > 0, adding
    atom σ would increase the likelihood.

    Parameters
    ----------
    prior_atoms : array-like
        Current atoms in the support of G
    prior_weights : array-like
        Current weights
    sigma : float
        Candidate atom to evaluate
    mu : float
        Current location estimate
    N_obs : array-like
        Observations
    denom : ndarray, optional
        Pre-computed mixture density (for efficiency)

    Returns
    -------
    float
        Directional derivative DG(σ)

    Formula
    -------
    DG(σ) = (1/n) Σ_i [φ(X_i; μ, σ²) / f(X_i | μ, G)] - 1

    Notes
    -----
    - If DG(σ) > 0: Adding weight at σ increases likelihood
    - If DG(σ) ≤ 0: σ should not be added to support
    - Used in vertex direction method for NPMLE computation
    """
    n = len(N_obs)
    if denom is None:
        denom = mixture_density(prior_atoms, prior_weights, mu, N_obs)
    denom = np.maximum(denom, 1e-300)
    numer = norm.pdf(N_obs, loc=mu, scale=sigma)
    return (1.0/n) * np.sum(numer/denom) - 1.0


def grad_DG_sigma(prior_atoms, prior_weights, sigma, mu, N_obs, denom=None):
    """
    Compute gradient of directional derivative with respect to σ.

    This gradient is used to optimize the choice of new atom σ* when adding
    atoms to the support of G. We seek σ* that maximizes DG(σ).

    Parameters
    ----------
    prior_atoms : array-like
        Current atoms in support of G
    prior_weights : array-like
        Current weights
    sigma : float
        Point at which to evaluate gradient
    mu : float
        Current location estimate
    N_obs : array-like
        Observations
    denom : ndarray, optional
        Pre-computed mixture density (for efficiency)

    Returns
    -------
    float
        Gradient ∂DG/∂σ evaluated at σ

    Notes
    -----
    Uses the derivative of Gaussian PDF with respect to scale parameter:
        ∂φ/∂σ = φ(z) * (z² - 1) / σ - φ(z) / σ
    where z = (x - μ) / σ

    Used in gradient-based optimization to find optimal new atom.
    """
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
    """
    Find optimal new atom to add to support of G.

    Searches for σ* ∈ [σ_min, σ_max] that maximizes the directional derivative
    DG(σ). If DG(σ*) > 0, adding this atom will increase the likelihood.

    Parameters
    ----------
    atoms : array-like
        Current atoms in support of G
    weights : array-like
        Current weights for atoms
    mu : float
        Current location estimate
    N_obs : array-like
        Observations
    si_min : float
        Minimum allowed standard deviation
    si_max : float
        Maximum allowed standard deviation

    Returns
    -------
    s_star : float
        Optimal atom σ* that maximizes DG(σ)
    val : float
        Value of DG(σ*) - if > 0, this atom should be added

    Algorithm
    ---------
    Uses scipy.optimize.minimize_scalar with bounded method to find:
        σ* = argmax_{σ ∈ [σ_min, σ_max]} DG(σ)

    Falls back to midpoint if optimization fails.

    Notes
    -----
    This is a key step in the vertex direction method for computing NPMLE.
    New atoms are added iteratively until no improvement is possible.
    """
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
    """
    EM algorithm for computing NPMLE weights given fixed atoms.

    When the atoms (support points) of G are fixed, the optimal weights can be
    computed via Expectation-Maximization (EM). This is a convex optimization
    problem.

    Parameters
    ----------
    x : array-like
        Observations
    scales : array-like
        Fixed standard deviation atoms [σ_1, ..., σ_K]
    mu : float, optional
        Location parameter (default: 0.0)
    w_init : array-like, optional
        Initial weights (default: uniform weights)
    max_iter : int, optional
        Maximum EM iterations (default: 500)
    tol : float, optional
        Convergence tolerance on weight changes (default: 1e-8)

    Returns
    -------
    w : ndarray of shape (K,)
        Optimal mixture weights that maximize likelihood

    Algorithm
    ---------
    E-step: Compute responsibilities
        r_{ik} = w_k φ(x_i; μ, σ_k²) / Σ_j w_j φ(x_i; μ, σ_j²)

    M-step: Update weights
        w_k = (1/n) Σ_i r_{ik}

    Notes
    -----
    - Converges to global optimum (convex problem)
    - Numerical stabilization: floor values at 1e-12, 1e-300
    - Weights automatically sum to 1
    """
    # Convert inputs to arrays
    x = np.asarray(x, dtype=float)
    scales = np.asarray(scales, dtype=float)
    n = x.size  # Number of observations
    K = scales.size  # Number of mixture components

    # Initialize weights
    if w_init is None:
        w = np.ones(K) / K  # Uniform initialization
    else:
        w = np.asarray(w_init, dtype=float).copy()
        w = np.maximum(w, 1e-12)  # Prevent exact zeros
        w /= w.sum()  # Normalize to sum to 1

    # Pre-compute Gaussian densities φ(x_i; μ, σ_k²) for all i, k
    # This is the key computational bottleneck, so we do it once
    phi = np.zeros((n, K))
    for j in range(K):
        phi[:, j] = norm.pdf(x, loc=mu, scale=scales[j])

    # EM iterations
    for _ in range(max_iter):
        # E-step: Compute mixture density f(x_i) = Σ_k w_k φ_k(x_i)
        f = phi @ w  # Matrix-vector product (n,K) @ (K,) = (n,)
        f = np.maximum(f, 1e-300)  # Numerical floor to prevent division by zero

        # E-step: Compute responsibilities r_ik = w_k φ_k(x_i) / f(x_i)
        resp = phi * w[None, :]  # Broadcasting: (n,K) * (1,K) = (n,K)
        resp /= f[:, None]  # Normalize: (n,K) / (n,1) = (n,K)

        # M-step: Update weights w_k = (1/n) Σ_i r_ik
        w_new = resp.mean(axis=0)  # Average over observations
        w_new = np.maximum(w_new, 1e-12)  # Numerical floor
        w_new /= w_new.sum()  # Normalize to sum to 1

        # Check convergence: max absolute change in weights
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break  # Converged
        w = w_new

    return w


def merge_atoms_weights(atoms, weights, atom_tol=1e-3):
    """
    Merge atoms that are too close together.

    When NPMLE adds multiple atoms, some may end up very close to each other.
    This function merges nearby atoms to create a cleaner representation and
    avoid numerical issues.

    Parameters
    ----------
    atoms : array-like
        Standard deviation values [σ_1, ..., σ_K]
    weights : array-like
        Corresponding weights [w_1, ..., w_K]
    atom_tol : float, optional
        Distance threshold for merging (default: 1e-3)

    Returns
    -------
    merged_atoms : ndarray
        Merged atoms (sorted)
    merged_weights : ndarray
        Corresponding merged weights (sum of merged components)

    Algorithm
    ---------
    1. Sort atoms in ascending order
    2. Iterate through sorted atoms
    3. If |σ_i - σ_{i-1}| < tol: merge by adding weights
    4. Otherwise: keep as separate atom

    Notes
    -----
    Preserves total weight: sum(merged_weights) = sum(weights)
    """
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
    """
    Compute NPMLE for scale mixture G when location μ is fixed.

    Estimates the mixing distribution G on [σ_min, σ_max] that maximizes
    the likelihood, given a fixed location parameter μ. Uses vertex direction
    method: alternates between optimizing weights (EM) and adding new atoms.

    Parameters
    ----------
    N_obs : array-like
        Observations X_1, ..., X_n
    mu : float
        Fixed location parameter
    si_min : float
        Minimum standard deviation in support of G
    si_max : float
        Maximum standard deviation in support of G
    atoms_init : array-like, optional
        Initial atoms (default: [std(X)])
    weights_init : array-like, optional
        Initial weights (default: uniform)
    weight_tol : float, optional
        Tolerance for pruning near-zero weights (default: 1e-6)
    atom_tol : float, optional
        Tolerance for merging close atoms (default: 1e-6)
    progress_tol : float, optional
        Convergence tolerance on log-likelihood change (default: 1e-6)
    iter_max : int, optional
        Maximum iterations (default: 20)
    verbose : bool, optional
        Print progress (default: False)

    Returns
    -------
    atoms : ndarray
        Standard deviations in support of Ĝ
    weights : ndarray
        Corresponding mixture weights

    Algorithm
    ---------
    1. Initialize with single atom (or provided atoms)
    2. Repeat until convergence:
        a. Update weights via EM (fixed atoms)
        b. Prune atoms with weight < weight_tol
        c. Find optimal new atom σ* = argmax DG(σ)
        d. If DG(σ*) > threshold: add σ* to support and recompute weights
        e. Check convergence via log-likelihood change
    3. Merge close atoms for clean representation

    Notes
    -----
    This implements the vertex direction method (Bohning 1985, Lindsay 1983).
    The estimated G is a discrete distribution (finite support).
    """
    n = len(N_obs)

    # Initialize atoms: start with single atom at empirical std
    if atoms_init is None:
        atoms = np.array([max(si_min, np.std(N_obs))])
    else:
        atoms = np.asarray(atoms_init).copy()

    # Initialize weights: uniform over atoms
    if weights_init is None:
        weights = np.ones(len(atoms)) / len(atoms)
    else:
        weights = np.asarray(weights_init).copy()

    prev_ll = -np.inf

    # Main NPMLE loop: alternate between weight and atom updates
    for it in range(iter_max):
        # Step 1: Update weights for current atoms via EM
        weights = get_weights_npmle_EM(N_obs, atoms, mu=mu, w_init=weights)

        # Step 2: Prune atoms with negligible weight
        mask = weights > weight_tol
        atoms = atoms[mask]
        weights = weights[mask]
        weights /= weights.sum()  # Renormalize after pruning

        # Safety: ensure we always have at least one atom
        if len(atoms) == 0:
            atoms = np.array([max(si_min, np.std(N_obs))])
            weights = np.array([1.0])

        # Step 3: Search for optimal new atom to add
        s_star, dg_val = update_atoms(atoms, weights, mu, N_obs, si_min, si_max)

        # Step 4: If directional derivative > 0, adding this atom improves likelihood
        if dg_val > 1e-6:
            atoms = np.append(atoms, s_star)
            weights = np.append(weights, 0.0)  # Initialize with zero weight
            # Recompute all weights with new atom
            weights = get_weights_npmle_EM(N_obs, atoms, mu=mu, w_init=weights)

        # Step 5: Check convergence via log-likelihood
        ll = avg_loglikelihood(atoms, weights, mu, N_obs)
        if abs(ll - prev_ll) < progress_tol:
            break  # Converged: likelihood not improving
        prev_ll = ll

    # Post-processing: merge atoms that are too close together
    atoms, weights = merge_atoms_weights(atoms, weights, atom_tol)

    return atoms, weights


def mle_location_scale_mixture(N_obs, atoms, weights, mu_init=None):
    """
    Compute MLE for location μ given fixed scale mixture G.

    When the mixing distribution G (atoms and weights) is fixed, this computes
    the optimal location parameter μ that maximizes the likelihood.

    Parameters
    ----------
    N_obs : array-like
        Observations X_1, ..., X_n
    atoms : array-like
        Standard deviation atoms of G
    weights : array-like
        Mixture weights
    mu_init : float, optional
        Initial guess for μ (default: median of observations)

    Returns
    -------
    mu_hat : float
        MLE estimate of location parameter

    Algorithm
    ---------
    Uses BFGS optimization to minimize negative log-likelihood:
        -L(μ) = -(1/n) Σ_i log[Σ_k w_k φ(X_i; μ, σ_k²)]

    Gradient is computed analytically for faster convergence.

    Notes
    -----
    Falls back to median if optimization fails.
    The problem is generally well-conditioned for reasonable data.
    """
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
    """
    Full NPMLE with alternating optimization for μ and G.

    Jointly estimates the location parameter μ and the scale mixture distribution G
    by alternating between optimizing each while holding the other fixed. This is
    the main NPMLE estimator used in experiments.

    Parameters
    ----------
    N_obs : array-like
        Observations X_1, ..., X_n
    si_min : float, optional
        Minimum standard deviation (default: 0, adjusted to 0.01)
    si_max : float, optional
        Maximum standard deviation (default: inf, adjusted to 10*std(X))
    verbose : bool, optional
        Print iteration progress (default: False)
    iter_max : int, optional
        Maximum outer iterations (default: 20)

    Returns
    -------
    atoms : ndarray
        Estimated standard deviations in support of Ĝ
    weights : ndarray
        Estimated mixture weights
    mu : float
        Estimated location parameter

    Algorithm
    ---------
    1. Initialize: μ = median(X), G = single atom at std(X)
    2. Repeat until convergence:
        a. Update G given fixed μ (via npmle_heteroskedastic)
        b. Update μ given fixed G (via mle_location_scale_mixture)
        c. Compute log-likelihood
        d. Check convergence: |LL_new - LL_old| < tol
    3. Return (Ĝ, μ̂)

    Notes
    -----
    - This is a block coordinate ascent algorithm on (μ, G)
    - Each step increases (or maintains) the likelihood
    - Typically converges in 5-15 iterations
    - Provides adaptive estimation without knowing true variances

    Theoretical Properties
    ----------------------
    Under regularity conditions, the NPMLE achieves:
    - Consistency: (μ̂, Ĝ) → (μ*, G*) as n → ∞
    - Near-parametric rate: |μ̂ - μ*| = O(1/√n) in favorable cases
    - Adaptivity: No need to know G or choose tuning parameters
    """
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
    """
    Run a single simulation for Subset-of-Signals experiment.

    Worker function for parallel execution. Generates data from subset-of-signals
    prior and evaluates three estimators: median, iterative truncation, and NPMLE.

    Parameters
    ----------
    args : tuple
        (s, n, true_mu, si_min, si_max, B, mu0, m_power, seed) where:
        - s: simulation index
        - n: sample size
        - true_mu: true mean parameter
        - si_min: minimum standard deviation
        - si_max: maximum standard deviation
        - B: truncation bound
        - mu0: initial mean estimate
        - m_power: power for computing m = n^m_power
        - seed: random seed

    Returns
    -------
    dict
        Results containing:
        - n: sample size
        - sim: simulation index
        - err_median: absolute error of median estimator
        - err_trunc: absolute error of truncation estimator
        - err_npmle: absolute error of NPMLE (or nan if failed)

    Notes
    -----
    Designed for use with multiprocessing.Pool.map/imap.
    Each worker uses independent random seed for reproducibility.
    """
    s, n, true_mu, si_min, si_max, B, mu0, m_power, seed = args
    rng = np.random.default_rng(seed)

    m = max(1, int(n ** m_power))

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
    """
    Run a single simulation for Two-Point Prior experiment.

    Worker function for parallel execution. Generates data from two-point prior
    and evaluates four estimators: MLE (oracle), median, truncated mean, and NPMLE.

    Parameters
    ----------
    args : tuple
        (s, n, true_mu, prior_atoms, prior_weights, weight_power, sigma_power,
         si_min, si_max, B, seed) where:
        - s: simulation index
        - n: sample size
        - true_mu: true mean parameter
        - prior_atoms: [σ_1, σ_2] base atoms
        - prior_weights: [w_1, w_2] base weights
        - weight_power: if > 0, use dynamic weights w_1 = n^(-weight_power)
        - sigma_power: if > 0, use dynamic variance σ_2 = n^(sigma_power)
        - si_min: minimum σ for NPMLE search
        - si_max: maximum σ for NPMLE search
        - B: truncation bound
        - seed: random seed

    Returns
    -------
    dict
        Results containing:
        - n: sample size
        - sim: simulation index
        - err_mle_known: absolute error of oracle MLE (known variances)
        - err_median: absolute error of median estimator
        - err_trunc: absolute error of truncated mean
        - err_npmle: absolute error of NPMLE (or nan if failed)

    Notes
    -----
    Dynamic parameters allow testing scenarios where:
    - Weights vary with n: models rare contamination (w_1 = n^(-α) → 0)
    - Variances scale with n: models growing heterogeneity (σ_2 = n^β)

    Oracle MLE provides theoretical lower bound on achievable error.
    """
    s, n, true_mu, prior_atoms, prior_weights, weight_power, sigma_power, si_min, si_max, B, seed = args
    rng = np.random.default_rng(seed)

    # Compute dynamic weights if weight_power > 0
    if weight_power > 0:
        w1 = n ** (-weight_power)
        w1 = min(max(w1, 1e-10), 1 - 1e-10)  # Clamp to valid range
        weights = np.array([w1, 1 - w1])
    else:
        weights = prior_weights

    # Compute dynamic variance atoms if sigma_power > 0
    # sigma_1 stays fixed, sigma_2 scales as n^(sigma_power)
    if sigma_power > 0:
        atoms = np.array([prior_atoms[0], n ** sigma_power])
    else:
        atoms = prior_atoms

    # Generate data from two-point prior
    X, true_sigmas = generate_data_from_prior(atoms, weights, true_mu, n, rng=rng)

    # 1. MLE with known G (known sigmas) - Oracle estimator
    mu_hat_mle_known = mle_known_sigma(X, true_sigmas)
    err_mle_known = np.abs(mu_hat_mle_known - true_mu)

    # 2. Median estimator
    mu_hat_median = median_estimator(X)
    err_median = np.abs(mu_hat_median - true_mu)

    # 3. Truncated mean estimator (simple clip-and-mean)
    mu_hat_trunc = np.mean(np.clip(X, -B, B))
    err_trunc = np.abs(mu_hat_trunc - true_mu)

    # 4. NPMLE estimator (unknown G)
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
        'err_median': err_median,
        'err_trunc': err_trunc,
        'err_npmle': err_npmle
    }


def compute_stats(results_dict, n_values):
    """
    Compute mean and standard error for each estimator at each n.

    Aggregates simulation results to compute summary statistics across
    multiple simulation runs at each sample size.

    Parameters
    ----------
    results_dict : dict
        Dictionary mapping n -> list of error values
        e.g., {100: [0.1, 0.15, 0.12, ...], 1000: [...], ...}
    n_values : array-like
        Sample sizes tested

    Returns
    -------
    means : ndarray
        Mean error at each n (length = len(n_values))
    stds : ndarray
        Standard error of mean at each n (std / sqrt(num_sims))

    Notes
    -----
    - Automatically filters out nan values before computing statistics
    - If all values are nan for a given n, returns nan for that entry
    - Standard error (not standard deviation) is returned for error bars
    """
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


def run_experiment_sos(n_values, n_sims, true_mu, si_min, si_max, B, mu0, m_power, n_workers, output_dir):
    """
    Run Subset-of-Signals experiment with parallel processing.

    Orchestrates a full simulation experiment comparing estimators on adversarial
    subset-of-signals prior. Generates data in parallel, aggregates results, and
    creates publication-quality plots.

    Parameters
    ----------
    n_values : array-like
        Sample sizes to test [n_1, n_2, ..., n_k]
    n_sims : int
        Number of simulation runs per sample size
    true_mu : float
        True mean parameter
    si_min : float
        Minimum standard deviation (for "good" observations)
    si_max : float
        Maximum standard deviation (for "bad" observations)
    B : float
        Truncation bound for iterative truncation estimator
    mu0 : float
        Initial mean estimate for truncation algorithm
    m_power : float
        Power for computing m = n^m_power (fraction of good observations)
    n_workers : int
        Number of parallel workers
    output_dir : str
        Directory for saving results and plots

    Returns
    -------
    results_data : dict
        Complete results dictionary containing:
        - n_values: tested sample sizes
        - results_median/truncation/npmle: raw error values
        - median/trunc/npmle_means: mean errors at each n
        - median/trunc/npmle_stds: standard errors at each n
        - params: experiment parameters

    Side Effects
    ------------
    - Saves results to {output_dir}/experiment1_sos_results.pkl
    - Saves plot to {output_dir}/experiment1_sos_comparison.png
    - Logs progress to console and log file

    Performance
    -----------
    For n_values with k points and n_sims simulations each:
    - Total simulations: k × n_sims
    - Parallel speedup: near-linear up to number of physical cores
    - Typical rate: 10-100 simulations/second depending on n

    Notes
    -----
    This experiment tests robustness to adversarial contamination where
    a fraction 1 - m/n of observations have very large variance.
    """
    log("=" * 60)
    log("EXPERIMENT 1: Subset-of-Signals Prior")
    log("=" * 60)
    log(f"n values: {n_values}")
    log(f"Simulations per n: {n_sims}")
    log(f"Using {n_workers} workers")
    log(f"Parameters: si_min={si_min}, si_max={si_max}, B={B}, mu0={mu0}, m_power={m_power}")
    log("=" * 60)

    # Prepare all simulation arguments
    all_args = []
    base_seed = 42
    for n in n_values:
        for s in range(n_sims):
            seed = base_seed + n * n_sims + s
            all_args.append((s, n, true_mu, si_min, si_max, B, mu0, m_power, seed))

    total_sims = len(all_args)
    log(f"Total simulations to run: {total_sims}")

    # Run in parallel with progress tracking
    start_time = time.time()
    results = []
    completed = 0
    last_progress = 0

    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(run_single_simulation_sos, all_args):
            results.append(result)
            completed += 1

            # Log progress every 10% or at key milestones
            progress = int(100 * completed / total_sims)
            if progress >= last_progress + 10 or completed == total_sims:
                elapsed_so_far = time.time() - start_time
                rate = completed / elapsed_so_far if elapsed_so_far > 0 else 0
                eta = (total_sims - completed) / rate if rate > 0 else 0
                log(f"Progress: {completed}/{total_sims} ({progress}%) | "
                    f"Rate: {rate:.1f} sims/sec | ETA: {eta:.0f}s")
                last_progress = progress

    elapsed = time.time() - start_time
    log(f"Completed {len(results)} simulations in {elapsed:.2f} seconds ({len(results)/elapsed:.1f} sims/sec)")

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
            'm_power': m_power,
            'n_sims': n_sims
        }
    }

    results_file = os.path.join(output_dir, 'experiment1_sos_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results_data, f)
    log(f"Results saved to {results_file}")

    # Log summary statistics
    log(f"Summary - Mean errors at largest n={n_values[-1]}:")
    log(f"  Median: {median_means[-1]:.6f} +/- {median_stds[-1]:.6f}")
    log(f"  Truncation: {trunc_means[-1]:.6f} +/- {trunc_stds[-1]:.6f}")
    log(f"  NPMLE: {npmle_means[-1]:.6f} +/- {npmle_stds[-1]:.6f}")

    # Create plot
    create_plot_sos(n_values, median_means, median_stds, trunc_means, trunc_stds,
                    npmle_means, npmle_stds, si_min, si_max, B, mu0, m_power, true_mu, n_sims, output_dir)

    return results_data


def run_experiment_2pt(n_values, n_sims, true_mu, prior_atoms, prior_weights,
                       weight_power, sigma_power, si_min, si_max, B, n_workers, output_dir):
    """
    Run Two-Point Prior experiment with parallel processing.

    Orchestrates simulation experiment comparing all estimators on two-point
    scale mixture. Supports dynamic weights and variances that scale with n.
    Includes oracle MLE as theoretical baseline.

    Parameters
    ----------
    n_values : array-like
        Sample sizes to test
    n_sims : int
        Number of simulations per sample size
    true_mu : float
        True mean parameter
    prior_atoms : array-like
        Base standard deviation atoms [σ_1, σ_2]
    prior_weights : array-like
        Base mixture weights [w_1, w_2]
    weight_power : float
        If > 0: use dynamic weights w_1 = n^(-weight_power)
        If = 0: use fixed weights from prior_weights
    sigma_power : float
        If > 0: use dynamic variance σ_2 = n^(sigma_power)
        If = 0: use fixed atoms from prior_atoms
    si_min : float
        Minimum σ for NPMLE search range
    si_max : float
        Maximum σ for NPMLE search range
    B : float
        Truncation bound for truncated mean estimator
    n_workers : int
        Number of parallel workers
    output_dir : str
        Directory for saving results and plots

    Returns
    -------
    results_data : dict
        Complete results dictionary containing:
        - n_values: tested sample sizes
        - results_mle_known/median/trunc/npmle: raw error values
        - mle_known/median/trunc/npmle_means: mean errors
        - mle_known/median/trunc/npmle_stds: standard errors
        - params: experiment parameters

    Side Effects
    ------------
    - Saves results to {output_dir}/experiment2_2pt_results.pkl
    - Saves 4 plots:
        * experiment2_all_estimators.png: All estimators comparison
        * experiment2_mle_vs_npmle.png: Oracle vs NPMLE only
        * experiment2_error_ratio.png: Ratios relative to oracle
        * experiment2_combined.png: 2-panel combined figure

    Notes
    -----
    Dynamic parameters model important scenarios:
    - Rare contamination: w_1 = n^(-α) → 0 as n → ∞
    - Growing heterogeneity: σ_2 = n^β → ∞ as n → ∞

    Oracle MLE (known variances) provides Cramér-Rao lower bound.
    Goal is for NPMLE to approach oracle performance adaptively.
    """
    log("\n" + "=" * 60)
    log("EXPERIMENT 2: Two-Point Prior (All Estimators)")
    log("=" * 60)
    log(f"Prior atoms: {prior_atoms}, weights: {prior_weights}")
    if weight_power > 0:
        log(f"Dynamic weights: w1 = n^(-{weight_power}), w2 = 1 - w1")
    if sigma_power > 0:
        log(f"Dynamic variance: sigma_1 = {prior_atoms[0]}, sigma_2 = n^({sigma_power})")
    log(f"n values: {n_values}")
    log(f"Simulations per n: {n_sims}")
    log(f"Using {n_workers} workers")
    log(f"NPMLE search range: [{si_min}, {si_max}]")
    log(f"Truncation bound B: {B}")
    log("=" * 60)

    # Prepare all simulation arguments
    all_args = []
    base_seed = 10000
    for n in n_values:
        for s in range(n_sims):
            seed = base_seed + n * n_sims + s
            all_args.append((s, n, true_mu, prior_atoms, prior_weights, weight_power, sigma_power, si_min, si_max, B, seed))

    total_sims = len(all_args)
    log(f"Total simulations to run: {total_sims}")

    # Run in parallel with progress tracking
    start_time = time.time()
    results = []
    completed = 0
    last_progress = 0

    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(run_single_simulation_2pt, all_args):
            results.append(result)
            completed += 1

            # Log progress every 10% or at key milestones
            progress = int(100 * completed / total_sims)
            if progress >= last_progress + 10 or completed == total_sims:
                elapsed_so_far = time.time() - start_time
                rate = completed / elapsed_so_far if elapsed_so_far > 0 else 0
                eta = (total_sims - completed) / rate if rate > 0 else 0
                log(f"Progress: {completed}/{total_sims} ({progress}%) | "
                    f"Rate: {rate:.1f} sims/sec | ETA: {eta:.0f}s")
                last_progress = progress

    elapsed = time.time() - start_time
    log(f"Completed {len(results)} simulations in {elapsed:.2f} seconds ({len(results)/elapsed:.1f} sims/sec)")

    # Organize results
    results_mle_known = {n: [] for n in n_values}
    results_median = {n: [] for n in n_values}
    results_trunc = {n: [] for n in n_values}
    results_npmle = {n: [] for n in n_values}

    for r in results:
        n = r['n']
        results_mle_known[n].append(r['err_mle_known'])
        results_median[n].append(r['err_median'])
        results_trunc[n].append(r['err_trunc'])
        results_npmle[n].append(r['err_npmle'])

    # Compute statistics
    mle_known_means, mle_known_stds = compute_stats(results_mle_known, n_values)
    median_means, median_stds = compute_stats(results_median, n_values)
    trunc_means, trunc_stds = compute_stats(results_trunc, n_values)
    npmle_means, npmle_stds = compute_stats(results_npmle, n_values)

    # Save results
    results_data = {
        'n_values': n_values,
        'results_mle_known': results_mle_known,
        'results_median': results_median,
        'results_trunc': results_trunc,
        'results_npmle': results_npmle,
        'mle_known_means': mle_known_means,
        'mle_known_stds': mle_known_stds,
        'median_means': median_means,
        'median_stds': median_stds,
        'trunc_means': trunc_means,
        'trunc_stds': trunc_stds,
        'npmle_means': npmle_means,
        'npmle_stds': npmle_stds,
        'params': {
            'true_mu': true_mu,
            'prior_atoms': prior_atoms,
            'prior_weights': prior_weights,
            'weight_power': weight_power,
            'sigma_power': sigma_power,
            'si_min': si_min,
            'si_max': si_max,
            'B': B,
            'n_sims': n_sims
        }
    }

    results_file = os.path.join(output_dir, 'experiment2_2pt_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results_data, f)
    log(f"Results saved to {results_file}")

    # Log summary statistics
    log(f"Summary - Mean errors at largest n={n_values[-1]}:")
    log(f"  MLE (known G): {mle_known_means[-1]:.6f} +/- {mle_known_stds[-1]:.6f}")
    log(f"  Median:        {median_means[-1]:.6f} +/- {median_stds[-1]:.6f}")
    log(f"  Truncated:     {trunc_means[-1]:.6f} +/- {trunc_stds[-1]:.6f}")
    log(f"  NPMLE:         {npmle_means[-1]:.6f} +/- {npmle_stds[-1]:.6f}")

    # Create plot
    create_plot_2pt(n_values, mle_known_means, mle_known_stds,
                    median_means, median_stds, trunc_means, trunc_stds,
                    npmle_means, npmle_stds,
                    prior_atoms, prior_weights, weight_power, sigma_power, si_min, si_max, B, true_mu, n_sims, output_dir)

    return results_data


def create_plot_sos(n_values, median_means, median_stds, trunc_means, trunc_stds,
                    npmle_means, npmle_stds, si_min, si_max, B, mu0, m_power, true_mu, n_sims, output_dir):
    """
    Create publication-quality plot for Subset-of-Signals experiment.

    Generates log-log plot comparing median, iterative truncation, and NPMLE
    estimators. Includes error bars, parameter information, and proper formatting.

    Parameters
    ----------
    n_values : array-like
        Sample sizes tested
    median_means, trunc_means, npmle_means : array-like
        Mean absolute errors at each n
    median_stds, trunc_stds, npmle_stds : array-like
        Standard errors at each n
    si_min, si_max : float
        Standard deviation range
    B : float
        Truncation bound
    mu0 : float
        Initial mean estimate
    m_power : float
        Power parameter (m = n^m_power)
    true_mu : float
        True mean
    n_sims : int
        Number of simulations per n
    output_dir : str
        Directory to save plot

    Saves
    -----
    {output_dir}/experiment1_sos_comparison.png (300 dpi)

    Plot Features
    -------------
    - Log-log scale for easy interpretation of convergence rates
    - Error bars representing standard error of mean
    - Color-coded by estimator (blue=median, orange=trunc, green=NPMLE)
    - Parameter box with experiment details
    - Grid for readability
    - Publication-ready 12×8 inch figure
    """
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
    ax.set_title(f'Comparison of Estimators: Subset-of-Signals Prior', fontsize=16)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Add parameter box
    param_text = (f'Parameters:\n'
                  f'$\\mu^* = {true_mu}$\n'
                  f'$\\sigma_{{min}} = {si_min}$\n'
                  f'$\\sigma_{{max}} = {si_max}$\n'
                  f'$B = {B}$ (truncation)\n'
                  f'$\\mu_0 = {mu0}$ (init)\n'
                  f'$m = n^{{{m_power}}}$\n'
                  f'Simulations: {n_sims}')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.02, param_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)

    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'experiment1_sos_comparison.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    log(f"Plot saved to {plot_file}")


def create_plot_2pt(n_values, mle_means, mle_stds,
                    median_means, median_stds, trunc_means, trunc_stds,
                    npmle_means, npmle_stds,
                    prior_atoms, prior_weights, weight_power, sigma_power, si_min, si_max, B, true_mu, n_sims, output_dir):
    """
    Create comprehensive plot suite for Two-Point Prior experiment.

    Generates four publication-quality plots showing different perspectives
    on estimator performance: absolute errors, oracle comparison, relative
    performance, and combined view.

    Parameters
    ----------
    n_values : array-like
        Sample sizes tested
    mle_means, median_means, trunc_means, npmle_means : array-like
        Mean absolute errors at each n
    mle_stds, median_stds, trunc_stds, npmle_stds : array-like
        Standard errors at each n
    prior_atoms : array-like
        Base atoms [σ_1, σ_2]
    prior_weights : array-like
        Base weights [w_1, w_2]
    weight_power : float
        Dynamic weight power (0 = fixed)
    sigma_power : float
        Dynamic variance power (0 = fixed)
    si_min, si_max : float
        NPMLE search range
    B : float
        Truncation bound
    true_mu : float
        True mean
    n_sims : int
        Number of simulations
    output_dir : str
        Directory to save plots

    Saves
    -----
    Four plots (all 300 dpi):
    1. experiment2_all_estimators.png
       - All four estimators with error bars
       - Reference line showing 1/√n parametric rate
       - Log-log scale

    2. experiment2_mle_vs_npmle.png
       - Oracle MLE vs NPMLE only (cleaner comparison)
       - Shows adaptivity gap

    3. experiment2_error_ratio.png
       - Each estimator's error / Oracle error
       - Horizontal line at 1 = oracle performance
       - Shows relative efficiency

    4. experiment2_combined.png
       - Two-panel figure: absolute errors (left) + ratios (right)
       - Comprehensive single-figure summary

    Notes
    -----
    Color scheme:
    - Red: Oracle MLE (theoretical lower bound)
    - Blue: Median
    - Orange: Truncated mean
    - Green: NPMLE

    The 1/√n reference line shows parametric convergence rate.
    NPMLE aims to achieve this rate adaptively without knowing G.
    """

    colors = {
        'mle_known': '#d62728',  # red
        'median': '#1f77b4',      # blue
        'truncation': '#ff7f0e',  # orange
        'npmle': '#2ca02c'        # green
    }

    valid_npmle = ~np.isnan(npmle_means)
    n_ref = np.array(n_values, dtype=float)

    # Parameter text for all plots
    if weight_power > 0:
        weight_str = f'$w_1 = n^{{-{weight_power}}}$'
    else:
        weight_str = f'$({prior_weights[0]}, {prior_weights[1]})$'
    if sigma_power > 0:
        sigma_str = f'$\\sigma_1 = {prior_atoms[0]}, \\sigma_2 = n^{{{sigma_power}}}$'
    else:
        sigma_str = f'$\\sigma \\in {{{prior_atoms[0]}, {prior_atoms[1]}}}$'
    param_text = (f'Parameters:\n'
                  f'$\\mu^* = {true_mu}$\n'
                  f'Prior atoms: {sigma_str}\n'
                  f'Prior weights: {weight_str}\n'
                  f'$B = {B}$ (truncation)\n'
                  f'NPMLE range: $[{si_min}, {si_max}]$\n'
                  f'Simulations: {n_sims}')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)

    # =========================================================================
    # Plot 1: All estimators comparison
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 8))

    # MLE (known G) - Oracle
    ax.errorbar(n_values, mle_means, yerr=mle_stds,
                fmt='o-', color=colors['mle_known'], linewidth=2, markersize=8,
                capsize=5, capthick=2, label='MLE (known $G$) - Oracle', elinewidth=2)

    # Median
    ax.errorbar(n_values, median_means, yerr=median_stds,
                fmt='d-', color=colors['median'], linewidth=2, markersize=8,
                capsize=5, capthick=2, label='Median', elinewidth=2)

    # Truncated Mean
    ax.errorbar(n_values, trunc_means, yerr=trunc_stds,
                fmt='s-', color=colors['truncation'], linewidth=2, markersize=8,
                capsize=5, capthick=2, label=f'Truncated Mean (B={B})', elinewidth=2)

    # NPMLE
    ax.errorbar(n_values[valid_npmle], npmle_means[valid_npmle],
                yerr=npmle_stds[valid_npmle],
                fmt='^-', color=colors['npmle'], linewidth=2, markersize=8,
                capsize=5, capthick=2, label='NPMLE (unknown $G$)', elinewidth=2)

    # Reference line: 1/sqrt(n)
    ref_sqrt_n = mle_means[0] * np.sqrt(n_values[0]) / np.sqrt(n_ref)
    ax.plot(n_ref, ref_sqrt_n, '--', color='gray', alpha=0.7, linewidth=1.5,
            label=r'$\propto 1/\sqrt{n}$ (parametric)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Sample Size $n$', fontsize=14)
    ax.set_ylabel(r'Absolute Error $|\hat{\mu} - \mu^*|$', fontsize=14)
    ax.set_title(f'Comparison of All Estimators: Two-Point Prior', fontsize=16)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.text(0.02, 0.02, param_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)

    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'experiment2_all_estimators.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    log(f"Plot saved to {plot_file}")

    # =========================================================================
    # Plot 2: MLE vs NPMLE only (cleaner comparison)
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(12, 8))

    ax2.errorbar(n_values, mle_means, yerr=mle_stds,
                fmt='o-', color=colors['mle_known'], linewidth=2, markersize=8,
                capsize=5, capthick=2, label='MLE (known $G$)', elinewidth=2)

    ax2.errorbar(n_values[valid_npmle], npmle_means[valid_npmle],
                yerr=npmle_stds[valid_npmle],
                fmt='^-', color=colors['npmle'], linewidth=2, markersize=8,
                capsize=5, capthick=2, label='NPMLE (unknown $G$)', elinewidth=2)

    ax2.plot(n_ref, ref_sqrt_n, '--', color='gray', alpha=0.7, linewidth=1.5,
            label=r'$\propto 1/\sqrt{n}$')

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Sample Size $n$', fontsize=14)
    ax2.set_ylabel(r'Absolute Error $|\hat{\mu} - \mu^*|$', fontsize=14)
    ax2.set_title(f'MLE vs NPMLE: Two-Point Prior', fontsize=16)
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.text(0.02, 0.02, param_text, transform=ax2.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)

    plt.tight_layout()
    plot_file2 = os.path.join(output_dir, 'experiment2_mle_vs_npmle.png')
    plt.savefig(plot_file2, dpi=300, bbox_inches='tight')
    plt.close()
    log(f"Plot saved to {plot_file2}")

    # =========================================================================
    # Plot 3: Error ratio plot (each estimator / MLE)
    # =========================================================================
    fig3, ax3 = plt.subplots(figsize=(12, 6))

    valid_mle = mle_means > 0

    # NPMLE / MLE ratio
    valid_both = valid_npmle & valid_mle
    if np.any(valid_both):
        ratio_npmle = npmle_means[valid_both] / mle_means[valid_both]
        ax3.plot(n_values[valid_both], ratio_npmle, '^-', color=colors['npmle'],
                linewidth=2, markersize=8, label='NPMLE / MLE')

    # Median / MLE ratio
    if np.any(valid_mle):
        ratio_median = median_means[valid_mle] / mle_means[valid_mle]
        ax3.plot(n_values[valid_mle], ratio_median, 'd-', color=colors['median'],
                linewidth=2, markersize=8, label='Median / MLE')

    # Truncated / MLE ratio
    if np.any(valid_mle):
        ratio_trunc = trunc_means[valid_mle] / mle_means[valid_mle]
        ax3.plot(n_values[valid_mle], ratio_trunc, 's-', color=colors['truncation'],
                linewidth=2, markersize=8, label='Truncated / MLE')

    ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='Oracle (MLE)')
    ax3.set_xscale('log')
    ax3.set_xlabel('Sample Size $n$', fontsize=14)
    ax3.set_ylabel(r'Error Ratio (Estimator / MLE)', fontsize=14)
    ax3.set_title(f'Performance Relative to Oracle MLE: Two-Point Prior', fontsize=16)
    ax3.grid(True, which='both', linestyle='--', alpha=0.5)
    ax3.tick_params(axis='both', which='major', labelsize=12)
    ax3.legend(fontsize=11, loc='best')
    ax3.text(0.02, 0.98, param_text, transform=ax3.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    ratio_plot_file = os.path.join(output_dir, 'experiment2_error_ratio.png')
    plt.savefig(ratio_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    log(f"Ratio plot saved to {ratio_plot_file}")

    # =========================================================================
    # Plot 4: Combined 2-panel figure
    # =========================================================================
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(16, 6))

    # Left panel: All estimators
    ax4a.errorbar(n_values, mle_means, yerr=mle_stds,
                fmt='o-', color=colors['mle_known'], linewidth=2, markersize=6,
                capsize=3, label='MLE (Oracle)', elinewidth=1.5)
    ax4a.errorbar(n_values, median_means, yerr=median_stds,
                fmt='d-', color=colors['median'], linewidth=2, markersize=6,
                capsize=3, label='Median', elinewidth=1.5)
    ax4a.errorbar(n_values, trunc_means, yerr=trunc_stds,
                fmt='s-', color=colors['truncation'], linewidth=2, markersize=6,
                capsize=3, label='Truncated', elinewidth=1.5)
    ax4a.errorbar(n_values[valid_npmle], npmle_means[valid_npmle],
                yerr=npmle_stds[valid_npmle],
                fmt='^-', color=colors['npmle'], linewidth=2, markersize=6,
                capsize=3, label='NPMLE', elinewidth=1.5)
    ax4a.plot(n_ref, ref_sqrt_n, '--', color='gray', alpha=0.7, linewidth=1.5)

    ax4a.set_xscale('log')
    ax4a.set_yscale('log')
    ax4a.set_xlabel('Sample Size $n$', fontsize=14)
    ax4a.set_ylabel(r'Absolute Error $|\hat{\mu} - \mu^*|$', fontsize=14)
    ax4a.set_title('All Estimators', fontsize=14)
    ax4a.legend(fontsize=9, loc='upper right')
    ax4a.grid(True, which='both', linestyle='--', alpha=0.5)

    # Right panel: Ratios
    if np.any(valid_both):
        ax4b.plot(n_values[valid_both], ratio_npmle, '^-', color=colors['npmle'],
                linewidth=2, markersize=6, label='NPMLE')
    if np.any(valid_mle):
        ax4b.plot(n_values[valid_mle], ratio_median, 'd-', color=colors['median'],
                linewidth=2, markersize=6, label='Median')
        ax4b.plot(n_values[valid_mle], ratio_trunc, 's-', color=colors['truncation'],
                linewidth=2, markersize=6, label='Truncated')
    ax4b.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    ax4b.set_xscale('log')
    ax4b.set_xlabel('Sample Size $n$', fontsize=14)
    ax4b.set_ylabel('Error Ratio (/ MLE)', fontsize=14)
    ax4b.set_title('Relative to Oracle', fontsize=14)
    ax4b.legend(fontsize=9, loc='best')
    ax4b.grid(True, which='both', linestyle='--', alpha=0.5)

    # Overall title
    if sigma_power > 0:
        sigma_title = f'$\\sigma_1 = {prior_atoms[0]}, \\sigma_2 = n^{{{sigma_power}}}$'
    else:
        sigma_title = f'$\\sigma \\in {{{prior_atoms[0]}, {prior_atoms[1]}}}$'
    if weight_power > 0:
        title_str = f'Two-Point Prior: {sigma_title}, $w_1 = n^{{-{weight_power}}}$'
    else:
        title_str = f'Two-Point Prior: {sigma_title}, weights $= ({prior_weights[0]}, {prior_weights[1]})$'
    fig4.suptitle(title_str, fontsize=14, y=1.02)

    plt.tight_layout()
    combined_plot_file = os.path.join(output_dir, 'experiment2_combined.png')
    plt.savefig(combined_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    log(f"Combined plot saved to {combined_plot_file}")


def main():
    """
    Main entry point for NPMLE simulation experiments.

    Parses command-line arguments, sets up logging, and runs requested experiments
    with parallel processing. Coordinates the full experimental pipeline from data
    generation through analysis and visualization.

    Command-Line Interface
    ----------------------
    See --help for full argument list. Key arguments:

    General:
        --n-workers: Number of parallel workers (default: all CPUs)
        --n-sims: Simulations per sample size (default: 100)
        --n-points: Number of sample sizes to test (default: 15)
        --n-min, --n-max: Range of sample sizes (log10 scale)
        --experiment: Which experiment to run (sos, 2pt, both)

    Subset-of-Signals (sos):
        --sos-si-min, --sos-si-max: Variance range
        --sos-B: Truncation bound
        --sos-m-power: Fraction parameter (m = n^m_power)

    Two-Point (2pt):
        --tpt-prior-atoms: [σ_1, σ_2]
        --tpt-prior-weights: [w_1, w_2]
        --tpt-weight-power: Dynamic weight (w_1 = n^(-power))
        --tpt-sigma-power: Dynamic variance (σ_2 = n^power)

    Examples
    --------
    # Run both experiments with default settings:
    python npmle_tests_parallel.py

    # Run only SOS experiment with 50 workers:
    python npmle_tests_parallel.py --experiment sos --n-workers 50

    # Run 2pt experiment with dynamic weights and variances:
    python npmle_tests_parallel.py --experiment 2pt \\
        --tpt-weight-power 0.75 --tpt-sigma-power 0.5

    # Custom sample size range:
    python npmle_tests_parallel.py --n-min 2 --n-max 5 --n-points 20

    Output
    ------
    All output is saved to --output-dir (default: ./results/):
    - experiment_progress.log: Detailed execution log
    - experiment1_sos_results.pkl: SOS experiment data
    - experiment1_sos_comparison.png: SOS plot
    - experiment2_2pt_results.pkl: Two-point experiment data
    - experiment2_*.png: Multiple two-point plots

    Performance
    -----------
    - Automatic parallelization across all available CPUs
    - Progress tracking with ETA estimation
    - Typical throughput: 10-100 simulations/second
    - Memory efficient: each worker processes one simulation at a time

    Notes
    -----
    The script is designed for HPC clusters but works on any system with
    Python 3.7+ and the required dependencies (numpy, scipy, matplotlib).
    """
    parser = argparse.ArgumentParser(description='NPMLE Tests - Parallelized')

    # General experiment parameters
    parser.add_argument('--n-workers', type=int, default=None,
                        help='Number of parallel workers (default: all available CPUs)')
    parser.add_argument('--n-sims', type=int, default=100,
                        help='Number of simulations per n value (default: 100)')
    parser.add_argument('--n-points', type=int, default=15,
                        help='Number of n values to test (default: 15)')
    parser.add_argument('--n-min', type=float, default=2,
                        help='Log10 of minimum n (default: 2, i.e., n=100)')
    parser.add_argument('--n-max', type=float, default=4,
                        help='Log10 of maximum n (default: 5, i.e., n=100000)')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Output directory for results (default: ./results)')
    parser.add_argument('--experiment', type=str, default='both',
                        choices=['sos', '2pt', 'both'],
                        help='Which experiment to run (default: both)')
    parser.add_argument('--true-mu', type=float, default=2.0,
                        help='True mean for all experiments (default: 2.0)')

    # SOS experiment parameters
    parser.add_argument('--sos-si-min', type=float, default=0.7,
                        help='SOS: minimum sigma (default: 0.7)')
    parser.add_argument('--sos-si-max', type=float, default=150,
                        help='SOS: maximum sigma (default: 150)')
    parser.add_argument('--sos-B', type=float, default=10.0,
                        help='SOS: truncation bound B (default: 10.0)')
    parser.add_argument('--sos-mu0', type=float, default=1.0,
                        help='SOS: initial mean estimate (default: 1.0)')
    parser.add_argument('--sos-m-power', type=float, default=0.5,
                        help='SOS: power of n for computing m = max(1, floor(n^m_power)) (default: 0.5)')

    # Two-point experiment parameters
    parser.add_argument('--tpt-prior-atoms', type=float, nargs=2, default=[1.0, 30.0],
                        help='Two-point: prior atoms (default: 1.0 30.0)')
    parser.add_argument('--tpt-prior-weights', type=float, nargs=2, default=[0.01, 0.99],
                        help='Two-point: prior weights (default: 0.01 0.99)')
    parser.add_argument('--tpt-si-min', type=float, default=0.5,
                        help='Two-point: minimum sigma for NPMLE (default: 0.5)')
    parser.add_argument('--tpt-si-max', type=float, default=100.0,
                        help='Two-point: maximum sigma for NPMLE (default: 100.0)')
    parser.add_argument('--tpt-weight-power', type=float, default=0.75,
                        help='Two-point: power of n for dynamic weights. If > 0: w1=n^(-power), w2=1-w1. If 0: use fixed weights (default: 0)')
    parser.add_argument('--tpt-sigma-power', type=float, default=0.0,
                        help='Two-point: power of n for dynamic variance. If > 0: sigma_1=fixed, sigma_2=n^(power). If 0: use fixed atoms (default: 0)')
    parser.add_argument('--tpt-B', type=float, default=10.0,
                        help='Two-point: truncation bound B for truncated mean estimator (default: 10.0)')

    args = parser.parse_args()

    # Set number of workers
    if args.n_workers is None:
        n_workers = cpu_count()
    else:
        n_workers = min(args.n_workers, cpu_count())

    # Create output directory first (needed for log file)
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging
    global logger
    logger = setup_logging(args.output_dir, log_to_file=True)

    log("=" * 60)
    log("NPMLE Tests - Parallelized Version")
    log("=" * 60)
    log(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Output directory: {args.output_dir}")
    log(f"Available CPUs: {cpu_count()}")
    log(f"Using {n_workers} workers")
    log(f"True mu: {args.true_mu}")
    log(f"Experiment type: {args.experiment}")
    log(f"n values: {args.n_points} points from 10^{args.n_min} to 10^{args.n_max}")
    log(f"Simulations per n: {args.n_sims}")

    # Define n values
    n_values = np.logspace(args.n_min, args.n_max, args.n_points).astype(int)

    overall_start = time.time()

    # Experiment 1: Subset-of-Signals
    if args.experiment in ['sos', 'both']:
        run_experiment_sos(n_values, args.n_sims, args.true_mu, args.sos_si_min, args.sos_si_max,
                          args.sos_B, args.sos_mu0, args.sos_m_power, n_workers, args.output_dir)

    # Experiment 2: Two-Point Prior
    if args.experiment in ['2pt', 'both']:
        prior_atoms = np.array(args.tpt_prior_atoms)
        prior_weights = np.array(args.tpt_prior_weights)
        run_experiment_2pt(n_values, args.n_sims, args.true_mu, prior_atoms, prior_weights,
                          args.tpt_weight_power, args.tpt_sigma_power, args.tpt_si_min, args.tpt_si_max, args.tpt_B,
                          n_workers, args.output_dir)

    overall_elapsed = time.time() - overall_start
    hours, remainder = divmod(int(overall_elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)

    log("\n" + "=" * 60)
    log("All experiments completed!")
    log(f"Total runtime: {hours}h {minutes}m {seconds}s")
    log(f"Results saved to {args.output_dir}")
    log(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60)


if __name__ == '__main__':
    main()
