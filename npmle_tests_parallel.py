#!/usr/bin/env python3
"""
NPMLE Tests - Parallelized version for HPC cluster
Runs on a node with 120 CPUs and 8 H200 GPUs

This script is converted from NPMLE_py_tests.ipynb with multiprocessing support.
"""

import numpy as np
import math
import pickle
import os
import sys
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Literal
from scipy import stats, optimize as opt
from scipy.stats import norm, poisson, gaussian_kde
from scipy.optimize import root_scalar
from collections import Counter
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import argparse
import scipy
from scipy.stats import norm



# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class GeneralConfig:
    """General experiment configuration."""
    n_workers: Optional[int] = None      # Number of parallel workers (None = all CPUs)
    n_sims: int = 1                    # Number of simulations per n value
    n_points: int = 8                   # Number of n values to test
    n_min: float = 2.0                   # Log10 of minimum n (e.g., 2 means n=100)
    n_max: float = 4.0                  # Log10 of maximum n (e.g., 4 means n=10000)
    output_dir: str = './results'        # Output directory for results
    experiment: str = 'both'             # Which experiment: 'sos', '2pt', or 'both'
    true_mu: float = 0.0                 # True mean for all experiments


@dataclass
class SoSConfig:
    """Subset-of-Signals experiment configuration."""
    si_min: float = 0.7                  # Minimum sigma
    si_max: float = 150.0                # Maximum sigma
    B: float = 10.0                      # Truncation bound for iterative truncation
    mu0: float = 1.0                     # Initial mean estimate
    m_power: float = 1/2                 # Power of n for m = max(1, floor(n^m_power))


@dataclass 
class TwoPointConfig:
    """Two-Point Prior experiment configuration."""
    prior_atoms: List[float] = field(default_factory=lambda: [1.0, 30.0])
    prior_weights: List[float] = field(default_factory=lambda: [0.01, 0.99])
    si_min: float = 0.5                  # Minimum sigma for NPMLE search
    si_max: float = 100.0                # Maximum sigma for NPMLE search
    weight_power: float = 0.0            # If > 0: w1 = n^(-power), w2 = 1-w1
    sigma_power: float = 0.0             # If > 0: sigma_2 = n^(power)
    B: float = 10.0                      # Optional truncation bound for 2pt experiments


@dataclass
class ThreePointConfig:
    """Three-Point Prior experiment configuration."""
    prior_atoms: List[float] = field(default_factory=lambda: [1.0, 60.0, 100.0])
    prior_weights: List[float] = field(default_factory=lambda: [1.0/20.0, 9.0/10.0, 1.0/20.0])
    si_min: float = 0.5                  # Minimum sigma for NPMLE search
    si_max: float = 100.0                # Maximum sigma for NPMLE search


@dataclass
class GaussianPriorConfig:
    """Gaussian prior experiment: homoskedastic, sigma_i = si for all i."""
    si: float = 5.0                     # Common standard deviation
    si_min: float = 0.5                  # Minimum sigma for NPMLE search
    si_max: float = 100.0                # Maximum sigma for NPMLE search


@dataclass
class QuadraticVarianceConfig:
    """Quadratic variance prior: sigma_i = c * i for observation i."""
    c: float = 0.1                       # Scale factor (variance = (c*i)^2)
    si_min: float = 0.01                 # Minimum sigma for NPMLE search
    si_max: float = 500.0                # Maximum sigma for NPMLE search


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    general: GeneralConfig = field(default_factory=GeneralConfig)
    sos: SoSConfig = field(default_factory=SoSConfig)
    two_point: TwoPointConfig = field(default_factory=TwoPointConfig)
    three_point: ThreePointConfig = field(default_factory=ThreePointConfig)
    gaussian_prior: GaussianPriorConfig = field(default_factory=GaussianPriorConfig)
    quadratic_variance: QuadraticVarianceConfig = field(default_factory=QuadraticVarianceConfig)
    
    def print_config(self):
        """Print configuration in a readable format."""
        print("\n" + "=" * 60)
        print("EXPERIMENT CONFIGURATION")
        print("=" * 60)
        
        print("\n[General Settings]")
        print(f"  Workers:        {self.general.n_workers or 'all CPUs'}")
        print(f"  Simulations:    {self.general.n_sims} per n value")
        print(f"  n range:        10^{self.general.n_min} to 10^{self.general.n_max} ({self.general.n_points} points)")
        print(f"  True μ:         {self.general.true_mu}")
        print(f"  Output dir:     {self.general.output_dir}")
        print(f"  Experiment:     {self.general.experiment}")
        
        if self.general.experiment in ['sos', 'both', 'all']:
            print("\n[Subset-of-Signals (SoS)]")
            print(f"  σ range:        [{self.sos.si_min}, {self.sos.si_max}]")
            print(f"  Truncation B:   {self.sos.B}")
            print(f"  Initial μ₀:     {self.sos.mu0}")
            print(f"  m power:        n^{self.sos.m_power}")
        
        if self.general.experiment in ['2pt', 'both', 'all']:
            print("\n[Two-Point Prior]")
            print(f"  Prior atoms:    σ ∈ {{{self.two_point.prior_atoms[0]}, {self.two_point.prior_atoms[1]}}}")
            if self.two_point.weight_power > 0:
                print(f"  Prior weights:  w₁ = n^(-{self.two_point.weight_power}), w₂ = 1 - w₁")
            else:
                print(f"  Prior weights:  ({self.two_point.prior_weights[0]}, {self.two_point.prior_weights[1]})")
            if self.two_point.sigma_power > 0:
                print(f"  Dynamic σ₂:     σ₂ = n^{self.two_point.sigma_power}")
            print(f"  EB range:       [{self.two_point.si_min}, {self.two_point.si_max}]")
            print(f"  B (2pt):        {self.two_point.B}")

        if self.general.experiment in ['3pt', 'all']:
            print("\n[Three-Point Prior]")
            tp = self.three_point
            print(f"  Prior atoms:    σ ∈ {{{tp.prior_atoms[0]}, {tp.prior_atoms[1]}, {tp.prior_atoms[2]}}}")
            print(f"  Prior weights:  ({tp.prior_weights[0]}, {tp.prior_weights[1]}, {tp.prior_weights[2]})")
            print(f"  EB range:       [{tp.si_min}, {tp.si_max}]")
        
        print("=" * 60 + "\n")


# =============================================================================
# DEFAULT CONFIGURATION - MODIFY THIS TO CHANGE EXPERIMENT PARAMETERS
# =============================================================================

def get_default_config() -> ExperimentConfig:
    """
    Returns the default experiment configuration.
    
    MODIFY THIS FUNCTION TO CHANGE YOUR SIMULATION PARAMETERS.
    This is the main place to adjust settings for your experiments.
    """
    return ExperimentConfig(
        general=GeneralConfig(
            n_workers=None,          # Use all available CPUs
            n_sims=150,              # Simulations per n value
            n_points=10,             # Number of n values
            n_min=2.0,               # Start n 
            n_max=4.0,               # End n
            output_dir='./results',
            experiment='all',       # Default: run SoS + 2pt (use --experiment all for all experiments)
            true_mu=0.0,
        ),
        sos=SoSConfig(
            si_min=0.7,
            si_max=150.0,
            B=10.0,
            mu0=1.0,
            m_power=1/3,
        ),
        two_point=TwoPointConfig(
            prior_atoms=[1.0, 30.0],
            prior_weights=[0.01, 0.99],
            si_min=0.1,
            si_max=100.0,
            weight_power=0.0,        # Set > 0 for dynamic weights: w1 = n^(-power)
            sigma_power=0.0,         # Set > 0 for dynamic variance: σ₂ = n^(power)
        ),
        three_point=ThreePointConfig(
            prior_atoms=[1.0, 60.0, 100.0],
            prior_weights=[1.0/20.0, 9.0/10.0, 1.0/20.0],
            si_min=0.001,
            si_max=100.0,
        ),
        gaussian_prior=GaussianPriorConfig(
            si=5.0,
            si_min=0.01,
            si_max=100.0,
        ),
        quadratic_variance=QuadraticVarianceConfig(
            c=0.1,
            si_min=0.01,
            si_max=100000.0,
        ),
    )


def parse_args():
    """
    Parse command-line arguments for overriding the default configuration.
    
    This is primarily used by SLURM submission scripts (e.g., submit_npmle.sh),
    but is optional when running the script manually.
    """
    parser = argparse.ArgumentParser(description="NPMLE Tests - Parallelized version for HPC cluster")

    # General settings
    parser.add_argument("--n-workers", type=int, default=None, help="Number of parallel workers (processes)")
    parser.add_argument("--n-sims", type=int, default=None, help="Number of simulations per n value")
    parser.add_argument("--n-points", type=int, default=None, help="Number of n values to test")
    parser.add_argument("--n-min", type=float, default=None, help="Log10 of minimum n")
    parser.add_argument("--n-max", type=float, default=None, help="Log10 of maximum n")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")
    parser.add_argument("--experiment", type=str, default=None,
                        choices=["sos", "2pt", "gaussian", "quadratic", "3pt", "both", "all"],
                        help="Which experiment(s) to run")
    parser.add_argument("--true-mu", type=float, default=None, help="True mean for all experiments")

    # SOS experiment settings
    parser.add_argument("--sos-si-min", type=float, default=None, help="Minimum sigma for SOS prior")
    parser.add_argument("--sos-si-max", type=float, default=None, help="Maximum sigma for SOS prior")
    parser.add_argument("--sos-B", type=float, default=None, help="Truncation bound B for SOS estimator")
    parser.add_argument("--sos-mu0", type=float, default=None, help="Initial mean estimate for SOS iterative truncation")
    parser.add_argument("--sos-m-power", type=float, default=None, help="Power for m = n^m_power in SOS experiment")

    # Two-point prior experiment settings
    parser.add_argument("--tpt-prior-atoms", type=float, nargs="+", default=None,
                        help="Prior atoms (sigma values) for two-point prior (space-separated)")
    parser.add_argument("--tpt-prior-weights", type=float, nargs="+", default=None,
                        help="Prior weights for two-point prior (space-separated, must sum to 1)")
    parser.add_argument("--tpt-weight-power", type=float, default=None,
                        help="If > 0: w1 = n^(-power), w2 = 1 - w1; if 0: use fixed weights")
    parser.add_argument("--tpt-si-min", type=float, default=None, help="Minimum sigma for NPMLE search (2pt)")
    parser.add_argument("--tpt-si-max", type=float, default=None, help="Maximum sigma for NPMLE search (2pt)")
    parser.add_argument("--tpt-B", type=float, default=None, help="Optional truncation bound B for 2pt experiment")

    # Three-point prior experiment settings
    parser.add_argument("--three-prior-atoms", type=float, nargs="+", default=None,
                        help="Prior atoms (sigma values) for three-point prior (space-separated)")
    parser.add_argument("--three-prior-weights", type=float, nargs="+", default=None,
                        help="Prior weights for three-point prior (space-separated, must sum to 1)")
    parser.add_argument("--three-si-min", type=float, default=None,
                        help="Minimum sigma for NPMLE search (3pt)")
    parser.add_argument("--three-si-max", type=float, default=None,
                        help="Maximum sigma for NPMLE search (3pt)")

    # Gaussian prior experiment settings
    parser.add_argument("--gaussian-si", type=float, default=None, help="Common sigma for Gaussian prior")
    parser.add_argument("--gaussian-si-min", type=float, default=None, help="NPMLE search min (Gaussian prior)")
    parser.add_argument("--gaussian-si-max", type=float, default=None, help="NPMLE search max (Gaussian prior)")

    # Quadratic variance prior experiment settings
    parser.add_argument("--quadratic-c", type=float, default=None, help="Scale c for quadratic variance (sigma_i = c*i)")
    parser.add_argument("--quadratic-si-min", type=float, default=None, help="NPMLE search min (quadratic prior)")
    parser.add_argument("--quadratic-si-max", type=float, default=None, help="NPMLE search max (quadratic prior)")

    args = parser.parse_args()
    return args


def config_from_args(args) -> ExperimentConfig:
    """
    Build an ExperimentConfig from parsed CLI arguments, overriding defaults
    only where arguments are provided.
    """
    config = get_default_config()

    # General settings
    if args.n_workers is not None:
        config.general.n_workers = args.n_workers
    if args.n_sims is not None:
        config.general.n_sims = args.n_sims
    if args.n_points is not None:
        config.general.n_points = args.n_points
    if args.n_min is not None:
        config.general.n_min = args.n_min
    if args.n_max is not None:
        config.general.n_max = args.n_max
    if args.output_dir is not None:
        config.general.output_dir = args.output_dir
    if args.experiment is not None:
        config.general.experiment = args.experiment
    if args.true_mu is not None:
        config.general.true_mu = args.true_mu

    # SOS settings
    if args.sos_si_min is not None:
        config.sos.si_min = args.sos_si_min
    if args.sos_si_max is not None:
        config.sos.si_max = args.sos_si_max
    if args.sos_B is not None:
        config.sos.B = args.sos_B
    if args.sos_mu0 is not None:
        config.sos.mu0 = args.sos_mu0
    if args.sos_m_power is not None:
        config.sos.m_power = args.sos_m_power

    # Two-point prior settings
    if args.tpt_prior_atoms is not None:
        config.two_point.prior_atoms = list(args.tpt_prior_atoms)
    if args.tpt_prior_weights is not None:
        config.two_point.prior_weights = list(args.tpt_prior_weights)
    if args.tpt_weight_power is not None:
        config.two_point.weight_power = args.tpt_weight_power
    if args.tpt_si_min is not None:
        config.two_point.si_min = args.tpt_si_min
    if args.tpt_si_max is not None:
        config.two_point.si_max = args.tpt_si_max
    if args.tpt_B is not None:
        config.two_point.B = args.tpt_B

    # Three-point prior settings
    if args.three_prior_atoms is not None:
        config.three_point.prior_atoms = list(args.three_prior_atoms)
    if args.three_prior_weights is not None:
        config.three_point.prior_weights = list(args.three_prior_weights)
    if args.three_si_min is not None:
        config.three_point.si_min = args.three_si_min
    if args.three_si_max is not None:
        config.three_point.si_max = args.three_si_max

    # Gaussian prior settings
    if args.gaussian_si is not None:
        config.gaussian_prior.si = args.gaussian_si
    if args.gaussian_si_min is not None:
        config.gaussian_prior.si_min = args.gaussian_si_min
    if args.gaussian_si_max is not None:
        config.gaussian_prior.si_max = args.gaussian_si_max

    # Quadratic variance prior settings
    if args.quadratic_c is not None:
        config.quadratic_variance.c = args.quadratic_c
    if args.quadratic_si_min is not None:
        config.quadratic_variance.si_min = args.quadratic_si_min
    if args.quadratic_si_max is not None:
        config.quadratic_variance.si_max = args.quadratic_si_max

    return config


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(output_dir, log_to_file=True):
    """Setup logging to both console and file"""
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
    """Log a message to both console and file"""
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

def gaussian_prior(si, n, rng=None):
    """Gaussian prior: homoskedastic, all observations have variance si^2"""
    if rng is None:
        rng = np.random.default_rng()
    return rng.normal(0, si, size=n)

def quadratic_variance_prior(c, n, rng=None):
    """Quadratic variance prior: each signal i has variance (c*i)^2, so sigma_i = c*i"""
    if rng is None:
        rng = np.random.default_rng()
    scales = c * (np.arange(1, n + 1)**2)  # std dev for observation i
    return rng.normal(0, scales)  # shape (n,)

def generate_data_gaussian_prior(mu, si, n, rng=None):
    """Generate data from Gaussian prior: X_i ~ N(mu, si^2) for all i."""
    if rng is None:
        rng = np.random.default_rng()
    X = mu + gaussian_prior(si, n, rng)
    true_sigmas = np.full(n, si)
    return X, true_sigmas

def generate_data_quadratic_variance_prior(mu, c, n, rng=None):
    """Generate data from quadratic variance prior: X_i ~ N(mu, (c*i)^2)."""
    if rng is None:
        rng = np.random.default_rng()
    true_sigmas = c * np.arange(1, n + 1)
    X = mu + rng.normal(0, true_sigmas)
    return X, true_sigmas

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
    """Directional derivative for atom update (single sigma value)"""
    n = len(N_obs)
    if denom is None:
        denom = mixture_density(prior_atoms, prior_weights, mu, N_obs)
    denom = np.maximum(denom, 1e-300)
    numer = norm.pdf(N_obs, loc=mu, scale=sigma)
    return (1.0/n) * np.sum(numer/denom) - 1.0


def DG_sigma_grid(prior_atoms, prior_weights, sigma_grid, mu, N_obs):
    """Vectorized directional derivative for grid of sigma values.
    
    Args:
        sigma_grid: array of shape (M,) - grid of sigma values to evaluate
        N_obs: array of shape (n,) - observations
    
    Returns:
        array of shape (M,) - DG values for each sigma in the grid
    """
    n = len(N_obs)
    denom = mixture_density(prior_atoms, prior_weights, mu, N_obs)  # shape (n,)
    denom = np.maximum(denom, 1e-300)
    
    # Reshape for broadcasting: N_obs (n, 1), sigma_grid (1, M), denom (n, 1)
    N_obs_2d = N_obs[:, np.newaxis]  # (n, 1)
    sigma_2d = sigma_grid[np.newaxis, :]  # (1, M)
    denom_2d = denom[:, np.newaxis]  # (n, 1)
    
    # numer shape: (n, M)
    numer = norm.pdf(N_obs_2d, loc=mu, scale=sigma_2d)
    
    # Sum over observations (axis=0), result shape: (M,)
    return (1.0/n) * np.sum(numer / denom_2d, axis=0) - 1.0


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

# def get_weights_npmle(atoms, weights, mu, N_obs):
#     """
#     Optimize mixture weights for fixed atoms and mu via L-BFGS-B,
#     maximizing the average log-likelihood.

#     Returns
#     -------
#     opt_success : bool
#     weights_hat : (K,) array, normalized
#     """
#     atoms = np.asarray(atoms, dtype=float)
#     N_obs = np.asarray(N_obs, dtype=float)

#     Phi = norm.pdf(N_obs[:, None], loc=mu, scale=atoms[None, :])

#     def target(w):
#         # L-BFGS-B works better if we keep it unconstrained but penalize invalid regions
#         w = np.asarray(w, dtype=float)
#         # if np.any(w < 0):
#         #     return np.inf
#         # if w.sum() <= 0:
#         #     return np.inf
#         # w = w / w.sum()
#         # negative average log-likelihood (we minimize this)
#         dens = Phi @ (w+1e-10) + 1e-20
#         ret = np.mean(np.log(dens))
#         return -ret + np.log(np.sum(w) + 1e-10)

#     def jac(w):
#         w = np.asarray(w, dtype=float)
#         dens = Phi @ (w+1e-10) + 1e-20
#         grad = Phi.T @ (1/dens) / len(N_obs)
#         return -grad + 1/(np.sum(w)+1e-10)


#     opt_success = False
#     number_ls = 20
#     opt_options = {'maxiter': int(number_ls)}
#     max_tries = 3
#     it = 0

#     # Start from uniform weights over the current atoms
#     # weights0 = np.ones(len(atoms), dtype=float) / len(atoms)
#     weights0 = weights / weights.sum()
#     m = len(atoms)
#     # print(weights0)
#     while (not opt_success) and (it < max_tries):
#         result = sp.optimize.minimize(
#             target,
#             weights0,
#             jac=jac,
#             method='L-BFGS-B',
#             tol=1e-10,
#             bounds=sp.optimize.Bounds(1e-10+np.zeros(m), np.inf + np.zeros(m), True))
#         if not result.success:
#             number_ls *= 3
#             print("Weight optimization failed. Increase maxls to ", number_ls)
#             opt_options = {'maxls': int(number_ls), 'maxfun': int(1e6)}
#             it += 1
#         else:
#             opt_success = True

#     if not opt_success:
#         print("Weight optimization (L-BFGS-B) failed; falling back to EM or previous weights.")
#         return weights0
#     else:
#         return result.x / np.sum(result.x)

def get_weights_npmle(atoms, weights, mu, N_obs):
    atoms = np.asarray(atoms, dtype=float)
    N_obs = np.asarray(N_obs, dtype=float)

    if np.any(atoms <= 0):
        raise ValueError("All atoms (scales) must be strictly positive.")

    Phi = norm.pdf(N_obs[:, None], loc=mu, scale=atoms[None, :])
    n, m = Phi.shape

    def target(w):
        dens = Phi @ w
        dens = np.maximum(dens, 1e-300)
        return -np.mean(np.log(dens))

    def jac(w):
        dens = Phi @ w
        dens = np.maximum(dens, 1e-300)
        return -(Phi.T @ (1.0 / dens)) / n

    weights0 = np.asarray(weights, dtype=float)
    weights0 = np.maximum(weights0, 1e-12)
    weights0 = weights0 / weights0.sum()

    constraints = [{
        'type': 'eq',
        'fun': lambda w: np.sum(w) - 1.0,
        'jac': lambda w: np.ones_like(w),
    }]
    bounds = [(0.0, 1.0)] * m

    result = scipy.optimize.minimize(
        target,
        weights0,
        jac=jac,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-12}
    )

    if not result.success:
        print("Weight optimization failed.")
        return weights0

    w = np.maximum(result.x, 0.0)
    w = w / w.sum()
    return w

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

def update_atoms_grid(atoms, weights, mu, N_obs, si_min, si_max, n_grid=5000, verbose=False):
    """Find new atom using grid search over sigma values.
    
    Uses vectorized DG_sigma_grid for efficiency.
    """
    x = np.linspace(si_min, si_max, n_grid)
    y = DG_sigma_grid(atoms, weights, x, mu, N_obs)  # Use vectorized version
    new_atom = x[np.argmax(y)]
    max_val = np.max(y)
    
    if verbose:
        print(f"Grid search: max DG = {max_val:.6f} at sigma = {new_atom:.4f}")
    
    return new_atom, max_val

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
        # Update atoms (add new atom)
        s_star, dg_val = update_atoms_grid(atoms, weights, mu, N_obs, si_min, si_max)

        if dg_val > 1e-6:
            atoms = np.append(atoms, s_star)
            weights = np.append(weights, 0.0)
            #weights = get_weights_npmle_EM(N_obs, atoms, mu=mu, w_init=weights)
        
        # Update weights
        weights = get_weights_npmle(atoms=atoms, weights=weights, mu=mu, N_obs = N_obs)
        # weights = get_weights_npmle_EM(N_obs, atoms, mu=mu, w_init=weights)

        # Prune zero-weight atoms
        mask = weights > weight_tol
        atoms = atoms[mask]
        weights = weights[mask]
        weights /= weights.sum()

        if len(atoms) == 0:
            atoms = np.array([max(si_min, np.std(N_obs))])
            weights = np.array([1.0])

        # Check convergence
        ll = avg_loglikelihood(atoms, weights, mu, N_obs)
        if abs(ll - prev_ll) < progress_tol:
            break
        prev_ll = ll

    # Merge close atoms
    atoms, weights = merge_atoms_weights(atoms, weights, atom_tol)

    return atoms, weights


def mle_location_grid(N_obs, atoms, weights, mu_init=None, n_grid=5000):
    """
    MLE for location using grid search over mu values.
    
    This computes the log-likelihood for each mu in the grid and returns
    the mu that maximizes it.
    """
    N_obs = np.asarray(N_obs, dtype=float)
    atoms = np.asarray(atoms, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Create grid of mu values
    grid = np.linspace(np.min(N_obs), np.max(N_obs), n_grid)
    
    # Compute log-likelihood for each mu in the grid
    # We need to vectorize this properly
    n = len(N_obs)
    K = len(atoms)
    
    # Reshape for broadcasting:
    # N_obs: (n,) -> (n, 1) 
    # grid: (M,) -> (1, M)
    # atoms: (K,) -> (K, 1, 1)
    # weights: (K,) -> (K, 1, 1)
    
    N_obs_2d = N_obs[:, np.newaxis]  # (n, M) after broadcast
    grid_2d = grid[np.newaxis, :]     # (n, M) after broadcast
    
    # Compute mixture density for all (observation, mu) pairs
    # Result shape: (n, M)
    densities = np.zeros((n, len(grid)))
    for i in range(K):
        # norm.pdf with obs (n,1) and mu (1,M) broadcasts to (n, M)
        densities += weights[i] * norm.pdf(N_obs_2d, loc=grid_2d, scale=atoms[i])
    
    densities = np.maximum(densities, 1e-300)
    
    # Average log-likelihood for each mu: mean over observations (axis=0)
    LL = np.mean(np.log(densities), axis=0)  # Shape: (M,)
    
    mu_hat = grid[np.argmax(LL)]
    return mu_hat


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

    si_min = max(si_min, np.min(np.abs(N_obs)))
    si_max = min(si_max, np.max(np.abs(N_obs)))

    mu = np.median(N_obs)
    # atoms = np.array([(si_min + si_max) / 2])
    #weights = np.array([1.0])

    grid_size = 5
    atoms = np.linspace(si_min, si_max, grid_size)
    weights = np.ones(grid_size, dtype=float) / grid_size

    # dependency on initialization test:
    prev_ll = -np.inf

    init_ll = avg_loglikelihood(atoms, weights, mu, N_obs)
    # print(f"Iteration {0}: mu={mu: .6f}, ll={init_ll:.6f}, atoms={len(atoms)}")
    for iter_count in range(iter_max):
        # Update scale mixture (fixed mu)
        atoms, weights = npmle_heteroskedastic(
            N_obs, mu, si_min, si_max,
            atoms_init=atoms, weights_init=weights,
            iter_max=iter_max, verbose=False
        )

        # Update mu (fixed scale mixture)
        # mu = mle_location_scale_mixture(N_obs, atoms, weights, mu_init=mu)
        mu = mle_location_grid(N_obs, atoms, weights, mu_init=mu)

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
            N_obs=X, si_min=0.0001, si_max=1000, iter_max=20, verbose=False)
        err_npmle = np.abs(mu_hat_npmle - true_mu)
    except Exception as e:
        err_npmle = np.nan
    # _, _, mu_hat_npmle = npmle_alternating_full(N_obs=X, si_min=si_min, si_max=si_max, iter_max=10, verbose=True)
    # err_npmle = np.abs(mu_hat_npmle - true_mu)
    return {
        'n': n,
        'sim': s,
        'err_median': err_median,
        'err_trunc': err_trunc,
        'err_npmle': err_npmle
    }


def run_single_simulation_2pt(args):
    """Run a single simulation for Two-Point Prior experiment"""
    s, n, true_mu, prior_atoms, prior_weights, weight_power, sigma_power, si_min, si_max, seed = args
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

    # 1. MLE Oracle (known sigma_i) - knows true variances for each observation
    mu_hat_mle_oracle = mle_known_sigma(X, true_sigmas)
    err_mle_oracle = np.abs(mu_hat_mle_oracle - true_mu)

    # 2. Median estimator
    mu_hat_median = median_estimator(X)
    err_median = np.abs(mu_hat_median - true_mu)

    # 3. MLE (Known G) - knows the prior distribution, uses grid search
    mu_hat_mle_known_G = mle_location_grid(X, atoms, weights)
    err_mle_known_G = np.abs(mu_hat_mle_known_G - true_mu)

    # 4. NPMLE estimator (unknown G)
    try:
        _, _, mu_hat_npmle = npmle_alternating_full(
            N_obs=X, si_min=0.001, si_max=500, iter_max=10, verbose=False)
        err_npmle = np.abs(mu_hat_npmle - true_mu)
    except Exception as e:
        err_npmle = np.nan

    return {
        'n': n,
        'sim': s,
        'err_mle_oracle': err_mle_oracle,
        'err_median': err_median,
        'err_mle_known_G': err_mle_known_G,
        'err_npmle': err_npmle
    }


def run_single_simulation_gaussian(args):
    """Run a single simulation for Gaussian Prior experiment"""
    s, n, true_mu, si, si_min, si_max, seed = args
    rng = np.random.default_rng(seed)

    X, true_sigmas = generate_data_gaussian_prior(true_mu, si, n, rng=rng)

    # 1. MLE Oracle (known sigma_i)
    mu_hat_mle_oracle = mle_known_sigma(X, true_sigmas)
    err_mle_oracle = np.abs(mu_hat_mle_oracle - true_mu)

    # 2. Median estimator
    mu_hat_median = median_estimator(X)
    err_median = np.abs(mu_hat_median - true_mu)

    # 3. MLE (Known G) - homoskedastic: atoms=[si], weights=[1]
    atoms = np.array([si])
    weights = np.array([1.0])
    mu_hat_mle_known_G = mle_location_grid(X, atoms, weights)
    err_mle_known_G = np.abs(mu_hat_mle_known_G - true_mu)

    # 4. EB (NPMLE) estimator
    try:
        _, _, mu_hat_npmle = npmle_alternating_full(
            N_obs=X, si_min=si_min, si_max=si_max, iter_max=10, verbose=False)
        err_npmle = np.abs(mu_hat_npmle - true_mu)
    except Exception:
        err_npmle = np.nan

    return {
        'n': n,
        'sim': s,
        'err_mle_oracle': err_mle_oracle,
        'err_median': err_median,
        'err_mle_known_G': err_mle_known_G,
        'err_npmle': err_npmle
    }


def run_single_simulation_quadratic(args):
    """Run a single simulation for Quadratic Variance Prior experiment"""
    s, n, true_mu, c, si_min, si_max, seed = args
    rng = np.random.default_rng(seed)

    X, true_sigmas = generate_data_quadratic_variance_prior(true_mu, c, n, rng=rng)

    # 1. MLE Oracle (known sigma_i)
    mu_hat_mle_oracle = mle_known_sigma(X, true_sigmas)
    err_mle_oracle = np.abs(mu_hat_mle_oracle - true_mu)

    # 2. Median estimator
    mu_hat_median = median_estimator(X)
    err_median = np.abs(mu_hat_median - true_mu)

    # 3. MLE (Known G) - sigma_i = c*i is known for each i, so same as MLE Oracle
    mu_hat_mle_known_G = mle_known_sigma(X, true_sigmas)
    err_mle_known_G = np.abs(mu_hat_mle_known_G - true_mu)

    # 4. EB (NPMLE) estimator
    try:
        _, _, mu_hat_npmle = npmle_alternating_full(
            N_obs=X, si_min=si_min, si_max=si_max, iter_max=10, verbose=False)
        err_npmle = np.abs(mu_hat_npmle - true_mu)
    except Exception:
        err_npmle = np.nan

    return {
        'n': n,
        'sim': s,
        'err_mle_oracle': err_mle_oracle,
        'err_median': err_median,
        'err_mle_known_G': err_mle_known_G,
        'err_npmle': err_npmle
    }


def run_single_simulation_3pt(args):
    """Run a single simulation for Three-Point Prior experiment"""
    (s, n, true_mu,
     prior_atoms, prior_weights,
     si_min, si_max, seed) = args

    rng = np.random.default_rng(seed)

    # Generate data from fixed three-point prior
    X, true_sigmas = generate_data_from_prior(prior_atoms, prior_weights, true_mu, n, rng=rng)

    # 1. MLE Oracle (known sigma_i)
    mu_hat_mle_oracle = mle_known_sigma(X, true_sigmas)
    err_mle_oracle = np.abs(mu_hat_mle_oracle - true_mu)

    # 2. Median estimator
    mu_hat_median = median_estimator(X)
    err_median = np.abs(mu_hat_median - true_mu)

    # 3. MLE (Known G) - three-point prior, known mixing distribution
    mu_hat_mle_known_G = mle_location_grid(X, prior_atoms, prior_weights)
    err_mle_known_G = np.abs(mu_hat_mle_known_G - true_mu)

    # 4. EB (NPMLE) estimator (unknown G)
    try:
        _, _, mu_hat_npmle = npmle_alternating_full(
            N_obs=X, si_min=si_min, si_max=si_max, iter_max=10, verbose=False
        )
        err_npmle = np.abs(mu_hat_npmle - true_mu)
    except Exception:
        err_npmle = np.nan

    return {
        'n': n,
        'sim': s,
        'err_mle_oracle': err_mle_oracle,
        'err_median': err_median,
        'err_mle_known_G': err_mle_known_G,
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


def run_experiment_sos(n_values, n_sims, true_mu, si_min, si_max, B, mu0, m_power, n_workers, output_dir):
    """Run Subset-of-Signals experiment with parallel processing"""
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
                       weight_power, sigma_power, si_min, si_max, n_workers, output_dir):
    """Run Two-Point Prior experiment with parallel processing"""
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
    log(f"EB search range: [{si_min}, {si_max}]")
    log("=" * 60)

    # Prepare all simulation arguments
    all_args = []
    base_seed = 10000
    for n in n_values:
        for s in range(n_sims):
            seed = base_seed + n * n_sims + s
            all_args.append((s, n, true_mu, prior_atoms, prior_weights, weight_power, sigma_power, si_min, si_max, seed))

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
    results_mle_oracle = {n: [] for n in n_values}
    results_median = {n: [] for n in n_values}
    results_mle_known_G = {n: [] for n in n_values}
    results_npmle = {n: [] for n in n_values}

    for r in results:
        n = r['n']
        results_mle_oracle[n].append(r['err_mle_oracle'])
        results_median[n].append(r['err_median'])
        results_mle_known_G[n].append(r['err_mle_known_G'])
        results_npmle[n].append(r['err_npmle'])

    # Compute statistics
    mle_oracle_means, mle_oracle_stds = compute_stats(results_mle_oracle, n_values)
    median_means, median_stds = compute_stats(results_median, n_values)
    mle_known_G_means, mle_known_G_stds = compute_stats(results_mle_known_G, n_values)
    npmle_means, npmle_stds = compute_stats(results_npmle, n_values)

    # Save results
    results_data = {
        'n_values': n_values,
        'results_mle_oracle': results_mle_oracle,
        'results_median': results_median,
        'results_mle_known_G': results_mle_known_G,
        'results_npmle': results_npmle,
        'mle_oracle_means': mle_oracle_means,
        'mle_oracle_stds': mle_oracle_stds,
        'median_means': median_means,
        'median_stds': median_stds,
        'mle_known_G_means': mle_known_G_means,
        'mle_known_G_stds': mle_known_G_stds,
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
            'n_sims': n_sims
        }
    }

    results_file = os.path.join(output_dir, 'experiment2_2pt_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results_data, f)
    log(f"Results saved to {results_file}")

    # Log summary statistics
    log(f"Summary - Mean errors at largest n={n_values[-1]}:")
    log(f"  MLE Oracle:    {mle_oracle_means[-1]:.6f} +/- {mle_oracle_stds[-1]:.6f}")
    log(f"  MLE (Known G): {mle_known_G_means[-1]:.6f} +/- {mle_known_G_stds[-1]:.6f}")
    log(f"  Median:        {median_means[-1]:.6f} +/- {median_stds[-1]:.6f}")
    log(f"  NPMLE:         {npmle_means[-1]:.6f} +/- {npmle_stds[-1]:.6f}")

    # Create plot
    create_plot_2pt(n_values, mle_oracle_means, mle_oracle_stds,
                    mle_known_G_means, mle_known_G_stds,
                    median_means, median_stds,
                    npmle_means, npmle_stds,
                    prior_atoms, prior_weights, weight_power, sigma_power, si_min, si_max, true_mu, n_sims, output_dir)

    return results_data


def run_experiment_gaussian(n_values, n_sims, true_mu, si, si_min, si_max, n_workers, output_dir):
    """Run Gaussian Prior experiment with parallel processing"""
    log("\n" + "=" * 60)
    log("EXPERIMENT 3: Gaussian Prior (Homoskedastic)")
    log("=" * 60)
    log(f"Prior: X_i ~ N(mu, sigma^2) with sigma = {si}")
    log(f"n values: {n_values}")
    log(f"Simulations per n: {n_sims}")
    log(f"Using {n_workers} workers")
    log(f"EB search range: [{si_min}, {si_max}]")
    log("=" * 60)

    all_args = []
    base_seed = 20000
    for n in n_values:
        for s in range(n_sims):
            seed = base_seed + n * n_sims + s
            all_args.append((s, n, true_mu, si, si_min, si_max, seed))

    total_sims = len(all_args)
    log(f"Total simulations to run: {total_sims}")

    start_time = time.time()
    results = []
    completed = 0
    last_progress = 0

    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(run_single_simulation_gaussian, all_args):
            results.append(result)
            completed += 1
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

    results_mle_oracle = {n: [] for n in n_values}
    results_median = {n: [] for n in n_values}
    results_mle_known_G = {n: [] for n in n_values}
    results_npmle = {n: [] for n in n_values}

    for r in results:
        n = r['n']
        results_mle_oracle[n].append(r['err_mle_oracle'])
        results_median[n].append(r['err_median'])
        results_mle_known_G[n].append(r['err_mle_known_G'])
        results_npmle[n].append(r['err_npmle'])

    mle_oracle_means, mle_oracle_stds = compute_stats(results_mle_oracle, n_values)
    median_means, median_stds = compute_stats(results_median, n_values)
    mle_known_G_means, mle_known_G_stds = compute_stats(results_mle_known_G, n_values)
    npmle_means, npmle_stds = compute_stats(results_npmle, n_values)

    results_data = {
        'n_values': n_values,
        'results_mle_oracle': results_mle_oracle,
        'results_median': results_median,
        'results_mle_known_G': results_mle_known_G,
        'results_npmle': results_npmle,
        'mle_oracle_means': mle_oracle_means,
        'mle_oracle_stds': mle_oracle_stds,
        'median_means': median_means,
        'median_stds': median_stds,
        'mle_known_G_means': mle_known_G_means,
        'mle_known_G_stds': mle_known_G_stds,
        'npmle_means': npmle_means,
        'npmle_stds': npmle_stds,
        'params': {
            'true_mu': true_mu,
            'si': si,
            'si_min': si_min,
            'si_max': si_max,
            'n_sims': n_sims
        }
    }

    results_file = os.path.join(output_dir, 'experiment3_gaussian_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results_data, f)
    log(f"Results saved to {results_file}")

    log(f"Summary - Mean errors at largest n={n_values[-1]}:")
    log(f"  MLE Oracle:    {mle_oracle_means[-1]:.6f} +/- {mle_oracle_stds[-1]:.6f}")
    log(f"  MLE (Known G): {mle_known_G_means[-1]:.6f} +/- {mle_known_G_stds[-1]:.6f}")
    log(f"  Median:        {median_means[-1]:.6f} +/- {median_stds[-1]:.6f}")
    log(f"  EB (NPMLE):    {npmle_means[-1]:.6f} +/- {npmle_stds[-1]:.6f}")

    create_plot_gaussian(n_values, mle_oracle_means, mle_oracle_stds,
                         mle_known_G_means, mle_known_G_stds,
                         median_means, median_stds,
                         npmle_means, npmle_stds,
                         si, si_min, si_max, true_mu, n_sims, output_dir)

    return results_data


def run_experiment_quadratic(n_values, n_sims, true_mu, c, si_min, si_max, n_workers, output_dir):
    """Run Quadratic Variance Prior experiment with parallel processing"""
    log("\n" + "=" * 60)
    log("EXPERIMENT 4: Quadratic Variance Prior")
    log("=" * 60)
    log(f"Prior: sigma_i = c * i, c = {c}")
    log(f"n values: {n_values}")
    log(f"Simulations per n: {n_sims}")
    log(f"Using {n_workers} workers")
    log(f"EB search range: [{si_min}, {si_max}]")
    log("=" * 60)

    all_args = []
    base_seed = 30000
    for n in n_values:
        for s in range(n_sims):
            seed = base_seed + n * n_sims + s
            all_args.append((s, n, true_mu, c, si_min, si_max, seed))

    total_sims = len(all_args)
    log(f"Total simulations to run: {total_sims}")

    start_time = time.time()
    results = []
    completed = 0
    last_progress = 0

    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(run_single_simulation_quadratic, all_args):
            results.append(result)
            completed += 1
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

    results_mle_oracle = {n: [] for n in n_values}
    results_median = {n: [] for n in n_values}
    results_mle_known_G = {n: [] for n in n_values}
    results_npmle = {n: [] for n in n_values}

    for r in results:
        n = r['n']
        results_mle_oracle[n].append(r['err_mle_oracle'])
        results_median[n].append(r['err_median'])
        results_mle_known_G[n].append(r['err_mle_known_G'])
        results_npmle[n].append(r['err_npmle'])

    mle_oracle_means, mle_oracle_stds = compute_stats(results_mle_oracle, n_values)
    median_means, median_stds = compute_stats(results_median, n_values)
    mle_known_G_means, mle_known_G_stds = compute_stats(results_mle_known_G, n_values)
    npmle_means, npmle_stds = compute_stats(results_npmle, n_values)

    results_data = {
        'n_values': n_values,
        'results_mle_oracle': results_mle_oracle,
        'results_median': results_median,
        'results_mle_known_G': results_mle_known_G,
        'results_npmle': results_npmle,
        'mle_oracle_means': mle_oracle_means,
        'mle_oracle_stds': mle_oracle_stds,
        'median_means': median_means,
        'median_stds': median_stds,
        'mle_known_G_means': mle_known_G_means,
        'mle_known_G_stds': mle_known_G_stds,
        'npmle_means': npmle_means,
        'npmle_stds': npmle_stds,
        'params': {
            'true_mu': true_mu,
            'c': c,
            'si_min': si_min,
            'si_max': si_max,
            'n_sims': n_sims
        }
    }

    results_file = os.path.join(output_dir, 'experiment4_quadratic_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results_data, f)
    log(f"Results saved to {results_file}")

    log(f"Summary - Mean errors at largest n={n_values[-1]}:")
    log(f"  MLE Oracle:    {mle_oracle_means[-1]:.6f} +/- {mle_oracle_stds[-1]:.6f}")
    log(f"  MLE (Known G): {mle_known_G_means[-1]:.6f} +/- {mle_known_G_stds[-1]:.6f}")
    log(f"  Median:        {median_means[-1]:.6f} +/- {median_stds[-1]:.6f}")
    log(f"  EB (NPMLE):    {npmle_means[-1]:.6f} +/- {npmle_stds[-1]:.6f}")

    create_plot_quadratic(n_values, mle_oracle_means, mle_oracle_stds,
                          mle_known_G_means, mle_known_G_stds,
                          median_means, median_stds,
                          npmle_means, npmle_stds,
                          c, si_min, si_max, true_mu, n_sims, output_dir)

    return results_data


def run_experiment_3pt(n_values, n_sims, true_mu, prior_atoms, prior_weights,
                       si_min, si_max, n_workers, output_dir):
    """Run Three-Point Prior experiment with parallel processing"""
    log("\n" + "=" * 60)
    log("EXPERIMENT 5: Three-Point Prior (All Estimators)")
    log("=" * 60)

    # Ensure numpy arrays
    prior_atoms = np.array(prior_atoms, dtype=float)
    prior_weights = np.array(prior_weights, dtype=float)

    log(f"Prior atoms: {prior_atoms}, weights: {prior_weights}")
    log(f"n values: {n_values}")
    log(f"Simulations per n: {n_sims}")
    log(f"Using {n_workers} workers")
    log(f"EB search range: [{si_min}, {si_max}]")
    log("=" * 60)

    # Prepare all simulation arguments
    all_args = []
    base_seed = 40000
    for n in n_values:
        for s in range(n_sims):
            seed = base_seed + n * n_sims + s
            all_args.append(
                (s, n, true_mu, prior_atoms, prior_weights, si_min, si_max, seed)
            )

    total_sims = len(all_args)
    log(f"Total simulations to run: {total_sims}")

    start_time = time.time()
    results = []
    completed = 0
    last_progress = 0

    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(run_single_simulation_3pt, all_args):
            results.append(result)
            completed += 1
            progress = int(100 * completed / total_sims)
            if progress >= last_progress + 10 or completed == total_sims:
                elapsed_so_far = time.time() - start_time
                rate = completed / elapsed_so_far if elapsed_so_far > 0 else 0
                eta = (total_sims - completed) / rate if rate > 0 else 0
                log(
                    f"Progress: {completed}/{total_sims} ({progress}%) | "
                    f"Rate: {rate:.1f} sims/sec | ETA: {eta:.0f}s"
                )
                last_progress = progress

    elapsed = time.time() - start_time
    log(
        f"Completed {len(results)} simulations in {elapsed:.2f} seconds "
        f"({len(results) / elapsed:.1f} sims/sec)"
    )

    # Organize results
    results_mle_oracle = {n: [] for n in n_values}
    results_median = {n: [] for n in n_values}
    results_mle_known_G = {n: [] for n in n_values}
    results_npmle = {n: [] for n in n_values}

    for r in results:
        n = r["n"]
        results_mle_oracle[n].append(r["err_mle_oracle"])
        results_median[n].append(r["err_median"])
        results_mle_known_G[n].append(r["err_mle_known_G"])
        results_npmle[n].append(r["err_npmle"])

    # Compute statistics
    mle_oracle_means, mle_oracle_stds = compute_stats(results_mle_oracle, n_values)
    median_means, median_stds = compute_stats(results_median, n_values)
    mle_known_G_means, mle_known_G_stds = compute_stats(results_mle_known_G, n_values)
    npmle_means, npmle_stds = compute_stats(results_npmle, n_values)

    # Save results
    results_data = {
        "n_values": n_values,
        "results_mle_oracle": results_mle_oracle,
        "results_median": results_median,
        "results_mle_known_G": results_mle_known_G,
        "results_npmle": results_npmle,
        "mle_oracle_means": mle_oracle_means,
        "mle_oracle_stds": mle_oracle_stds,
        "median_means": median_means,
        "median_stds": median_stds,
        "mle_known_G_means": mle_known_G_means,
        "mle_known_G_stds": mle_known_G_stds,
        "npmle_means": npmle_means,
        "npmle_stds": npmle_stds,
        "params": {
            "true_mu": true_mu,
            "prior_atoms": prior_atoms,
            "prior_weights": prior_weights,
            "si_min": si_min,
            "si_max": si_max,
            "n_sims": n_sims,
        },
    }

    results_file = os.path.join(output_dir, "experiment5_3pt_results.pkl")
    with open(results_file, "wb") as f:
        pickle.dump(results_data, f)
    log(f"Results saved to {results_file}")

    log(f"Summary - Mean errors at largest n={n_values[-1]}:")
    log(
        f"  MLE Oracle:    {mle_oracle_means[-1]:.6f} +/- {mle_oracle_stds[-1]:.6f}"
    )
    log(
        f"  MLE (Known G): {mle_known_G_means[-1]:.6f} +/- {mle_known_G_stds[-1]:.6f}"
    )
    log(f"  Median:        {median_means[-1]:.6f} +/- {median_stds[-1]:.6f}")
    log(f"  EB (NPMLE):    {npmle_means[-1]:.6f} +/- {npmle_stds[-1]:.6f}")

    create_plot_3pt(
        n_values,
        mle_oracle_means,
        mle_oracle_stds,
        mle_known_G_means,
        mle_known_G_stds,
        median_means,
        median_stds,
        npmle_means,
        npmle_stds,
        prior_atoms,
        prior_weights,
        si_min,
        si_max,
        true_mu,
        n_sims,
        output_dir,
    )

    return results_data


def create_plot_sos(n_values, median_means, median_stds, trunc_means, trunc_stds,
                    npmle_means, npmle_stds, si_min, si_max, B, mu0, m_power, true_mu, n_sims, output_dir):
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
                capsize=5, capthick=2, label='EB', elinewidth=2)

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


def create_plot_2pt(n_values, mle_oracle_means, mle_oracle_stds,
                    mle_known_G_means, mle_known_G_stds,
                    median_means, median_stds,
                    npmle_means, npmle_stds,
                    prior_atoms, prior_weights, weight_power, sigma_power, si_min, si_max, true_mu, n_sims, output_dir):
    """Create plots for Two-Point Prior experiment with all estimators"""

    colors = {
        'mle_oracle': '#d62728',   # red - MLE Oracle (known sigma_i)
        'mle_known_G': '#9467bd',  # purple - MLE (Known G)
        'median': '#1f77b4',       # blue
        'npmle': '#2ca02c'         # green
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
                  f'EB range: $[{si_min}, {si_max}]$\n'
                  f'Simulations: {n_sims}')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)

    # =========================================================================
    # Plot 1: All estimators comparison
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 8))

    # MLE Oracle (known sigma_i)
    ax.errorbar(n_values, mle_oracle_means, yerr=mle_oracle_stds,
                fmt='o-', color=colors['mle_oracle'], linewidth=2, markersize=8,
                capsize=5, capthick=2, label=r'MLE Oracle (known $\sigma_i$)', elinewidth=2)

    # MLE (Known G)
    ax.errorbar(n_values, mle_known_G_means, yerr=mle_known_G_stds,
                fmt='s-', color=colors['mle_known_G'], linewidth=2, markersize=8,
                capsize=5, capthick=2, label='MLE (Known $G$)', elinewidth=2)

    # Median
    ax.errorbar(n_values, median_means, yerr=median_stds,
                fmt='d-', color=colors['median'], linewidth=2, markersize=8,
                capsize=5, capthick=2, label='Median', elinewidth=2)

    # NPMLE
    ax.errorbar(n_values[valid_npmle], npmle_means[valid_npmle],
                yerr=npmle_stds[valid_npmle],
                fmt='^-', color=colors['npmle'], linewidth=2, markersize=8,
                capsize=5, capthick=2, label='EB (unknown $G$)', elinewidth=2)

    # Reference line: 1/sqrt(n)
    ref_sqrt_n = mle_oracle_means[0] * np.sqrt(n_values[0]) / np.sqrt(n_ref)
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
    # Plot 2: MLE Oracle vs NPMLE only (cleaner comparison)
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(12, 8))

    ax2.errorbar(n_values, mle_oracle_means, yerr=mle_oracle_stds,
                fmt='o-', color=colors['mle_oracle'], linewidth=2, markersize=8,
                capsize=5, capthick=2, label=r'MLE Oracle (known $\sigma_i$)', elinewidth=2)

    ax2.errorbar(n_values[valid_npmle], npmle_means[valid_npmle],
                yerr=npmle_stds[valid_npmle],
                fmt='^-', color=colors['npmle'], linewidth=2, markersize=8,
                capsize=5, capthick=2, label='EB (unknown $G$)', elinewidth=2)

    ax2.plot(n_ref, ref_sqrt_n, '--', color='gray', alpha=0.7, linewidth=1.5,
            label=r'$\propto 1/\sqrt{n}$')

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Sample Size $n$', fontsize=14)
    ax2.set_ylabel(r'Absolute Error $|\hat{\mu} - \mu^*|$', fontsize=14)
    ax2.set_title(f'MLE Oracle vs EB: Two-Point Prior', fontsize=16)
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
    # Plot 3: Error ratio plot (each estimator / MLE Oracle)
    # =========================================================================
    fig3, ax3 = plt.subplots(figsize=(12, 6))

    valid_mle = mle_oracle_means > 0

    # NPMLE / MLE Oracle ratio
    valid_both = valid_npmle & valid_mle
    if np.any(valid_both):
        ratio_npmle = npmle_means[valid_both] / mle_oracle_means[valid_both]
        ax3.plot(n_values[valid_both], ratio_npmle, '^-', color=colors['npmle'],
                linewidth=2, markersize=8, label='EB / MLE Oracle')

    # MLE (Known G) / MLE Oracle ratio
    if np.any(valid_mle):
        ratio_mle_known_G = mle_known_G_means[valid_mle] / mle_oracle_means[valid_mle]
        ax3.plot(n_values[valid_mle], ratio_mle_known_G, 's-', color=colors['mle_known_G'],
                linewidth=2, markersize=8, label='MLE (Known $G$) / MLE Oracle')

    # Median / MLE Oracle ratio
    if np.any(valid_mle):
        ratio_median = median_means[valid_mle] / mle_oracle_means[valid_mle]
        ax3.plot(n_values[valid_mle], ratio_median, 'd-', color=colors['median'],
                linewidth=2, markersize=8, label='Median / MLE Oracle')

    ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='MLE Oracle')
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
    ax4a.errorbar(n_values, mle_oracle_means, yerr=mle_oracle_stds,
                fmt='o-', color=colors['mle_oracle'], linewidth=2, markersize=6,
                capsize=3, label=r'MLE Oracle (known $\sigma_i$)', elinewidth=1.5)
    ax4a.errorbar(n_values, mle_known_G_means, yerr=mle_known_G_stds,
                fmt='s-', color=colors['mle_known_G'], linewidth=2, markersize=6,
                capsize=3, label='MLE (Known $G$)', elinewidth=1.5)
    ax4a.errorbar(n_values, median_means, yerr=median_stds,
                fmt='d-', color=colors['median'], linewidth=2, markersize=6,
                capsize=3, label='Median', elinewidth=1.5)
    ax4a.errorbar(n_values[valid_npmle], npmle_means[valid_npmle],
                yerr=npmle_stds[valid_npmle],
                fmt='^-', color=colors['npmle'], linewidth=2, markersize=6,
                capsize=3, label='EB', elinewidth=1.5)
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
        ax4b.plot(n_values[valid_mle], ratio_mle_known_G, 's-', color=colors['mle_known_G'],
                linewidth=2, markersize=6, label='MLE (Known $G$)')
        ax4b.plot(n_values[valid_mle], ratio_median, 'd-', color=colors['median'],
                linewidth=2, markersize=6, label='Median')
    ax4b.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    ax4b.set_xscale('log')
    ax4b.set_xlabel('Sample Size $n$', fontsize=14)
    ax4b.set_ylabel('Error Ratio (/ MLE Oracle)', fontsize=14)
    ax4b.set_title('Relative to MLE Oracle', fontsize=14)
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


def create_plot_3pt(n_values, mle_oracle_means, mle_oracle_stds,
                    mle_known_G_means, mle_known_G_stds,
                    median_means, median_stds,
                    npmle_means, npmle_stds,
                    prior_atoms, prior_weights, si_min, si_max, true_mu, n_sims, output_dir):
    """Create plots for Three-Point Prior experiment with all estimators."""

    colors = {
        'mle_oracle': '#d62728',   # red - MLE Oracle (known sigma_i)
        'mle_known_G': '#9467bd',  # purple - MLE (Known G)
        'median': '#1f77b4',       # blue
        'npmle': '#2ca02c'         # green
    }

    valid_npmle = ~np.isnan(npmle_means)
    n_ref = np.array(n_values, dtype=float)

    sigma_str = f'$\\sigma \\in {{{prior_atoms[0]}, {prior_atoms[1]}, {prior_atoms[2]}}}$'
    weight_str = f'$({prior_weights[0]}, {prior_weights[1]}, {prior_weights[2]})$'
    param_text = (f'Parameters:\n'
                  f'$\\mu^* = {true_mu}$\n'
                  f'Prior atoms: {sigma_str}\n'
                  f'Prior weights: {weight_str}\n'
                  f'EB range: $[{si_min}, {si_max}]$\n'
                  f'Simulations: {n_sims}')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)

    # Plot 1: All estimators
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.errorbar(n_values, mle_oracle_means, yerr=mle_oracle_stds,
                fmt='o-', color=colors['mle_oracle'], linewidth=2, markersize=8,
                capsize=5, capthick=2, label=r'MLE Oracle (known $\sigma_i$)', elinewidth=2)
    ax.errorbar(n_values, mle_known_G_means, yerr=mle_known_G_stds,
                fmt='s-', color=colors['mle_known_G'], linewidth=2, markersize=8,
                capsize=5, capthick=2, label='MLE (Known $G$)', elinewidth=2)
    ax.errorbar(n_values, median_means, yerr=median_stds,
                fmt='d-', color=colors['median'], linewidth=2, markersize=8,
                capsize=5, capthick=2, label='Median', elinewidth=2)
    ax.errorbar(n_values[valid_npmle], npmle_means[valid_npmle],
                yerr=npmle_stds[valid_npmle],
                fmt='^-', color=colors['npmle'], linewidth=2, markersize=8,
                capsize=5, capthick=2, label='EB (unknown $G$)', elinewidth=2)

    ref_sqrt_n = mle_oracle_means[0] * np.sqrt(n_values[0]) / np.sqrt(n_ref)
    ax.plot(n_ref, ref_sqrt_n, '--', color='gray', alpha=0.7, linewidth=1.5,
            label=r'$\propto 1/\sqrt{n}$ (parametric)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Sample Size $n$', fontsize=14)
    ax.set_ylabel(r'Absolute Error $|\hat{\mu} - \mu^*|$', fontsize=14)
    ax.set_title('Comparison of All Estimators: Three-Point Prior', fontsize=16)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.text(0.02, 0.02, param_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)

    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'experiment5_3pt_all_estimators.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    log(f"Plot saved to {plot_file}")

    # Plot 2: MLE Oracle vs EB
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    ax2.errorbar(n_values, mle_oracle_means, yerr=mle_oracle_stds,
                 fmt='o-', color=colors['mle_oracle'], linewidth=2, markersize=8,
                 capsize=5, capthick=2, label=r'MLE Oracle (known $\sigma_i$)', elinewidth=2)
    ax2.errorbar(n_values[valid_npmle], npmle_means[valid_npmle],
                 yerr=npmle_stds[valid_npmle],
                 fmt='^-', color=colors['npmle'], linewidth=2, markersize=8,
                 capsize=5, capthick=2, label='EB (unknown $G$)', elinewidth=2)
    ax2.plot(n_ref, ref_sqrt_n, '--', color='gray', alpha=0.7, linewidth=1.5,
             label=r'$\propto 1/\sqrt{n}$')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Sample Size $n$', fontsize=14)
    ax2.set_ylabel(r'Absolute Error $|\hat{\mu} - \mu^*|$', fontsize=14)
    ax2.set_title('MLE Oracle vs EB: Three-Point Prior', fontsize=16)
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.text(0.02, 0.02, param_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='bottom', bbox=props)
    plt.tight_layout()
    plot_file2 = os.path.join(output_dir, 'experiment5_3pt_mle_vs_npmle.png')
    plt.savefig(plot_file2, dpi=300, bbox_inches='tight')
    plt.close()
    log(f"Plot saved to {plot_file2}")

    # Plot 3: Error ratio
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    valid_mle = mle_oracle_means > 0
    valid_both = valid_npmle & valid_mle
    if np.any(valid_both):
        ratio_npmle = npmle_means[valid_both] / mle_oracle_means[valid_both]
        ax3.plot(n_values[valid_both], ratio_npmle, '^-', color=colors['npmle'],
                 linewidth=2, markersize=8, label='EB / MLE Oracle')
    if np.any(valid_mle):
        ratio_mle_known_G = mle_known_G_means[valid_mle] / mle_oracle_means[valid_mle]
        ax3.plot(n_values[valid_mle], ratio_mle_known_G, 's-', color=colors['mle_known_G'],
                 linewidth=2, markersize=8, label='MLE (Known $G$) / MLE Oracle')
        ratio_median = median_means[valid_mle] / mle_oracle_means[valid_mle]
        ax3.plot(n_values[valid_mle], ratio_median, 'd-', color=colors['median'],
                 linewidth=2, markersize=8, label='Median / MLE Oracle')
    ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='MLE Oracle')
    ax3.set_xscale('log')
    ax3.set_xlabel('Sample Size $n$', fontsize=14)
    ax3.set_ylabel(r'Error Ratio (Estimator / MLE)', fontsize=14)
    ax3.set_title('Performance Relative to Oracle MLE: Three-Point Prior', fontsize=16)
    ax3.grid(True, which='both', linestyle='--', alpha=0.5)
    ax3.tick_params(axis='both', which='major', labelsize=12)
    ax3.legend(fontsize=11, loc='best')
    ax3.text(0.02, 0.98, param_text, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    plt.tight_layout()
    ratio_plot_file = os.path.join(output_dir, 'experiment5_3pt_error_ratio.png')
    plt.savefig(ratio_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    log(f"Ratio plot saved to {ratio_plot_file}")

    # Plot 4: Combined 2-panel figure
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(16, 6))
    ax4a.errorbar(n_values, mle_oracle_means, yerr=mle_oracle_stds,
                  fmt='o-', color=colors['mle_oracle'], linewidth=2, markersize=6,
                  capsize=3, label=r'MLE Oracle (known $\sigma_i$)', elinewidth=1.5)
    ax4a.errorbar(n_values, mle_known_G_means, yerr=mle_known_G_stds,
                  fmt='s-', color=colors['mle_known_G'], linewidth=2, markersize=6,
                  capsize=3, label='MLE (Known $G$)', elinewidth=1.5)
    ax4a.errorbar(n_values, median_means, yerr=median_stds,
                  fmt='d-', color=colors['median'], linewidth=2, markersize=6,
                  capsize=3, label='Median', elinewidth=1.5)
    ax4a.errorbar(n_values[valid_npmle], npmle_means[valid_npmle],
                  yerr=npmle_stds[valid_npmle],
                  fmt='^-', color=colors['npmle'], linewidth=2, markersize=6,
                  capsize=3, label='EB', elinewidth=1.5)
    ax4a.plot(n_ref, ref_sqrt_n, '--', color='gray', alpha=0.7, linewidth=1.5)
    ax4a.set_xscale('log')
    ax4a.set_yscale('log')
    ax4a.set_xlabel('Sample Size $n$', fontsize=14)
    ax4a.set_ylabel(r'Absolute Error $|\hat{\mu} - \mu^*|$', fontsize=14)
    ax4a.set_title('All Estimators', fontsize=14)
    ax4a.legend(fontsize=9, loc='upper right')
    ax4a.grid(True, which='both', linestyle='--', alpha=0.5)

    if np.any(valid_both):
        ax4b.plot(n_values[valid_both], ratio_npmle, '^-', color=colors['npmle'],
                  linewidth=2, markersize=6, label='NPMLE')
    if np.any(valid_mle):
        ax4b.plot(n_values[valid_mle], ratio_mle_known_G, 's-', color=colors['mle_known_G'],
                  linewidth=2, markersize=6, label='MLE (Known $G$)')
        ax4b.plot(n_values[valid_mle], ratio_median, 'd-', color=colors['median'],
                  linewidth=2, markersize=6, label='Median')
    ax4b.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    ax4b.set_xscale('log')
    ax4b.set_xlabel('Sample Size $n$', fontsize=14)
    ax4b.set_ylabel('Error Ratio (/ MLE Oracle)', fontsize=14)
    ax4b.set_title('Relative to MLE Oracle', fontsize=14)
    ax4b.legend(fontsize=9, loc='best')
    ax4b.grid(True, which='both', linestyle='--', alpha=0.5)

    title_str = (f'Three-Point Prior: '
                 f'$\\sigma \\in {{{prior_atoms[0]}, {prior_atoms[1]}, {prior_atoms[2]}}}$, '
                 f'weights $= ({prior_weights[0]}, {prior_weights[1]}, {prior_weights[2]})$')
    fig4.suptitle(title_str, fontsize=14, y=1.02)
    plt.tight_layout()
    combined_plot_file = os.path.join(output_dir, 'experiment5_3pt_combined.png')
    plt.savefig(combined_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    log(f"Combined plot saved to {combined_plot_file}")


def create_plot_gaussian(n_values, mle_oracle_means, mle_oracle_stds,
                        mle_known_G_means, mle_known_G_stds,
                        median_means, median_stds,
                        npmle_means, npmle_stds,
                        si, si_min, si_max, true_mu, n_sims, output_dir):
    """Create plots for Gaussian Prior experiment (same format as 2pt)"""
    colors = {
        'mle_oracle': '#d62728',
        'mle_known_G': '#9467bd',
        'median': '#1f77b4',
        'npmle': '#2ca02c'
    }
    valid_npmle = ~np.isnan(npmle_means)
    n_ref = np.array(n_values, dtype=float)
    param_text = (f'Parameters:\n'
                  f'$\\mu^* = {true_mu}$\n'
                  f'Prior: $\\sigma = {si}$ (homoskedastic)\n'
                  f'EB range: $[{si_min}, {si_max}]$\n'
                  f'Simulations: {n_sims}')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)

    # Plot 1: All estimators
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.errorbar(n_values, mle_oracle_means, yerr=mle_oracle_stds,
                fmt='o-', color=colors['mle_oracle'], linewidth=2, markersize=8,
                capsize=5, capthick=2, label=r'MLE Oracle (known $\sigma_i$)', elinewidth=2)
    ax.errorbar(n_values, mle_known_G_means, yerr=mle_known_G_stds,
                fmt='s-', color=colors['mle_known_G'], linewidth=2, markersize=8,
                capsize=5, capthick=2, label='MLE (Known $G$)', elinewidth=2)
    ax.errorbar(n_values, median_means, yerr=median_stds,
                fmt='d-', color=colors['median'], linewidth=2, markersize=8,
                capsize=5, capthick=2, label='Median', elinewidth=2)
    ax.errorbar(n_values[valid_npmle], npmle_means[valid_npmle], yerr=npmle_stds[valid_npmle],
                fmt='^-', color=colors['npmle'], linewidth=2, markersize=8,
                capsize=5, capthick=2, label='EB (unknown $G$)', elinewidth=2)
    ref_sqrt_n = mle_oracle_means[0] * np.sqrt(n_values[0]) / np.sqrt(n_ref)
    ax.plot(n_ref, ref_sqrt_n, '--', color='gray', alpha=0.7, linewidth=1.5,
            label=r'$\propto 1/\sqrt{n}$ (parametric)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Sample Size $n$', fontsize=14)
    ax.set_ylabel(r'Absolute Error $|\hat{\mu} - \mu^*|$', fontsize=14)
    ax.set_title('Comparison of All Estimators: Gaussian Prior', fontsize=16)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.text(0.02, 0.02, param_text, transform=ax.transAxes, fontsize=9, verticalalignment='bottom', bbox=props)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'experiment3_gaussian_all_estimators.png'), dpi=300, bbox_inches='tight')
    plt.close()
    log(f"Plot saved to experiment3_gaussian_all_estimators.png")

    # Plot 2: MLE Oracle vs EB
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    ax2.errorbar(n_values, mle_oracle_means, yerr=mle_oracle_stds,
                 fmt='o-', color=colors['mle_oracle'], linewidth=2, markersize=8,
                 capsize=5, capthick=2, label=r'MLE Oracle (known $\sigma_i$)', elinewidth=2)
    ax2.errorbar(n_values[valid_npmle], npmle_means[valid_npmle], yerr=npmle_stds[valid_npmle],
                 fmt='^-', color=colors['npmle'], linewidth=2, markersize=8,
                 capsize=5, capthick=2, label='EB (unknown $G$)', elinewidth=2)
    ax2.plot(n_ref, ref_sqrt_n, '--', color='gray', alpha=0.7, linewidth=1.5, label=r'$\propto 1/\sqrt{n}$')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Sample Size $n$', fontsize=14)
    ax2.set_ylabel(r'Absolute Error $|\hat{\mu} - \mu^*|$', fontsize=14)
    ax2.set_title('MLE Oracle vs EB: Gaussian Prior', fontsize=16)
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.text(0.02, 0.02, param_text, transform=ax2.transAxes, fontsize=9, verticalalignment='bottom', bbox=props)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'experiment3_gaussian_mle_vs_eb.png'), dpi=300, bbox_inches='tight')
    plt.close()
    log(f"Plot saved to experiment3_gaussian_mle_vs_eb.png")

    # Plot 3: Error ratio
    ratio_plot_file = os.path.join(output_dir, 'experiment3_gaussian_error_ratio.png')
    plt.figure(figsize=(12, 6))
    valid_mle = mle_oracle_means > 0
    valid_both = valid_npmle & valid_mle
    if np.any(valid_both):
        ratio_npmle = npmle_means[valid_both] / mle_oracle_means[valid_both]
        plt.plot(n_values[valid_both], ratio_npmle, '^-', color=colors['npmle'],
                 linewidth=2, markersize=8, label='EB / MLE Oracle')
    if np.any(valid_mle):
        ratio_mle_known_G = mle_known_G_means[valid_mle] / mle_oracle_means[valid_mle]
        plt.plot(n_values[valid_mle], ratio_mle_known_G, 's-', color=colors['mle_known_G'],
                 linewidth=2, markersize=8, label='MLE (Known $G$) / MLE Oracle')
        ratio_median = median_means[valid_mle] / mle_oracle_means[valid_mle]
        plt.plot(n_values[valid_mle], ratio_median, 'd-', color=colors['median'],
                 linewidth=2, markersize=8, label='Median / MLE Oracle')
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='MLE Oracle')
    plt.xscale('log')
    plt.xlabel('Sample Size $n$', fontsize=14)
    plt.ylabel(r'Error Ratio (Estimator / MLE)', fontsize=14)
    plt.title('Performance Relative to Oracle MLE: Gaussian Prior', fontsize=16)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(fontsize=11, loc='best')
    plt.text(0.02, 0.98, param_text, transform=plt.gca().transAxes, fontsize=9, verticalalignment='top', bbox=props)
    plt.tight_layout()
    plt.savefig(ratio_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    log(f"Ratio plot saved to {ratio_plot_file}")


def create_plot_quadratic(n_values, mle_oracle_means, mle_oracle_stds,
                         mle_known_G_means, mle_known_G_stds,
                         median_means, median_stds,
                         npmle_means, npmle_stds,
                         c, si_min, si_max, true_mu, n_sims, output_dir):
    """Create plots for Quadratic Variance Prior experiment (same format as 2pt)"""
    colors = {
        'mle_oracle': '#d62728',
        'mle_known_G': '#9467bd',
        'median': '#1f77b4',
        'npmle': '#2ca02c'
    }
    valid_npmle = ~np.isnan(npmle_means)
    n_ref = np.array(n_values, dtype=float)
    param_text = (f'Parameters:\n'
                  f'$\\mu^* = {true_mu}$\n'
                  f'Prior: $\\sigma_i = c \\cdot i^2$, $c = {c}$\n'
                  f'EB range: $[{si_min}, {si_max}]$\n'
                  f'Simulations: {n_sims}')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)

    # Plot 1: All estimators
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.errorbar(n_values, mle_oracle_means, yerr=mle_oracle_stds,
                fmt='o-', color=colors['mle_oracle'], linewidth=2, markersize=8,
                capsize=5, capthick=2, label=r'MLE Oracle (known $\sigma_i$)', elinewidth=2)
    ax.errorbar(n_values, mle_known_G_means, yerr=mle_known_G_stds,
                fmt='s-', color=colors['mle_known_G'], linewidth=2, markersize=8,
                capsize=5, capthick=2, label='MLE (Known $G$)', elinewidth=2)
    ax.errorbar(n_values, median_means, yerr=median_stds,
                fmt='d-', color=colors['median'], linewidth=2, markersize=8,
                capsize=5, capthick=2, label='Median', elinewidth=2)
    ax.errorbar(n_values[valid_npmle], npmle_means[valid_npmle], yerr=npmle_stds[valid_npmle],
                fmt='^-', color=colors['npmle'], linewidth=2, markersize=8,
                capsize=5, capthick=2, label='EB (unknown $G$)', elinewidth=2)
    ref_sqrt_n = mle_oracle_means[0] * np.sqrt(n_values[0]) / np.sqrt(n_ref)
    ax.plot(n_ref, ref_sqrt_n, '--', color='gray', alpha=0.7, linewidth=1.5,
            label=r'$\propto 1/\sqrt{n}$ (parametric)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Sample Size $n$', fontsize=14)
    ax.set_ylabel(r'Absolute Error $|\hat{\mu} - \mu^*|$', fontsize=14)
    ax.set_title('Comparison of All Estimators: Quadratic Variance Prior', fontsize=16)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.text(0.02, 0.02, param_text, transform=ax.transAxes, fontsize=9, verticalalignment='bottom', bbox=props)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'experiment4_quadratic_all_estimators.png'), dpi=300, bbox_inches='tight')
    plt.close()
    log(f"Plot saved to experiment4_quadratic_all_estimators.png")

    # Plot 2: MLE Oracle vs EB
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    ax2.errorbar(n_values, mle_oracle_means, yerr=mle_oracle_stds,
                 fmt='o-', color=colors['mle_oracle'], linewidth=2, markersize=8,
                 capsize=5, capthick=2, label=r'MLE Oracle (known $\sigma_i$)', elinewidth=2)
    ax2.errorbar(n_values[valid_npmle], npmle_means[valid_npmle], yerr=npmle_stds[valid_npmle],
                 fmt='^-', color=colors['npmle'], linewidth=2, markersize=8,
                 capsize=5, capthick=2, label='EB (unknown $G$)', elinewidth=2)
    ax2.plot(n_ref, ref_sqrt_n, '--', color='gray', alpha=0.7, linewidth=1.5, label=r'$\propto 1/\sqrt{n}$')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Sample Size $n$', fontsize=14)
    ax2.set_ylabel(r'Absolute Error $|\hat{\mu} - \mu^*|$', fontsize=14)
    ax2.set_title('MLE Oracle vs EB: Quadratic Variance Prior', fontsize=16)
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.text(0.02, 0.02, param_text, transform=ax2.transAxes, fontsize=9, verticalalignment='bottom', bbox=props)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'experiment4_quadratic_mle_vs_eb.png'), dpi=300, bbox_inches='tight')
    plt.close()
    log(f"Plot saved to experiment4_quadratic_mle_vs_eb.png")

    # Plot 3: Error ratio
    ratio_plot_file = os.path.join(output_dir, 'experiment4_quadratic_error_ratio.png')
    plt.figure(figsize=(12, 6))
    valid_mle = mle_oracle_means > 0
    valid_both = valid_npmle & valid_mle
    if np.any(valid_both):
        ratio_npmle = npmle_means[valid_both] / mle_oracle_means[valid_both]
        plt.plot(n_values[valid_both], ratio_npmle, '^-', color=colors['npmle'],
                 linewidth=2, markersize=8, label='EB / MLE Oracle')
    if np.any(valid_mle):
        ratio_mle_known_G = mle_known_G_means[valid_mle] / mle_oracle_means[valid_mle]
        plt.plot(n_values[valid_mle], ratio_mle_known_G, 's-', color=colors['mle_known_G'],
                 linewidth=2, markersize=8, label='MLE (Known $G$) / MLE Oracle')
        ratio_median = median_means[valid_mle] / mle_oracle_means[valid_mle]
        plt.plot(n_values[valid_mle], ratio_median, 'd-', color=colors['median'],
                 linewidth=2, markersize=8, label='Median / MLE Oracle')
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='MLE Oracle')
    plt.xscale('log')
    plt.xlabel('Sample Size $n$', fontsize=14)
    plt.ylabel(r'Error Ratio (Estimator / MLE)', fontsize=14)
    plt.title('Performance Relative to Oracle MLE: Quadratic Variance Prior', fontsize=16)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(fontsize=11, loc='best')
    plt.text(0.02, 0.98, param_text, transform=plt.gca().transAxes, fontsize=9, verticalalignment='top', bbox=props)
    plt.tight_layout()
    plt.savefig(ratio_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    log(f"Ratio plot saved to {ratio_plot_file}")


def main(config: ExperimentConfig = None):
    """
    Main entry point for running experiments.
    
    Args:
        config: ExperimentConfig object. If None, uses get_default_config().
    """
    # Get configuration
    if config is None:
        config = get_default_config()
    
    # Extract shortcuts for convenience
    gen = config.general
    sos = config.sos
    tpt = config.two_point
    gauss = config.gaussian_prior
    quad = config.quadratic_variance
    
    # Set number of workers
    if gen.n_workers is None:
        n_workers = cpu_count()
    else:
        n_workers = min(gen.n_workers, cpu_count())

    # Create output directory first (needed for log file)
    os.makedirs(gen.output_dir, exist_ok=True)

    # Setup logging
    global logger
    logger = setup_logging(gen.output_dir, log_to_file=True)

    # Print configuration
    log("=" * 60)
    log("EB Tests - Parallelized Version")
    log("=" * 60)
    log(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Output directory: {gen.output_dir}")
    log(f"Available CPUs: {cpu_count()}")
    log(f"Using {n_workers} workers")
    log(f"True mu: {gen.true_mu}")
    log(f"Experiment type: {gen.experiment}")
    log(f"n values: {gen.n_points} points from 10^{gen.n_min} to 10^{gen.n_max}")
    log(f"Simulations per n: {gen.n_sims}")

    # Define n values
    n_values = np.logspace(gen.n_min, gen.n_max, gen.n_points).astype(int)

    overall_start = time.time()

    # Experiment 1: Subset-of-Signals
    if gen.experiment in ['sos', 'both', 'all']:
        run_experiment_sos(
            n_values=n_values,
            n_sims=gen.n_sims,
            true_mu=gen.true_mu,
            si_min=sos.si_min,
            si_max=sos.si_max,
            B=sos.B,
            mu0=sos.mu0,
            m_power=sos.m_power,
            n_workers=n_workers,
            output_dir=gen.output_dir
        )

    # Experiment 2: Two-Point Prior
    if gen.experiment in ['2pt', 'both', 'all']:
        prior_atoms = np.array(tpt.prior_atoms)
        prior_weights = np.array(tpt.prior_weights)
        run_experiment_2pt(
            n_values=n_values,
            n_sims=gen.n_sims,
            true_mu=gen.true_mu,
            prior_atoms=prior_atoms,
            prior_weights=prior_weights,
            weight_power=tpt.weight_power,
            sigma_power=tpt.sigma_power,
            si_min=tpt.si_min,
            si_max=tpt.si_max,
            n_workers=n_workers,
            output_dir=gen.output_dir
        )

    # Experiment 3: Gaussian Prior
    if gen.experiment in ['gaussian', 'all']:
        run_experiment_gaussian(
            n_values=n_values,
            n_sims=gen.n_sims,
            true_mu=gen.true_mu,
            si=gauss.si,
            si_min=gauss.si_min,
            si_max=gauss.si_max,
            n_workers=n_workers,
            output_dir=gen.output_dir
        )

    # Experiment 5: Three-Point Prior
    if gen.experiment in ['3pt', 'all']:
        tp = config.three_point
        prior_atoms_3pt = np.array(tp.prior_atoms)
        prior_weights_3pt = np.array(tp.prior_weights)
        run_experiment_3pt(
            n_values=n_values,
            n_sims=gen.n_sims,
            true_mu=gen.true_mu,
            prior_atoms=prior_atoms_3pt,
            prior_weights=prior_weights_3pt,
            si_min=tp.si_min,
            si_max=tp.si_max,
            n_workers=n_workers,
            output_dir=gen.output_dir
        )

    # Experiment 4: Quadratic Variance Prior
    if gen.experiment in ['quadratic', 'all']:
        run_experiment_quadratic(
            n_values=n_values,
            n_sims=gen.n_sims,
            true_mu=gen.true_mu,
            c=quad.c,
            si_min=quad.si_min,
            si_max=quad.si_max,
            n_workers=n_workers,
            output_dir=gen.output_dir
        )

    overall_elapsed = time.time() - overall_start
    hours, remainder = divmod(int(overall_elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)

    log("\n" + "=" * 60)
    log("All experiments completed!")
    log(f"Total runtime: {hours}h {minutes}m {seconds}s")
    log(f"Results saved to {gen.output_dir}")
    log(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60)


if __name__ == '__main__':
    # When executed as a script (including via SLURM submission), parse any
    # command-line arguments to override the default configuration. If no
    # arguments are provided, this reduces to the default configuration.
    args = parse_args()
    cfg = config_from_args(args)
    main(cfg)
