#!/usr/bin/env python3
"""
Likelihood Landscape Visualization for MLE with Known G

This script demonstrates the non-convex nature of the log-likelihood
when estimating μ in a heteroskedastic Gaussian model with known
scale mixture prior G.

Model: X_i ~ N(μ, σ_i^2) where σ_i ~ G (discrete mixture)
"""

import numpy as np

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.stats import norm
import argparse
import os


def mixture_log_likelihood(mu, X, prior_atoms, prior_weights):
    """
    Compute the average log-likelihood for the mixture model.
    
    L(μ) = (1/n) Σᵢ log[ Σⱼ wⱼ φ((Xᵢ - μ)/σⱼ) / σⱼ ]
    
    where φ is the standard normal pdf.
    """
    n = len(X)
    K = len(prior_atoms)
    
    # Compute mixture density for each observation
    # Shape: (n,)
    mixture_densities = np.zeros(n)
    for j in range(K):
        mixture_densities += prior_weights[j] * norm.pdf(X, loc=mu, scale=prior_atoms[j])
    
    # Avoid log(0)
    mixture_densities = np.maximum(mixture_densities, 1e-300)
    
    return np.mean(np.log(mixture_densities))


def generate_data(true_mu, prior_atoms, prior_weights, n, seed=None):
    """Generate data from a scale mixture of Gaussians."""
    rng = np.random.default_rng(seed)
    
    # Sample which component each observation comes from
    components = rng.choice(len(prior_atoms), size=n, p=prior_weights)
    sigmas = prior_atoms[components]
    
    # Generate observations
    X = rng.normal(true_mu, sigmas)
    
    return X, sigmas


def plot_likelihood_landscape(ax, X, prior_atoms, prior_weights, true_mu, 
                               mu_range, title, show_data=False):
    """Plot the log-likelihood as a function of μ."""
    
    # Compute log-likelihood over range of mu values
    mu_values = np.linspace(mu_range[0], mu_range[1], 500)
    ll_values = np.array([mixture_log_likelihood(mu, X, prior_atoms, prior_weights) 
                          for mu in mu_values])
    
    # Find the MLE
    mle_idx = np.argmax(ll_values)
    mle_mu = mu_values[mle_idx]
    mle_ll = ll_values[mle_idx]
    
    # Plot the landscape
    ax.plot(mu_values, ll_values, 'b-', linewidth=2, label='Log-likelihood')
    
    # Mark the true mu
    true_ll = mixture_log_likelihood(true_mu, X, prior_atoms, prior_weights)
    ax.axvline(true_mu, color='green', linestyle='--', linewidth=1.5, 
               label=f'True $\\mu^* = {true_mu}$')
    ax.plot(true_mu, true_ll, 'go', markersize=10, zorder=5)
    
    # Mark the MLE
    ax.axvline(mle_mu, color='red', linestyle=':', linewidth=1.5,
               label=f'MLE $\\hat{{\\mu}} = {mle_mu:.3f}$')
    ax.plot(mle_mu, mle_ll, 'r*', markersize=15, zorder=5)
    
    # Show data points as rug plot if requested
    if show_data:
        y_min = ax.get_ylim()[0]
        ax.plot(X, np.full_like(X, y_min), '|', color='gray', alpha=0.3, markersize=10)
    
    ax.set_xlabel('$\\mu$', fontsize=14)
    ax.set_ylabel('Average Log-Likelihood', fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    return mle_mu, mle_ll


def plot_negative_log_likelihood(ax, X, prior_atoms, prior_weights, true_mu, 
                                  mu_range, title):
    """Plot the negative log-likelihood (loss function view)."""
    
    mu_values = np.linspace(mu_range[0], mu_range[1], 500)
    nll_values = np.array([-mixture_log_likelihood(mu, X, prior_atoms, prior_weights) 
                           for mu in mu_values])
    
    # Find the MLE (minimum of NLL)
    mle_idx = np.argmin(nll_values)
    mle_mu = mu_values[mle_idx]
    mle_nll = nll_values[mle_idx]
    
    ax.plot(mu_values, nll_values, 'b-', linewidth=2)
    
    # Mark true mu
    true_nll = -mixture_log_likelihood(true_mu, X, prior_atoms, prior_weights)
    ax.axvline(true_mu, color='green', linestyle='--', linewidth=1.5,
               label=f'True $\\mu^*$')
    ax.plot(true_mu, true_nll, 'go', markersize=10, zorder=5)
    
    # Mark MLE
    ax.axvline(mle_mu, color='red', linestyle=':', linewidth=1.5,
               label=f'MLE')
    ax.plot(mle_mu, mle_nll, 'r*', markersize=15, zorder=5)
    
    # Reference: convex parabola for comparison
    # Scale to match the plot
    parabola = 0.5 * (mu_values - true_mu)**2
    parabola = parabola * (nll_values.max() - nll_values.min()) / (parabola.max() - parabola.min() + 1e-10)
    parabola = parabola - parabola.min() + nll_values.min()
    ax.plot(mu_values, parabola, '--', color='gray', alpha=0.5, linewidth=1,
            label='Convex reference')
    
    ax.set_xlabel('$\\mu$', fontsize=14)
    ax.set_ylabel('Negative Log-Likelihood (Loss)', fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize likelihood landscape for MLE with known G')
    parser.add_argument('--n', type=int, default=100,
                        help='Sample size (default: 100)')
    parser.add_argument('--true-mu', type=float, default=0.0,
                        help='True mean (default: 0.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Output directory (default: ./results)')
    parser.add_argument('--mu-range', type=float, nargs=2, default=[-3, 3],
                        help='Range of mu to plot (default: -3 3)')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    true_mu = args.true_mu
    n = args.n
    mu_range = args.mu_range
    
    # =========================================================================
    # Define priors
    # =========================================================================
    
    # Two-point prior: small variance (signal) vs large variance (noise)
    two_point_atoms = np.array([0.5, 5.0])
    two_point_weights = np.array([0.3, 0.7])  # 30% signal, 70% noise
    
    # Three-point prior: small, medium, large variance
    three_point_atoms = np.array([0.3, 1.5, 8.0])
    three_point_weights = np.array([0.2, 0.3, 0.5])
    
    # Extreme two-point: very small vs very large (adversarial)
    extreme_atoms = np.array([0.1, 10.0])
    extreme_weights = np.array([0.1, 0.9])  # 10% precise, 90% noisy
    
    priors = [
        ('Two-Point Prior\n$\\sigma \\in \\{0.5, 5.0\\}$, $w = (0.3, 0.7)$',
         two_point_atoms, two_point_weights),
        ('Three-Point Prior\n$\\sigma \\in \\{0.3, 1.5, 8.0\\}$, $w = (0.2, 0.3, 0.5)$',
         three_point_atoms, three_point_weights),
        ('Extreme Two-Point Prior\n$\\sigma \\in \\{0.1, 10.0\\}$, $w = (0.1, 0.9)$',
         extreme_atoms, extreme_weights),
    ]
    
    # =========================================================================
    # Figure 1: Log-likelihood landscapes for different priors
    # =========================================================================
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, (title, atoms, weights) in zip(axes1, priors):
        X, _ = generate_data(true_mu, atoms, weights, n, seed=args.seed)
        plot_likelihood_landscape(ax, X, atoms, weights, true_mu, mu_range, 
                                  title, show_data=True)
    
    fig1.suptitle(f'Log-Likelihood Landscapes (n = {n}, $\\mu^* = {true_mu}$)', 
                  fontsize=16, y=1.02)
    plt.tight_layout()
    
    plot_file1 = os.path.join(args.output_dir, 'likelihood_landscape_comparison.png')
    plt.savefig(plot_file1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_file1}")
    
    # =========================================================================
    # Figure 2: Effect of sample size on landscape smoothness
    # =========================================================================
    sample_sizes = [20, 100, 500, 2000]
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    axes2 = axes2.flatten()
    
    # Use two-point prior
    atoms, weights = two_point_atoms, two_point_weights
    
    for ax, n_samples in zip(axes2, sample_sizes):
        X, _ = generate_data(true_mu, atoms, weights, n_samples, seed=args.seed)
        title = f'n = {n_samples}'
        plot_likelihood_landscape(ax, X, atoms, weights, true_mu, mu_range, title)
    
    fig2.suptitle(f'Effect of Sample Size on Likelihood Landscape\n'
                  f'Two-Point Prior: $\\sigma \\in \\{{0.5, 5.0\\}}$, $w = (0.3, 0.7)$',
                  fontsize=14, y=1.02)
    plt.tight_layout()
    
    plot_file2 = os.path.join(args.output_dir, 'likelihood_landscape_sample_size.png')
    plt.savefig(plot_file2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_file2}")
    
    # =========================================================================
    # Figure 3: Multiple random seeds to show landscape variability
    # =========================================================================
    fig3, axes3 = plt.subplots(2, 3, figsize=(18, 10))
    
    atoms, weights = extreme_atoms, extreme_weights
    n_fixed = 50  # Small n to show more variability
    
    for i, ax in enumerate(axes3.flatten()):
        seed = args.seed + i * 100
        X, _ = generate_data(true_mu, atoms, weights, n_fixed, seed=seed)
        title = f'Random Seed {seed}'
        plot_likelihood_landscape(ax, X, atoms, weights, true_mu, mu_range, title)
    
    fig3.suptitle(f'Likelihood Landscape Variability Across Samples (n = {n_fixed})\n'
                  f'Extreme Prior: $\\sigma \\in \\{{0.1, 10.0\\}}$, $w = (0.1, 0.9)$',
                  fontsize=14, y=1.02)
    plt.tight_layout()
    
    plot_file3 = os.path.join(args.output_dir, 'likelihood_landscape_variability.png')
    plt.savefig(plot_file3, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_file3}")
    
    # =========================================================================
    # Figure 4: Negative log-likelihood (loss function perspective)
    # =========================================================================
    fig4, axes4 = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, (title, atoms, weights) in zip(axes4, priors):
        X, _ = generate_data(true_mu, atoms, weights, n, seed=args.seed)
        plot_negative_log_likelihood(ax, X, atoms, weights, true_mu, mu_range, title)
    
    fig4.suptitle(f'Negative Log-Likelihood (Loss Function View, n = {n})\n'
                  f'Gray dashed line shows a convex reference for comparison',
                  fontsize=14, y=1.02)
    plt.tight_layout()
    
    plot_file4 = os.path.join(args.output_dir, 'nll_landscape_comparison.png')
    plt.savefig(plot_file4, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_file4}")
    
    # =========================================================================
    # Figure 5: Zoomed view showing local structure
    # =========================================================================
    fig5, axes5 = plt.subplots(1, 2, figsize=(14, 5))
    
    # Generate data with extreme prior
    X, sigmas = generate_data(true_mu, extreme_atoms, extreme_weights, 200, seed=args.seed)
    
    # Wide view
    plot_likelihood_landscape(axes5[0], X, extreme_atoms, extreme_weights, 
                              true_mu, [-5, 5], 'Wide View', show_data=True)
    
    # Zoomed view around true mu
    plot_likelihood_landscape(axes5[1], X, extreme_atoms, extreme_weights,
                              true_mu, [-1, 1], 'Zoomed View (±1 around $\\mu^*$)')
    
    fig5.suptitle(f'Extreme Two-Point Prior: Local vs Global Structure (n = 200)',
                  fontsize=14, y=1.02)
    plt.tight_layout()
    
    plot_file5 = os.path.join(args.output_dir, 'likelihood_landscape_zoom.png')
    plt.savefig(plot_file5, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_file5}")
    
    # =========================================================================
    # Figure 6: Overlay multiple sample realizations
    # =========================================================================
    fig6, ax6 = plt.subplots(figsize=(12, 7))
    
    atoms, weights = two_point_atoms, two_point_weights
    n_realizations = 20
    n_samples = 50
    mu_values = np.linspace(mu_range[0], mu_range[1], 300)
    
    # Plot multiple realizations with transparency
    for i in range(n_realizations):
        X, _ = generate_data(true_mu, atoms, weights, n_samples, seed=args.seed + i)
        ll_values = np.array([mixture_log_likelihood(mu, X, atoms, weights) 
                              for mu in mu_values])
        ax6.plot(mu_values, ll_values, 'b-', alpha=0.2, linewidth=1)
    
    # Plot population likelihood (very large sample)
    X_large, _ = generate_data(true_mu, atoms, weights, 10000, seed=args.seed)
    ll_large = np.array([mixture_log_likelihood(mu, X_large, atoms, weights) 
                         for mu in mu_values])
    ax6.plot(mu_values, ll_large, 'r-', linewidth=3, label='Large sample (n=10000)')
    
    ax6.axvline(true_mu, color='green', linestyle='--', linewidth=2,
                label=f'True $\\mu^* = {true_mu}$')
    
    ax6.set_xlabel('$\\mu$', fontsize=14)
    ax6.set_ylabel('Average Log-Likelihood', fontsize=14)
    ax6.set_title(f'Likelihood Landscape: {n_realizations} Realizations (n = {n_samples} each)\n'
                  f'Two-Point Prior: $\\sigma \\in \\{{0.5, 5.0\\}}$, $w = (0.3, 0.7)$',
                  fontsize=14)
    ax6.legend(fontsize=12)
    ax6.grid(True, alpha=0.3)
    
    plot_file6 = os.path.join(args.output_dir, 'likelihood_landscape_realizations.png')
    plt.savefig(plot_file6, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_file6}")
    
    print(f"\nAll plots saved to {args.output_dir}/")
    print("\nKey observations:")
    print("1. The log-likelihood is generally NOT convex (compare to gray parabola)")
    print("2. With small n, the landscape can have local maxima and flat regions")
    print("3. Extreme priors (large variance ratio) create more challenging landscapes")
    print("4. As n increases, the landscape smooths and approaches the population optimum")


if __name__ == '__main__':
    main()
