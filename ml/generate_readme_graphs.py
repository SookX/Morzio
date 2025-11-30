#!/usr/bin/env python3
"""
Generate visualization graphs for the Morzio README.
Creates professional seaborn-styled plots explaining the ML model and installment formula.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import math

# Set seaborn style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def sigmoid(x):
    """Sigmoid function with clipping for numerical stability."""
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def affordability_factor(rho, rho_0=0.45, k_f=5.0):
    """Calculate affordability factor f(ρ)."""
    return 1.0 - sigmoid(k_f * (rho - rho_0))


def anomaly_weight(rho, rho_1=0.5, k_a=5.0):
    """Calculate anomaly weight w(ρ)."""
    return sigmoid(k_a * (rho - rho_1))


def anomaly_factor(epsilon, rho):
    """Calculate anomaly factor a(ε, ρ)."""
    w = anomaly_weight(rho)
    return 1.0 - w * (epsilon ** 2)


def normalize_anomaly_score(score, threshold=2.0, k=1.2):
    """Normalize anomaly score to [0, 1]."""
    return sigmoid(k * (score - threshold))


def calculate_installments(rho, epsilon, max_n=48):
    """Calculate final installments."""
    f_rho = affordability_factor(rho)
    a_e_rho = anomaly_factor(epsilon, rho)
    n_star = max_n * f_rho * a_e_rho
    return np.clip(n_star, 0, max_n)


def plot_affordability_factor():
    """Plot 1: Affordability Factor vs Ratio."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rho = np.linspace(0, 1.5, 500)
    f_rho = affordability_factor(rho)
    
    ax.plot(rho, f_rho, linewidth=3, color='#2ecc71', label=r'$f(\rho) = 1 - \sigma(k_f \cdot (\rho - \rho_0))$')
    ax.axvline(x=0.45, color='#e74c3c', linestyle='--', linewidth=2, label=r'$\rho_0 = 0.45$ (threshold)')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    
    # Annotate regions
    ax.fill_between(rho[rho <= 0.3], f_rho[rho <= 0.3], alpha=0.3, color='#2ecc71', label='Low risk zone')
    ax.fill_between(rho[rho >= 0.6], f_rho[rho >= 0.6], alpha=0.3, color='#e74c3c', label='High risk zone')
    
    ax.set_xlabel(r'Affordability Ratio $\rho = \frac{|A|}{I + 1}$', fontsize=14)
    ax.set_ylabel(r'Affordability Factor $f(\rho)$', fontsize=14)
    ax.set_title('Affordability Factor: How Purchase/Income Ratio Affects Installments', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlim(0, 1.5)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "affordability_curve.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated: affordability_curve.png")


def plot_anomaly_factor_heatmap():
    """Plot 2: Anomaly Factor Heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    rho = np.linspace(0, 1.2, 100)
    epsilon = np.linspace(0, 1, 100)
    RHO, EPS = np.meshgrid(rho, epsilon)
    
    A = anomaly_factor(EPS, RHO)
    
    heatmap = ax.contourf(RHO, EPS, A, levels=20, cmap='RdYlGn')
    cbar = plt.colorbar(heatmap, ax=ax, label=r'Anomaly Factor $a(\epsilon, \rho)$')
    
    # Add contour lines
    contours = ax.contour(RHO, EPS, A, levels=[0.2, 0.4, 0.6, 0.8], colors='black', linewidths=1, alpha=0.5)
    ax.clabel(contours, inline=True, fontsize=10, fmt='%.1f')
    
    ax.set_xlabel(r'Affordability Ratio $\rho$', fontsize=14)
    ax.set_ylabel(r'Normalized Anomaly Score $\epsilon$', fontsize=14)
    ax.set_title('Anomaly Factor: Risk Penalization Based on Spending Patterns', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "anomaly_factor_heatmap.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated: anomaly_factor_heatmap.png")


def plot_installment_surface():
    """Plot 3: 3D Installment Surface."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    rho = np.linspace(0, 1.2, 50)
    epsilon = np.linspace(0, 1, 50)
    RHO, EPS = np.meshgrid(rho, epsilon)
    
    N = calculate_installments(RHO, EPS)
    
    surf = ax.plot_surface(RHO, EPS, N, cmap='viridis', edgecolor='none', alpha=0.9)
    
    ax.set_xlabel(r'Affordability Ratio $\rho$', fontsize=12, labelpad=10)
    ax.set_ylabel(r'Anomaly Score $\epsilon$', fontsize=12, labelpad=10)
    ax.set_zlabel('Max Installments (N)', fontsize=12, labelpad=10)
    ax.set_title('Installment Decision Surface\n$N^* = N_{max} \\cdot f(\\rho) \\cdot a(\\epsilon, \\rho)$', fontsize=16, fontweight='bold')
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Installments')
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "installment_surface.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated: installment_surface.png")


def plot_installment_heatmap():
    """Plot 4: Installment Decision Heatmap (more readable 2D version)."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    rho = np.linspace(0, 1.2, 100)
    epsilon = np.linspace(0, 1, 100)
    RHO, EPS = np.meshgrid(rho, epsilon)
    
    N = calculate_installments(RHO, EPS)
    
    # Apply decline logic
    decline_mask = (EPS > 0.90) & (RHO > 0.2)
    N[decline_mask] = 0
    
    heatmap = ax.contourf(RHO, EPS, N, levels=np.arange(0, 52, 4), cmap='YlGnBu')
    cbar = plt.colorbar(heatmap, ax=ax, label='Max Installments (N)', ticks=np.arange(0, 52, 8))
    
    # Mark decline zone
    ax.fill_between([0.2, 1.2], [0.9, 0.9], [1.0, 1.0], alpha=0.4, color='red', label='Decline Zone')
    
    # Add key contour lines
    contours = ax.contour(RHO, EPS, N, levels=[12, 24, 36, 44], colors='white', linewidths=1.5)
    ax.clabel(contours, inline=True, fontsize=11, fmt='%d')
    
    ax.set_xlabel(r'Affordability Ratio $\rho = \frac{\text{Purchase}}{\text{Income} + 1}$', fontsize=14)
    ax.set_ylabel(r'Normalized Anomaly Score $\epsilon$', fontsize=14)
    ax.set_title('Installment Decision Map: Maximum Installments by Risk Profile', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "installment_heatmap.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated: installment_heatmap.png")


def plot_anomaly_score_distribution():
    """Plot 5: Simulated Anomaly Score Distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Simulate anomaly scores (most normal, some anomalous)
    np.random.seed(42)
    normal_scores = np.random.exponential(scale=0.8, size=5000) + 0.5
    anomaly_scores = np.random.normal(loc=8, scale=2, size=200)
    anomaly_scores = anomaly_scores[anomaly_scores > 4]
    
    all_scores = np.concatenate([normal_scores, anomaly_scores])
    
    # Plot histogram
    ax.hist(all_scores, bins=50, density=True, alpha=0.7, color='#3498db', edgecolor='white', label='Score Distribution')
    
    # Add threshold lines
    ax.axvline(x=2.0, color='#f39c12', linestyle='--', linewidth=2.5, label=r'$\tau = 2.0$ (anomaly threshold)')
    ax.axvline(x=5.0, color='#e74c3c', linestyle='--', linewidth=2.5, label='High risk threshold')
    
    # Add regions
    ax.fill_betweenx([0, 0.5], 0, 2, alpha=0.2, color='#2ecc71', label='Normal behavior')
    ax.fill_betweenx([0, 0.5], 5, 15, alpha=0.2, color='#e74c3c', label='Anomalous behavior')
    
    ax.set_xlabel('Anomaly Score (S)', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_title('Anomaly Score Distribution: Normal vs Suspicious Users', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 0.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "anomaly_distribution.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated: anomaly_distribution.png")


def plot_vae_architecture():
    """Plot 6: VAE Architecture Diagram."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    encoder_color = '#3498db'
    latent_color = '#9b59b6'
    decoder_color = '#2ecc71'
    
    # Input
    ax.add_patch(plt.Rectangle((0.5, 4), 1.5, 2, facecolor='#ecf0f1', edgecolor='black', linewidth=2))
    ax.text(1.25, 5, 'Input\n(20)', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Encoder layers
    encoder_positions = [(2.5, 3.5, 'FC(20→64)\nSiLU'), (4, 3.5, 'FC(64→128)\nSiLU'), (5.5, 3.5, 'FC(128→128)')]
    for i, (x, y, label) in enumerate(encoder_positions):
        ax.add_patch(plt.Rectangle((x, y), 1.2, 3, facecolor=encoder_color, edgecolor='black', linewidth=2, alpha=0.8))
        ax.text(x + 0.6, y + 1.5, label, ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    
    # Encoder label
    ax.text(4, 7.5, 'ENCODER', ha='center', va='center', fontsize=14, fontweight='bold', color=encoder_color)
    
    # Latent space
    ax.add_patch(plt.Rectangle((7, 2.5), 1.5, 2, facecolor=latent_color, edgecolor='black', linewidth=2, alpha=0.8))
    ax.text(7.75, 3.5, 'μ', ha='center', va='center', fontsize=14, color='white', fontweight='bold')
    ax.add_patch(plt.Rectangle((7, 5.5), 1.5, 2, facecolor=latent_color, edgecolor='black', linewidth=2, alpha=0.8))
    ax.text(7.75, 6.5, 'log σ²', ha='center', va='center', fontsize=12, color='white', fontweight='bold')
    
    # Sampling
    ax.add_patch(plt.Circle((9, 5), 0.6, facecolor='#f39c12', edgecolor='black', linewidth=2))
    ax.text(9, 5, 'z', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(9, 3.5, 'z = μ + σ·ε', ha='center', va='center', fontsize=10, style='italic')
    
    # Latent label
    ax.text(8.25, 8.5, 'LATENT SPACE', ha='center', va='center', fontsize=14, fontweight='bold', color=latent_color)
    
    # Decoder layers
    decoder_positions = [(10, 3.5, 'FC(256→128)\nSiLU'), (11.5, 3.5, 'FC(128→64)\nSiLU'), (13, 3.5, 'FC(64→20)')]
    for i, (x, y, label) in enumerate(decoder_positions):
        ax.add_patch(plt.Rectangle((x, y), 1.2, 3, facecolor=decoder_color, edgecolor='black', linewidth=2, alpha=0.8))
        ax.text(x + 0.6, y + 1.5, label, ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    
    # Decoder label
    ax.text(11.5, 7.5, 'DECODER', ha='center', va='center', fontsize=14, fontweight='bold', color=decoder_color)
    
    # Arrows
    arrow_style = dict(arrowstyle='->', color='black', lw=2)
    ax.annotate('', xy=(2.5, 5), xytext=(2, 5), arrowprops=arrow_style)
    ax.annotate('', xy=(4, 5), xytext=(3.7, 5), arrowprops=arrow_style)
    ax.annotate('', xy=(5.5, 5), xytext=(5.2, 5), arrowprops=arrow_style)
    ax.annotate('', xy=(7, 5), xytext=(6.7, 5), arrowprops=arrow_style)
    ax.annotate('', xy=(9.4, 5), xytext=(8.5, 5), arrowprops=arrow_style)
    ax.annotate('', xy=(10, 5), xytext=(9.6, 5), arrowprops=arrow_style)
    ax.annotate('', xy=(11.5, 5), xytext=(11.2, 5), arrowprops=arrow_style)
    ax.annotate('', xy=(13, 5), xytext=(12.7, 5), arrowprops=arrow_style)
    
    # Output
    ax.add_patch(plt.Rectangle((14.2, 4), 1.5, 2, facecolor='#ecf0f1', edgecolor='black', linewidth=2))
    ax.text(14.95, 5, 'Output\n(20)', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.annotate('', xy=(14.2, 5), xytext=(14.2, 5), arrowprops=arrow_style)
    
    # Loss annotation
    ax.text(7.5, 0.8, r'$\mathcal{L} = \text{MSE}(x, \hat{x}) + D_{KL}(q(z|x) \| p(z))$', 
            ha='center', va='center', fontsize=14, style='italic', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_title('Variational Autoencoder (VAE) Architecture for Anomaly Detection', fontsize=18, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "vae_architecture.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated: vae_architecture.png")


def plot_feature_importance():
    """Plot 7: Feature Importance/Weights."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    features = [
        'spend_to_income_ratio_30d', 'spend_to_income_ratio_90d',
        'avg_txn_over_income_ratio_90d', 'current_txn_amount', 'current_txn_mcc',
        'estimated_monthly_income', 'last_inflow_amount', 'days_since_last_inflow',
        'credit_score', 'total_spend_30d', 'total_spend_90d', 'transaction_count_30d',
        'transaction_count_90d', 'avg_txn_amount_30d', 'avg_txn_amount_90d',
        'max_txn_amount_90d', 'txn_amount_median_90d', 'spend_volatility_30d',
        'spend_volatility_90d', 'txn_count_30d_norm'
    ]
    
    weights = [2.0, 2.0, 1.5, 1.5, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0, 
               1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    
    colors = ['#e74c3c' if w > 1 else '#3498db' for w in weights]
    
    y_pos = np.arange(len(features))
    ax.barh(y_pos, weights, color=colors, edgecolor='white', height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=10)
    ax.set_xlabel('Feature Weight', fontsize=14)
    ax.set_title('Feature Importance Weights in Anomaly Detection', fontsize=16, fontweight='bold')
    ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlim(0, 2.5)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='High importance (>1.0)'),
        Patch(facecolor='#3498db', label='Standard importance (1.0)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated: feature_importance.png")


def main():
    print("\n" + "=" * 60)
    print("GENERATING README VISUALIZATIONS")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    plot_affordability_factor()
    plot_anomaly_factor_heatmap()
    plot_installment_surface()
    plot_installment_heatmap()
    plot_anomaly_score_distribution()
    plot_vae_architecture()
    plot_feature_importance()
    
    print("\n" + "=" * 60)
    print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nFiles saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

