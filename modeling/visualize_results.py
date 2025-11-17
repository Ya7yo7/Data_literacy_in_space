"""visualize_results.py

Generate comparison plots and spatial maps for model evaluation results.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_score_distributions(result_df: pd.DataFrame, output_dir: Path):
    """Plot score distributions for all three models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    models = ['nasa', 'rf', 'sim']
    titles = ['NASA Constraints', 'Random Forest', 'Similarity (KDE)']
    colors = ['blue', 'green', 'purple']
    
    for ax, model, title, color in zip(axes, models, titles, colors):
        scores = result_df[f'{model}_score']
        
        ax.hist(scores, bins=100, alpha=0.7, color=color, edgecolor='black')
        ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
        ax.set_xlabel('Score', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Add statistics text
        mean_val = scores.mean()
        median_val = scores.median()
        ax.text(0.02, 0.98, f'Mean: {mean_val:.3f}\nMedian: {median_val:.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'score_distributions.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: score_distributions.png")
    plt.close()


def plot_consensus_map(result_df: pd.DataFrame, output_dir: Path):
    """Plot spatial map colored by consensus count."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sample for visualization (too many points)
    sample_size = min(50000, len(result_df))
    sample_df = result_df.sample(n=sample_size, random_state=42)
    
    scatter = ax.scatter(
        sample_df['long_east_deg'],
        sample_df['lat_north_deg'],
        c=sample_df['consensus_count'],
        cmap='RdYlGn',
        s=1,
        alpha=0.6,
        vmin=0,
        vmax=3
    )
    
    cbar = plt.colorbar(scatter, ax=ax, label='Models in Agreement')
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['0 (All Reject)', '1', '2', '3 (All Accept)'])
    
    ax.set_xlabel('Longitude East (degrees)', fontsize=12)
    ax.set_ylabel('Latitude North (degrees)', fontsize=12)
    ax.set_title('Model Consensus Map\n(Sample of 50k sites)', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'consensus_map.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: consensus_map.png")
    plt.close()


def plot_individual_predictions(result_df: pd.DataFrame, output_dir: Path):
    """Plot individual model predictions on spatial map."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    models = ['nasa', 'rf', 'sim']
    titles = ['NASA Constraints', 'Random Forest', 'Similarity (KDE)']
    
    # Sample for visualization
    sample_size = min(50000, len(result_df))
    sample_df = result_df.sample(n=sample_size, random_state=42)
    
    for ax, model, title in zip(axes, models, titles):
        preds = sample_df[f'{model}_pred']
        
        # Plot unsuitable (gray) first, then suitable (green) on top
        unsuitable = sample_df[preds == 0]
        suitable = sample_df[preds == 1]
        
        ax.scatter(unsuitable['long_east_deg'], unsuitable['lat_north_deg'],
                  c='lightgray', s=1, alpha=0.5, label='Unsuitable')
        ax.scatter(suitable['long_east_deg'], suitable['lat_north_deg'],
                  c='green', s=1, alpha=0.7, label='Suitable')
        
        ax.set_xlabel('Longitude East (degrees)', fontsize=10)
        ax.set_ylabel('Latitude North (degrees)', fontsize=10)
        ax.set_title(f'{title}\n{preds.sum():,} suitable ({100*preds.mean():.1f}%)',
                    fontsize=12, fontweight='bold')
        ax.legend(markerscale=5)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'individual_predictions_map.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: individual_predictions_map.png")
    plt.close()


def plot_score_correlations(result_df: pd.DataFrame, output_dir: Path):
    """Plot pairwise score correlations."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    pairs = [('nasa', 'rf'), ('nasa', 'sim'), ('rf', 'sim')]
    
    # Sample for scatter plots
    sample_size = min(10000, len(result_df))
    sample_df = result_df.sample(n=sample_size, random_state=42)
    
    for ax, (model1, model2) in zip(axes, pairs):
        x = sample_df[f'{model1}_score']
        y = sample_df[f'{model2}_score']
        
        ax.scatter(x, y, alpha=0.3, s=1)
        ax.set_xlabel(f'{model1.upper()} Score', fontsize=12)
        ax.set_ylabel(f'{model2.upper()} Score', fontsize=12)
        ax.set_title(f'{model1.upper()} vs {model2.upper()}', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Add correlation coefficient
        corr = np.corrcoef(x, y)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'score_correlations.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: score_correlations.png")
    plt.close()


def plot_feature_space(result_df: pd.DataFrame, output_dir: Path):
    """Plot altitude-slope decision space for all models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    models = ['nasa', 'rf', 'sim']
    titles = ['NASA Constraints', 'Random Forest', 'Similarity (KDE)']
    
    # Sample for visualization
    sample_size = min(20000, len(result_df))
    sample_df = result_df.sample(n=sample_size, random_state=42)
    
    for ax, model, title in zip(axes, models, titles):
        preds = sample_df[f'{model}_pred']
        
        # Plot by prediction
        unsuitable = sample_df[preds == 0]
        suitable = sample_df[preds == 1]
        
        ax.scatter(unsuitable['slope_deg'], unsuitable['altitude_m'],
                  c='red', s=1, alpha=0.3, label='Unsuitable')
        ax.scatter(suitable['slope_deg'], suitable['altitude_m'],
                  c='green', s=1, alpha=0.5, label='Suitable')
        
        ax.set_xlabel('Slope (degrees)', fontsize=10)
        ax.set_ylabel('Altitude (meters)', fontsize=10)
        ax.set_title(f'{title}', fontsize=12, fontweight='bold')
        ax.legend(markerscale=5)
        ax.grid(alpha=0.3)
        
        # Add NASA constraint lines for reference (first plot only)
        if model == 'nasa':
            ax.axvline(5, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='NASA slope limit')
            ax.axhline(-1300, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='NASA altitude limit')
            ax.legend(markerscale=5, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_space_altitude_slope.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: feature_space_altitude_slope.png")
    plt.close()


def plot_consensus_breakdown(result_df: pd.DataFrame, output_dir: Path):
    """Plot consensus category breakdown."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Consensus count bar chart
    consensus_counts = result_df['consensus_count'].value_counts().sort_index()
    colors_bar = ['red', 'orange', 'yellow', 'green']
    
    ax1.bar(consensus_counts.index, consensus_counts.values, color=colors_bar, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Number of Models Predicting Suitable', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Consensus Distribution', fontsize=14, fontweight='bold')
    ax1.set_xticks([0, 1, 2, 3])
    ax1.grid(axis='y', alpha=0.3)
    
    # Add percentages on bars
    total = len(result_df)
    for i, count in enumerate(consensus_counts.values):
        pct = 100 * count / total
        ax1.text(i, count, f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Consensus category pie chart
    category_counts = result_df['consensus_category'].value_counts()
    category_labels = ['All Reject', 'Minority Accept (1/3)', 'Majority Accept (2/3)', 'All Accept']
    colors_pie = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
    
    # Map categories to colors
    category_order = ['all_reject', 'minority_accept', 'majority_accept', 'all_accept']
    category_values = [category_counts.get(cat, 0) for cat in category_order]
    
    wedges, texts, autotexts = ax2.pie(
        category_values,
        labels=category_labels,
        colors=colors_pie,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 11}
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax2.set_title('Consensus Categories', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'consensus_breakdown.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: consensus_breakdown.png")
    plt.close()


def main():
    """Generate all visualization plots."""
    print("="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Load results
    results_dir = Path(__file__).parent / 'output' / 'results'
    plots_dir = Path(__file__).parent / 'output' / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📂 Loading evaluation results...")
    results_path = results_dir / 'candidate_evaluations.csv'
    result_df = pd.read_csv(results_path)
    print(f"  ✓ Loaded {len(result_df):,} evaluated sites")
    
    # Set plotting style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    print("\n📊 Generating plots...")
    
    plot_score_distributions(result_df, plots_dir)
    plot_consensus_breakdown(result_df, plots_dir)
    plot_consensus_map(result_df, plots_dir)
    plot_individual_predictions(result_df, plots_dir)
    plot_score_correlations(result_df, plots_dir)
    plot_feature_space(result_df, plots_dir)
    
    print(f"\n✓ All plots saved to: {plots_dir}")
    print("\n" + "="*70)
    print("✅ VISUALIZATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
