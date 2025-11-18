"""pairwise_agreement.py

Analyze which pairs of models agree most often.
Generate detailed breakdown of model agreement patterns.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def analyze_pairwise_agreement(result_df: pd.DataFrame) -> dict:
    """
    Analyze pairwise agreement between models.
    
    Returns:
        Dictionary with agreement statistics
    """
    n_total = len(result_df)
    
    # Extract predictions
    nasa_pred = result_df['nasa_pred'].values
    rf_pred = result_df['rf_pred'].values
    sim_pred = result_df['sim_pred'].values
    
    # Pairwise agreement (both say same thing)
    nasa_rf_agree = (nasa_pred == rf_pred).sum()
    nasa_sim_agree = (nasa_pred == sim_pred).sum()
    rf_sim_agree = (rf_pred == sim_pred).sum()
    
    # Both accept
    nasa_rf_both_accept = ((nasa_pred == 1) & (rf_pred == 1)).sum()
    nasa_sim_both_accept = ((nasa_pred == 1) & (sim_pred == 1)).sum()
    rf_sim_both_accept = ((rf_pred == 1) & (sim_pred == 1)).sum()
    
    # Both reject
    nasa_rf_both_reject = ((nasa_pred == 0) & (rf_pred == 0)).sum()
    nasa_sim_both_reject = ((nasa_pred == 0) & (sim_pred == 0)).sum()
    rf_sim_both_reject = ((rf_pred == 0) & (sim_pred == 0)).sum()
    
    # Disagreement patterns
    nasa_accept_rf_reject = ((nasa_pred == 1) & (rf_pred == 0)).sum()
    nasa_reject_rf_accept = ((nasa_pred == 0) & (rf_pred == 1)).sum()
    
    nasa_accept_sim_reject = ((nasa_pred == 1) & (sim_pred == 0)).sum()
    nasa_reject_sim_accept = ((nasa_pred == 0) & (sim_pred == 1)).sum()
    
    rf_accept_sim_reject = ((rf_pred == 1) & (sim_pred == 0)).sum()
    rf_reject_sim_accept = ((rf_pred == 0) & (sim_pred == 1)).sum()
    
    return {
        'total': n_total,
        'nasa_rf': {
            'agree': nasa_rf_agree,
            'agree_pct': 100 * nasa_rf_agree / n_total,
            'both_accept': nasa_rf_both_accept,
            'both_reject': nasa_rf_both_reject,
            'nasa_only': nasa_accept_rf_reject,
            'rf_only': nasa_reject_rf_accept
        },
        'nasa_sim': {
            'agree': nasa_sim_agree,
            'agree_pct': 100 * nasa_sim_agree / n_total,
            'both_accept': nasa_sim_both_accept,
            'both_reject': nasa_sim_both_reject,
            'nasa_only': nasa_accept_sim_reject,
            'sim_only': nasa_reject_sim_accept
        },
        'rf_sim': {
            'agree': rf_sim_agree,
            'agree_pct': 100 * rf_sim_agree / n_total,
            'both_accept': rf_sim_both_accept,
            'both_reject': rf_sim_both_reject,
            'rf_only': rf_accept_sim_reject,
            'sim_only': rf_reject_sim_accept
        }
    }


def plot_pairwise_agreement(stats: dict, result_df: pd.DataFrame, output_dir: Path):
    """Generate pairwise agreement visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract predictions for combination analysis
    nasa_pred = result_df['nasa_pred'].values
    rf_pred = result_df['rf_pred'].values
    sim_pred = result_df['sim_pred'].values
    
    # Prepare data
    pairs = ['NASA vs RF', 'NASA vs SIM', 'RF vs SIM']
    agreement_pcts = [
        stats['nasa_rf']['agree_pct'],
        stats['nasa_sim']['agree_pct'],
        stats['rf_sim']['agree_pct']
    ]
    
    # 1. Overall agreement bar chart
    ax = axes[0, 0]
    colors = ['#ff6b6b', '#4ecdc4', '#95e1d3']
    bars = ax.bar(pairs, agreement_pcts, color=colors, edgecolor='black', linewidth=2)
    ax.set_ylabel('Agreement (%)', fontsize=12, fontweight='bold')
    ax.set_title('Overall Model Agreement\n(% of sites where both models give same answer)', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    # Add percentage labels on bars
    for bar, pct in zip(bars, agreement_pcts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{pct:.1f}%', ha='center', va='bottom', 
                fontsize=12, fontweight='bold')
    
    # 2. Stacked bar: agreement breakdown
    ax = axes[0, 1]
    
    both_accept = [
        stats['nasa_rf']['both_accept'],
        stats['nasa_sim']['both_accept'],
        stats['rf_sim']['both_accept']
    ]
    both_reject = [
        stats['nasa_rf']['both_reject'],
        stats['nasa_sim']['both_reject'],
        stats['rf_sim']['both_reject']
    ]
    
    x_pos = np.arange(len(pairs))
    width = 0.6
    
    p1 = ax.bar(x_pos, both_accept, width, label='Both Accept', color='#2ecc71')
    p2 = ax.bar(x_pos, both_reject, width, bottom=both_accept, label='Both Reject', color='#e74c3c')
    
    ax.set_ylabel('Number of Sites', fontsize=12, fontweight='bold')
    ax.set_title('Agreement Breakdown\n(Both Accept vs Both Reject)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(pairs)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add counts on bars
    for i, (accept, reject) in enumerate(zip(both_accept, both_reject)):
        # Both accept
        if accept > 0:
            ax.text(i, accept/2, f'{accept:,}', ha='center', va='center',
                   color='white', fontweight='bold', fontsize=10)
        # Both reject
        if reject > 0:
            ax.text(i, accept + reject/2, f'{reject:,}', ha='center', va='center',
                   color='white', fontweight='bold', fontsize=10)
    
    # 3. Disagreement patterns (who accepts when other rejects)
    ax = axes[1, 0]
    
    # NASA vs RF
    nasa_only_rf = stats['nasa_rf']['nasa_only']
    rf_only_nasa = stats['nasa_rf']['rf_only']
    
    # NASA vs SIM
    nasa_only_sim = stats['nasa_sim']['nasa_only']
    sim_only_nasa = stats['nasa_sim']['sim_only']
    
    # RF vs SIM
    rf_only_sim = stats['rf_sim']['rf_only']
    sim_only_rf = stats['rf_sim']['sim_only']
    
    x_labels = ['NASA\nvs RF', 'NASA\nvs SIM', 'RF\nvs SIM']
    model1_only = [nasa_only_rf, nasa_only_sim, rf_only_sim]
    model2_only = [rf_only_nasa, sim_only_nasa, sim_only_rf]
    
    x_pos = np.arange(len(x_labels))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, model1_only, width, 
                   label='First model accepts, second rejects', color='#3498db')
    bars2 = ax.bar(x_pos + width/2, model2_only, width,
                   label='Second model accepts, first rejects', color='#e67e22')
    
    ax.set_ylabel('Number of Sites', fontsize=12, fontweight='bold')
    ax.set_title('Disagreement Patterns\n(One accepts, other rejects)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Add counts on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 5000,
                       f'{int(height):,}', ha='center', va='bottom', 
                       fontsize=9, fontweight='bold')
    
    # 4. Venn-style diagram showing all combinations
    ax = axes[1, 1]
    ax.axis('off')
    
    # Count all 8 possible combinations
    n000 = ((nasa_pred == 0) & (rf_pred == 0) & (sim_pred == 0)).sum()  # All reject
    n001 = ((nasa_pred == 0) & (rf_pred == 0) & (sim_pred == 1)).sum()  # Only SIM
    n010 = ((nasa_pred == 0) & (rf_pred == 1) & (sim_pred == 0)).sum()  # Only RF
    n011 = ((nasa_pred == 0) & (rf_pred == 1) & (sim_pred == 1)).sum()  # RF+SIM
    n100 = ((nasa_pred == 1) & (rf_pred == 0) & (sim_pred == 0)).sum()  # Only NASA
    n101 = ((nasa_pred == 1) & (rf_pred == 0) & (sim_pred == 1)).sum()  # NASA+SIM
    n110 = ((nasa_pred == 1) & (rf_pred == 1) & (sim_pred == 0)).sum()  # NASA+RF
    n111 = ((nasa_pred == 1) & (rf_pred == 1) & (sim_pred == 1)).sum()  # All accept
    
    # Create text summary
    text = "All 8 Possible Combinations:\n\n"
    text += f"{'Combination':<25} {'Count':>10} {'%':>7}\n"
    text += "="*45 + "\n"
    
    combinations = [
        ("All 3 Accept", n111),
        ("NASA + SIM only", n101),
        ("NASA + RF only", n110),
        ("RF + SIM only", n011),
        ("NASA only", n100),
        ("RF only", n010),
        ("SIM only", n001),
        ("All 3 Reject", n000),
    ]
    
    for name, count in combinations:
        pct = 100 * count / stats['total']
        text += f"{name:<25} {count:>10,} {pct:>6.1f}%\n"
    
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
           fontsize=12, verticalalignment='center', horizontalalignment='center',
           family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_title('Complete Breakdown of All Combinations', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pairwise_agreement_analysis.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: pairwise_agreement_analysis.png")
    plt.close()


def print_agreement_summary(stats: dict):
    """Print detailed agreement statistics."""
    print("\n" + "="*70)
    print("PAIRWISE AGREEMENT ANALYSIS")
    print("="*70)
    
    print(f"\nTotal sites evaluated: {stats['total']:,}")
    
    print("\n" + "-"*70)
    print("NASA vs Random Forest:")
    print("-"*70)
    print(f"  Overall agreement:     {stats['nasa_rf']['agree']:>8,} ({stats['nasa_rf']['agree_pct']:>5.1f}%)")
    print(f"    Both accept:         {stats['nasa_rf']['both_accept']:>8,}")
    print(f"    Both reject:         {stats['nasa_rf']['both_reject']:>8,}")
    print(f"  Disagreement:          {stats['total'] - stats['nasa_rf']['agree']:>8,} ({100 - stats['nasa_rf']['agree_pct']:>5.1f}%)")
    print(f"    NASA only accepts:   {stats['nasa_rf']['nasa_only']:>8,}")
    print(f"    RF only accepts:     {stats['nasa_rf']['rf_only']:>8,}")
    
    print("\n" + "-"*70)
    print("NASA vs Similarity:")
    print("-"*70)
    print(f"  Overall agreement:     {stats['nasa_sim']['agree']:>8,} ({stats['nasa_sim']['agree_pct']:>5.1f}%)")
    print(f"    Both accept:         {stats['nasa_sim']['both_accept']:>8,}")
    print(f"    Both reject:         {stats['nasa_sim']['both_reject']:>8,}")
    print(f"  Disagreement:          {stats['total'] - stats['nasa_sim']['agree']:>8,} ({100 - stats['nasa_sim']['agree_pct']:>5.1f}%)")
    print(f"    NASA only accepts:   {stats['nasa_sim']['nasa_only']:>8,}")
    print(f"    SIM only accepts:    {stats['nasa_sim']['sim_only']:>8,}")
    
    print("\n" + "-"*70)
    print("Random Forest vs Similarity:")
    print("-"*70)
    print(f"  Overall agreement:     {stats['rf_sim']['agree']:>8,} ({stats['rf_sim']['agree_pct']:>5.1f}%)")
    print(f"    Both accept:         {stats['rf_sim']['both_accept']:>8,}")
    print(f"    Both reject:         {stats['rf_sim']['both_reject']:>8,}")
    print(f"  Disagreement:          {stats['total'] - stats['rf_sim']['agree']:>8,} ({100 - stats['rf_sim']['agree_pct']:>5.1f}%)")
    print(f"    RF only accepts:     {stats['rf_sim']['rf_only']:>8,}")
    print(f"    SIM only accepts:    {stats['rf_sim']['sim_only']:>8,}")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    # Find best agreeing pair
    pairs = [
        ('NASA vs SIM', stats['nasa_sim']['agree_pct']),
        ('NASA vs RF', stats['nasa_rf']['agree_pct']),
        ('RF vs SIM', stats['rf_sim']['agree_pct'])
    ]
    best_pair = max(pairs, key=lambda x: x[1])
    worst_pair = min(pairs, key=lambda x: x[1])
    
    print(f"\n✓ Highest agreement:  {best_pair[0]} ({best_pair[1]:.1f}%)")
    print(f"✗ Lowest agreement:   {worst_pair[0]} ({worst_pair[1]:.1f}%)")
    
    # Check if RF ever accepts when others reject
    if stats['rf_sim']['rf_only'] == 0:
        print(f"\n⚠️  RF NEVER accepts a site that Similarity rejects")
        print(f"    (RF is strictly more conservative than SIM)")
    
    if stats['nasa_rf']['rf_only'] == 0:
        print(f"\n⚠️  RF NEVER accepts a site that NASA rejects")
        print(f"    (RF is strictly more conservative than NASA)")


def main():
    """Generate pairwise agreement analysis."""
    print("="*70)
    print("PAIRWISE MODEL AGREEMENT ANALYSIS")
    print("="*70)
    
    # Load results
    results_dir = Path(__file__).parent / 'output' / 'results'
    plots_dir = Path(__file__).parent / 'output' / 'plots'
    
    print("\n📂 Loading evaluation results...")
    results_path = results_dir / 'candidate_evaluations.csv'
    result_df = pd.read_csv(results_path)
    print(f"  ✓ Loaded {len(result_df):,} evaluated sites")
    
    # Analyze agreement
    print("\n📊 Analyzing pairwise agreements...")
    stats = analyze_pairwise_agreement(result_df)
    
    # Print summary
    print_agreement_summary(stats)
    
    # Generate visualization
    print("\n🎨 Generating visualization...")
    plot_pairwise_agreement(stats, result_df, plots_dir)
    
    print(f"\n✓ Plot saved to: {plots_dir}")
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
