"""top_sites.py

Extract and display top-ranked candidate landing sites.
"""
import pandas as pd
from pathlib import Path


def main():
    """Extract top candidate sites by various criteria."""
    print("="*70)
    print("TOP CANDIDATE LANDING SITES")
    print("="*70)
    
    # Load results
    results_dir = Path(__file__).parent / 'output' / 'results'
    results_path = results_dir / 'candidate_evaluations.csv'
    
    print(f"\n📂 Loading results from: {results_path}")
    df = pd.read_csv(results_path)
    print(f"  ✓ Loaded {len(df):,} evaluated sites\n")
    
    # Top by consensus count
    print("="*70)
    print("🏆 TOP 20 SITES BY CONSENSUS COUNT (2/3 models agree)")
    print("="*70)
    
    majority = df[df['consensus_count'] == 2].copy()
    print(f"\nTotal sites with 2/3 consensus: {len(majority):,}\n")
    
    # Sort by average score
    majority['avg_score'] = (majority['nasa_score'] + majority['rf_score'] + majority['sim_score']) / 3
    top_consensus = majority.nlargest(20, 'avg_score')
    
    print(f"{'Rank':<6} {'Lon (°E)':<10} {'Lat (°N)':<10} {'Alt (m)':<10} "
          f"{'Slope (°)':<10} {'Rough (m)':<10} {'NASA':<8} {'RF':<8} {'SIM':<8} {'Avg':<8}")
    print("-"*100)
    
    for i, (idx, row) in enumerate(top_consensus.iterrows(), 1):
        print(f"{i:<6} {row['long_east_deg']:<10.3f} {row['lat_north_deg']:<10.3f} "
              f"{row['altitude_m']:<10.0f} {row['slope_deg']:<10.2f} {row['roughness_rms_m']:<10.2f} "
              f"{row['nasa_score']:<8.3f} {row['rf_score']:<8.3f} {row['sim_score']:<8.3f} "
              f"{row['avg_score']:<8.3f}")
    
    # Sites accepted by RF (extremely rare)
    print("\n" + "="*70)
    print("⭐ ALL SITES ACCEPTED BY RANDOM FOREST (0.3% of total)")
    print("="*70)
    
    rf_accepted = df[df['rf_pred'] == 1].copy()
    print(f"\nTotal RF-accepted sites: {len(rf_accepted):,}\n")
    
    # Sort by RF score
    top_rf = rf_accepted.nlargest(20, 'rf_score')
    
    print(f"{'Rank':<6} {'Lon (°E)':<10} {'Lat (°N)':<10} {'Alt (m)':<10} "
          f"{'Slope (°)':<10} {'Rough (m)':<10} {'NASA':<8} {'RF':<8} {'SIM':<8}")
    print("-"*100)
    
    for i, (idx, row) in enumerate(top_rf.iterrows(), 1):
        print(f"{i:<6} {row['long_east_deg']:<10.3f} {row['lat_north_deg']:<10.3f} "
              f"{row['altitude_m']:<10.0f} {row['slope_deg']:<10.2f} {row['roughness_rms_m']:<10.2f} "
              f"{row['nasa_score']:<8.3f} {row['rf_score']:<8.3f} {row['sim_score']:<8.3f}")
    
    # Top by individual model scores
    print("\n" + "="*70)
    print("🎯 TOP 10 SITES BY INDIVIDUAL MODEL SCORES")
    print("="*70)
    
    for model, name in [('nasa', 'NASA Constraints'), ('rf', 'Random Forest'), ('sim', 'Similarity KDE')]:
        print(f"\n{name}:")
        print(f"{'Lon (°E)':<10} {'Lat (°N)':<10} {'Alt (m)':<10} {'Slope (°)':<10} "
              f"{'Rough (m)':<10} {model.upper()+' Score':<12}")
        print("-"*70)
        
        top_model = df.nlargest(10, f'{model}_score')
        for idx, row in top_model.iterrows():
            print(f"{row['long_east_deg']:<10.3f} {row['lat_north_deg']:<10.3f} "
                  f"{row['altitude_m']:<10.0f} {row['slope_deg']:<10.2f} {row['roughness_rms_m']:<10.2f} "
                  f"{row[f'{model}_score']:<12.3f}")
    
    # Geographic distribution
    print("\n" + "="*70)
    print("🌍 GEOGRAPHIC DISTRIBUTION OF MAJORITY-CONSENSUS SITES")
    print("="*70)
    
    print(f"\nLongitude range: {majority['long_east_deg'].min():.2f}° - {majority['long_east_deg'].max():.2f}°E")
    print(f"Latitude range:  {majority['lat_north_deg'].min():.2f}° - {majority['lat_north_deg'].max():.2f}°N")
    print(f"\nAltitude:  min={majority['altitude_m'].min():.0f}m, max={majority['altitude_m'].max():.0f}m, "
          f"mean={majority['altitude_m'].mean():.0f}m")
    print(f"Slope:     min={majority['slope_deg'].min():.2f}°, max={majority['slope_deg'].max():.2f}°, "
          f"mean={majority['slope_deg'].mean():.2f}°")
    print(f"Roughness: min={majority['roughness_rms_m'].min():.2f}m, max={majority['roughness_rms_m'].max():.2f}m, "
          f"mean={majority['roughness_rms_m'].mean():.2f}m")
    
    # Save top sites
    output_dir = Path(__file__).parent / 'output' / 'results'
    
    # Majority consensus sites
    majority_path = output_dir / 'top_sites_majority_consensus.csv'
    top_consensus.to_csv(majority_path, index=False)
    print(f"\n💾 Saved top 20 majority-consensus sites to: {majority_path}")
    
    # RF accepted sites (all of them, since there are only 2,240)
    rf_path = output_dir / 'all_rf_accepted_sites.csv'
    rf_accepted.to_csv(rf_path, index=False)
    print(f"💾 Saved all {len(rf_accepted):,} RF-accepted sites to: {rf_path}")
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
