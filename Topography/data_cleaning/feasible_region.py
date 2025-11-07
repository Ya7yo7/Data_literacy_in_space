import numpy as np
import matplotlib.pyplot as plt

# --- 1. Constants ---
R_MARS = 3396190.0  # mean Mars radius (m)
DEG2RAD = np.pi / 180

# --- 2. Landing site coordinates (deg) ---
# InSight (Elysium Planitia)
lat1, lon1 = 4.502, 135.623
# Curiosity (Gale Crater)
lat2, lon2 = -4.5895, 137.4417

# --- 3. Convert to radians ---
phi1, lam1 = np.radians(lat1), np.radians(lon1)
phi2, lam2 = np.radians(lat2), np.radians(lon2)

# --- 4. Define global grid (around both sites) ---
lat_vec = np.linspace(-10, 10, 600)    # adjust bounds for study region
lon_vec = np.linspace(130, 142, 600)
LAT, LON = np.meshgrid(lat_vec, lon_vec, indexing="ij")

lat = np.radians(LAT)
lon = np.radians(LON)

# --- 5. Haversine distance function ---
def haversine(phi1, lam1, phi2, lam2, R=R_MARS):
    """Great-circle distance in meters."""
    dphi = phi2 - phi1
    dlam = lam2 - lam1
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.minimum(1, np.sqrt(a)))

# --- 6. Distances from each focus ---
d1 = haversine(phi1, lam1, lat, lon)
d2 = haversine(phi2, lam2, lat, lon)

# --- 7. Compute ellipse parameters ---
c = haversine(phi1, lam1, phi2, lam2) / 2.0   # half the foci separation
a = c + 60_000.0                              # control ellipse size (+60 km buffer)
ellipse_mask = (d1 + d2) <= (2 * a)

# --- 7b. Find extreme points of the ellipse ---
# Find indices where ellipse_mask is True
ellipse_indices = np.where(ellipse_mask)
if len(ellipse_indices[0]) > 0:
    ellipse_lats = LAT[ellipse_indices]
    ellipse_lons = LON[ellipse_indices]
    
    # Find extreme coordinates
    lat_min = ellipse_lats.min()
    lat_max = ellipse_lats.max()
    lon_min = ellipse_lons.min()
    lon_max = ellipse_lons.max()
else:
    lat_min, lat_max = LAT.min(), LAT.max()
    lon_min, lon_max = LON.min(), LON.max()

# --- 8. Visualization ---
plt.figure(figsize=(10,8))
plt.contourf(LON, LAT, ellipse_mask, levels=[0,0.5,1], colors=["#d0e0ff","#004080"], alpha=0.7)

# Plot landing sites
plt.scatter([lon1], [lat1], c='red', s=100, marker='o', edgecolors='black', linewidths=1.5, zorder=5)
plt.scatter([lon2], [lat2], c='red', s=100, marker='o', edgecolors='black', linewidths=1.5, zorder=5)

# Label landing sites
plt.text(lon1, lat1 + 0.3, 'InSight', fontsize=10, fontweight='bold', 
         ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
plt.text(lon2, lat2 - 0.3, 'Curiosity', fontsize=10, fontweight='bold', 
         ha='center', va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Draw constraint lines (parallels) for extreme points
plt.axhline(y=lat_min, color='green', linestyle='--', linewidth=1.5, alpha=0.8, label=f'Bottom: {lat_min:.2f}°N')
plt.axhline(y=lat_max, color='green', linestyle='--', linewidth=1.5, alpha=0.8, label=f'Top: {lat_max:.2f}°N')
plt.axvline(x=lon_min, color='orange', linestyle='--', linewidth=1.5, alpha=0.8, label=f'Left: {lon_min:.2f}°E')
plt.axvline(x=lon_max, color='orange', linestyle='--', linewidth=1.5, alpha=0.8, label=f'Right: {lon_max:.2f}°E')

plt.title("Feasible Elliptical Region between InSight and Curiosity\nwith Coordinate Constraints", fontsize=12, fontweight='bold')
plt.xlabel("Longitude (°E)", fontsize=11)
plt.ylabel("Latitude (°N)", fontsize=11)
plt.legend(loc='upper right', fontsize=9)
plt.colorbar(label="Inside Ellipse (1=True)")
plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
plt.tight_layout()
plt.show()

# --- 9. Print constraint information ---
print("\n" + "="*60)
print("ELLIPSE COORDINATE CONSTRAINTS")
print("="*60)
print(f"Latitude range:  {lat_min:.4f}°N  to  {lat_max:.4f}°N")
print(f"Longitude range: {lon_min:.4f}°E  to  {lon_max:.4f}°E")
print(f"\nTo filter data within ellipse, use:")
print(f"  lat_min = {lat_min:.4f}")
print(f"  lat_max = {lat_max:.4f}")
print(f"  lon_min = {lon_min:.4f}")
print(f"  lon_max = {lon_max:.4f}")
print("="*60)

# --- 9. Optional: export mask to file ---
# np.save("mars_feasible_mask.npy", ellipse_mask)
