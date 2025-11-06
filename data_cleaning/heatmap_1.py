#!/usr/bin/env python3
"""
Generate a rectangular random heatmap with a clear elliptical cutout.

Outputs:
 - Displays the figure
 - Saves an image `heatmap_ellipse_cutout.png` in the current directory

Usage:
    python heatmap_1.py

This script is standalone and uses numpy/matplotlib.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def make_heatmap(width=600, height=400, seed=42, n_hotspots=8):
    """Generate heatmap with clear Gaussian hotspots."""
    rng = np.random.default_rng(seed)
    
    # Start with low-level background noise
    data = rng.standard_normal((height, width)) * 0.3
    
    # Add Gaussian hotspots
    Y, X = np.mgrid[0:height, 0:width]
    
    for _ in range(n_hotspots):
        # Random center for each hotspot
        cx = rng.uniform(width * 0.1, width * 0.9)
        cy = rng.uniform(height * 0.1, height * 0.9)
        
        # Random intensity and size
        intensity = rng.uniform(3, 8)
        sigma_x = rng.uniform(width * 0.08, width * 0.15)
        sigma_y = rng.uniform(height * 0.08, height * 0.15)
        
        # Add Gaussian peak
        gaussian = intensity * np.exp(-((X - cx)**2 / (2 * sigma_x**2) + 
                                        (Y - cy)**2 / (2 * sigma_y**2)))
        data += gaussian
    
    return data


def ellipse_mask(shape, center, axes, angle=0.0):
    """Return boolean mask with True inside the ellipse.

    shape: (rows, cols)
    center: (cy, cx) in pixel coordinates
    axes: (a, b) semi-major/semi-minor in pixels (along x,y before rotation)
    angle: rotation in degrees (counter-clockwise)
    """
    rows, cols = shape
    Y, X = np.mgrid[0:rows, 0:cols]
    cy, cx = center
    a, b = axes
    theta = np.deg2rad(angle)
    # Translate
    xt = X - cx
    yt = Y - cy
    # Rotate coordinates by -theta to align ellipse axes
    xr = xt * np.cos(-theta) - yt * np.sin(-theta)
    yr = xt * np.sin(-theta) + yt * np.cos(-theta)
    mask = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0
    return mask


def plot_heatmap_with_cutout(data, mask, out_file="heatmap_ellipse_cutout.png"):
    # Show the full heatmap gradient across the entire rectangle
    # Only draw the ellipse boundary (no masking/cutout)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap('viridis')

    im = ax.imshow(data, origin='lower', cmap=cmap, interpolation='nearest')
    ax.set_title('Random heatmap with ellipse boundary')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')

    # Draw ellipse outline where mask is True
    # compute properties for outline
    ys, xs = np.where(mask)
    if len(xs) > 0:
        cx = xs.mean()
        cy = ys.mean()
        a = (xs.max() - xs.min()) / 2.0
        b = (ys.max() - ys.min()) / 2.0
        ell = Ellipse((cx, cy), width=2*a, height=2*b, edgecolor='red', facecolor='none', lw=3)
        ax.add_patch(ell)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Intensity')

    plt.tight_layout()
    fig.savefig(out_file, dpi=200)
    print(f"Saved heatmap with ellipse boundary to: {out_file}")
    plt.show()


def main():
    data = make_heatmap()
    rows, cols = data.shape
    # center the ellipse in the middle, size as fraction of the array
    center = (rows // 2, cols // 2)
    axes = (cols * 0.25, rows * 0.25)  # semi-major along x, semi-minor along y
    angle = -20.0  # degrees rotation

    mask = ellipse_mask(data.shape, center=center, axes=axes, angle=angle)
    # We want the cutout inside ellipse -> mask True means hidden
    plot_heatmap_with_cutout(data, mask)


if __name__ == '__main__':
    main()
