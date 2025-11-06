"""shared_utils.py

Common utility functions for topographic analysis including coordinate conversions
and local plane fitting for slope and roughness calculations.
"""

import numpy as np
from numpy.linalg import lstsq
from math import radians, cos, atan, degrees, pi


def meters_per_degree(lat_deg, radius_m):
    """
    Calculate meters per degree of latitude and longitude at a given location.
    
    Args:
        lat_deg: Latitude in degrees
        radius_m: Local planetary radius in meters
    
    Returns:
        tuple: (meters_per_deg_lon, meters_per_deg_lat)
    """
    m_per_deg_lat = pi * radius_m / 180.0
    m_per_deg_lon = m_per_deg_lat * cos(radians(lat_deg))
    return m_per_deg_lon, m_per_deg_lat


def lonlat_to_local_xy(lon_deg, lat_deg, lon0, lat0, radius_m):
    """
    Convert lon/lat coordinates to local x,y in meters using equirectangular projection.
    
    This is a simple approximation suitable for small regions. The projection is
    centered at (lon0, lat0) and uses the local radius for accurate distance calculation.
    
    Args:
        lon_deg: Longitude(s) in degrees (can be array)
        lat_deg: Latitude(s) in degrees (can be array)
        lon0: Center longitude in degrees
        lat0: Center latitude in degrees
        radius_m: Local planetary radius in meters at the center point
    
    Returns:
        tuple: (x, y) in meters, where x is east and y is north
    """
    m_per_deg_lon0, m_per_deg_lat0 = meters_per_degree(lat0, radius_m)
    x = (lon_deg - lon0) * m_per_deg_lon0
    y = (lat_deg - lat0) * m_per_deg_lat0
    return x, y


def fit_plane(x, y, z):
    """
    Fit a plane z = ax + by + c to 3D points using least squares.
    
    Args:
        x: Array of x coordinates (meters)
        y: Array of y coordinates (meters)
        z: Array of z elevations (meters)
    
    Returns:
        tuple: (a, b, c) - plane coefficients
        
    Raises:
        Exception if fitting fails (e.g., insufficient points or collinear data)
    """
    X = np.column_stack([x, y, np.ones_like(x)])
    coeffs, residuals, rank, s = lstsq(X, z, rcond=None)
    return coeffs[0], coeffs[1], coeffs[2]


def plane_slope_from_coeffs(a, b):
    """
    Calculate slope angle in degrees from plane coefficients.
    
    For plane z = ax + by + c, the gradient magnitude is sqrt(a² + b²),
    and the slope angle is arctan(gradient_magnitude).
    
    Args:
        a: Coefficient of x (dz/dx)
        b: Coefficient of y (dz/dy)
    
    Returns:
        float: Slope angle in degrees
    """
    gradient_magnitude = np.sqrt(a*a + b*b)
    slope_rad = atan(gradient_magnitude)
    return degrees(slope_rad)


def plane_residuals(x, y, z, a, b, c):
    """
    Calculate residuals (actual z - predicted z) for a fitted plane.
    
    Args:
        x: Array of x coordinates (meters)
        y: Array of y coordinates (meters)
        z: Array of actual z elevations (meters)
        a, b, c: Plane coefficients (z = ax + by + c)
    
    Returns:
        Array of residuals (same length as x, y, z)
    """
    z_predicted = a*x + b*y + c
    return z - z_predicted


def rms_roughness(residuals):
    """
    Calculate RMS (root mean square) roughness from plane-fit residuals.
    
    This represents the typical vertical deviation from the best-fit plane
    and is a standard measure of surface roughness.
    
    Args:
        residuals: Array of vertical deviations from fitted plane (meters)
    
    Returns:
        float: RMS roughness in meters
    """
    return np.sqrt(np.mean(residuals**2))


def plane_fit_slope_rough(x, y, z):
    """
    Combined function: fit plane and return both slope and roughness.
    
    This is a convenience wrapper that combines fit_plane, plane_slope_from_coeffs,
    plane_residuals, and rms_roughness into a single call.
    
    Args:
        x: Array of x coordinates (meters)
        y: Array of y coordinates (meters)
        z: Array of z elevations (meters)
    
    Returns:
        tuple: (slope_deg, roughness_rms_m)
    """
    a, b, c = fit_plane(x, y, z)
    slope_deg = plane_slope_from_coeffs(a, b)
    resid = plane_residuals(x, y, z, a, b, c)
    rough_rms_m = rms_roughness(resid)
    return slope_deg, rough_rms_m
