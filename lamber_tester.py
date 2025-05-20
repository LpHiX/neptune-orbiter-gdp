import numpy as np
from lambert_solver import lambert_solver

# Example: Earth to Jupiter transfer

# Sun's gravitational parameter (m³/s²)
mu_sun = 1.32712440018e20

# Earth's position (example in meters)
r1 = [1.49598e11, 0, 0]  # Earth at 1 AU

# Jupiter's position (example in meters)
r2 = [0, 7.78e11, 0]  # Jupiter at ~5.2 AU but rotated in position

# Time of flight (seconds) - 2.5 years
tof = 0.01 * 365.25 * 24 * 3600

# Solve Lambert's problem
v1, v2, a, success = lambert_solver(r1, r2, tof, mu=mu_sun)

print(f"Initial velocity: [{v1[0]:.2f}, {v1[1]:.2f}, {v1[2]:.2f}] m/s")
print(f"Final velocity: [{v2[0]:.2f}, {v2[1]:.2f}, {v2[2]:.2f}] m/s")
print(f"Semi-major axis: {a/1.496e11:.2f} AU ({a:.3e} m)")
print(f"Transfer type: {'Elliptical' if a > 0 else 'Hyperbolic'}")