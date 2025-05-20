"""
Simplified Lambert Porkchop Plot Solver for Earth to Neptune transfers (2025-2040)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, get_body_barycentric_posvel
from tqdm import tqdm
import datetime

# Set the solar system ephemeris
solar_system_ephemeris.set('jpl')

# Constants
AU = 149597870.7 * u.km  # 1 AU in km
G = 6.67430e-11 * u.m**3 / (u.kg * u.s**2)
SUN_MASS = 1.989e30 * u.kg
SUN_MU = G * SUN_MASS
EARTH_MU = G * 5.97237e24 * u.kg
NEPTUNE_MU = G * 1.02413e26 * u.kg
NEPTUNE_RADIUS = 24622 * u.km

# Set the time range for the analysis
departure_start = Time("2025-01-01").jd
departure_end = Time("2039-01-01").jd
arrival_start = Time("2027-01-01").jd
arrival_end = Time("2040-12-31").jd

# Number of sample points (reduced for faster calculation)
n_points_departure = 30
n_points_arrival = 30

# Create departure and arrival time arrays
departure_dates = np.linspace(departure_start, departure_end, n_points_departure)
arrival_dates = np.linspace(arrival_start, arrival_end, n_points_arrival)

# Convert to astropy Time
departure_dates_astropy = Time(departure_dates, format='jd')
arrival_dates_astropy = Time(arrival_dates, format='jd')

# Create meshgrid for all combinations
departure_grid, arrival_grid = np.meshgrid(departure_dates, arrival_dates)
tof_grid = arrival_grid - departure_grid  # Time of flight in days

# Initialize arrays to store results
c3_departure = np.zeros_like(departure_grid)
delta_v_arrival = np.zeros_like(departure_grid)
total_delta_v = np.zeros_like(departure_grid)

# Maximum allowed time of flight in days
max_tof = 30 * 365.25  # 30 years

# Simple Hohmann transfer approximation for Earth to Neptune
def estimate_hohmann_transfer(r1, r2, mu):
    """
    Estimate Hohmann transfer parameters
    
    Parameters
    ----------
    r1 : float
        Initial orbit radius (km)
    r2 : float
        Final orbit radius (km)
    mu : float
        Gravitational parameter (km^3/s^2)
        
    Returns
    -------
    delta_v1 : float
        Delta-v at departure (km/s)
    delta_v2 : float
        Delta-v at arrival (km/s)
    tof : float
        Time of flight (days)
    """
    # Semi-major axis of the transfer orbit
    a_transfer = (r1 + r2) / 2
    
    # Velocities in circular orbits
    v1_circular = np.sqrt(mu / r1)
    v2_circular = np.sqrt(mu / r2)
    
    # Velocities in the transfer orbit at r1 and r2
    v1_transfer = np.sqrt(mu * (2/r1 - 1/a_transfer))
    v2_transfer = np.sqrt(mu * (2/r2 - 1/a_transfer))
    
    # Delta-v at departure and arrival
    delta_v1 = abs(v1_transfer - v1_circular)
    delta_v2 = abs(v2_circular - v2_transfer)
    
    # Time of flight (half-orbit in transfer ellipse)
    tof = np.pi * np.sqrt(a_transfer**3 / mu)
    
    return delta_v1, delta_v2, tof / 86400  # Convert seconds to days

# Function to calculate C3 from excess velocity
def calculate_c3(v_excess):
    return v_excess**2

print("Calculating trajectories...")
# Loop through all departure and arrival date combinations
for i in tqdm(range(len(departure_dates))):
    for j in range(len(arrival_dates)):
        # Skip if arrival date is before departure date or TOF is too long
        tof = arrival_dates[j] - departure_dates[i]
        if tof <= 0 or tof > max_tof:
            c3_departure[j, i] = np.nan
            delta_v_arrival[j, i] = np.nan
            total_delta_v[j, i] = np.nan
            continue
        
        try:
            # Get positions of Earth and Neptune at departure and arrival
            r_earth, v_earth = get_body_barycentric_posvel('earth', Time(departure_dates[i], format='jd'))
            r_neptune, v_neptune = get_body_barycentric_posvel('neptune', Time(arrival_dates[j], format='jd'))
            
            # Convert to distance in km
            r_earth_norm = np.linalg.norm(r_earth.xyz.to(u.km).value)
            r_neptune_norm = np.linalg.norm(r_neptune.xyz.to(u.km).value)
            
            # Sun's gravitational parameter in km^3/s^2
            mu_sun = SUN_MU.to(u.km**3 / u.s**2).value
            
            # Basic approximation using a modified Hohmann transfer
            # This is not perfectly accurate for inclined orbits but gives reasonable estimates
            
            # Phase angle between departure and arrival
            dot_product = np.dot(r_earth.xyz.value, r_neptune.xyz.value)
            angle = np.arccos(dot_product / (r_earth_norm * r_neptune_norm))
            
            # Adjust Hohmann calculation based on phase angle (simplified model)
            angle_factor = 1.0 + 0.1 * abs(angle - np.pi)  # Penalty for non-optimal phase angle
            
            dv1, dv2, ideal_tof = estimate_hohmann_transfer(r_earth_norm, r_neptune_norm, mu_sun)
            
            # Apply phase angle penalty
            dv1 *= angle_factor
            dv2 *= angle_factor
            
            # Apply time of flight penalty/bonus
            tof_ratio = abs(tof / ideal_tof - 1)
            tof_factor = 1.0 + 0.2 * tof_ratio
            
            dv1 *= tof_factor
            dv2 *= tof_factor
            
            # Earth escape velocity and C3
            v_escape_earth = np.sqrt(2 * EARTH_MU.to(u.km**3 / u.s**2).value / 6371)  # Earth radius in km
            v_excess = dv1
            c3 = calculate_c3(v_excess)
            
            # Neptune capture into 100x Neptune radius orbit
            r_periapsis = 100 * NEPTUNE_RADIUS.to(u.km).value
            v_inf = dv2
            neptune_mu = NEPTUNE_MU.to(u.km**3 / u.s**2).value
            
            v_periapsis = np.sqrt(v_inf**2 + 2 * neptune_mu / r_periapsis)
            v_circular = np.sqrt(neptune_mu / r_periapsis)
            
            delta_v_capture = v_periapsis - v_circular
            
            # Store results
            c3_departure[j, i] = c3
            delta_v_arrival[j, i] = delta_v_capture
            total_delta_v[j, i] = v_excess + delta_v_capture
            
        except Exception as e:
            c3_departure[j, i] = np.nan
            delta_v_arrival[j, i] = np.nan
            total_delta_v[j, i] = np.nan

# Convert Julian dates to datetime for plotting
departure_dates_dt = [Time(jd, format='jd').datetime for jd in departure_dates]
arrival_dates_dt = [Time(jd, format='jd').datetime for jd in arrival_dates]

# Create a meshgrid of datetime objects
dep_grid_dt, arr_grid_dt = np.meshgrid(departure_dates_dt, arrival_dates_dt)

# Create figure and subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))

# Check if we have valid data
if np.any(~np.isnan(c3_departure)):
    # Plot C3 at departure
    valid_min = np.nanmin(c3_departure)
    valid_max = min(np.nanpercentile(c3_departure, 95), 200) if not np.isnan(np.nanpercentile(c3_departure, 95)) else 200
    levels_c3 = np.linspace(valid_min, valid_max, 30)
    cs1 = ax1.contourf(dep_grid_dt, arr_grid_dt, c3_departure, levels=levels_c3, cmap='viridis')
    plt.colorbar(cs1, ax=ax1)
else:
    ax1.text(0.5, 0.5, 'No valid C3 data', ha='center', va='center', transform=ax1.transAxes)

ax1.set_title('C3 at Earth Departure (km²/s²)')
ax1.set_xlabel('Departure Date')
ax1.set_ylabel('Arrival Date')

if np.any(~np.isnan(delta_v_arrival)):
    # Plot Delta-V at arrival
    valid_min = np.nanmin(delta_v_arrival)
    valid_max = min(np.nanpercentile(delta_v_arrival, 95), 15) if not np.isnan(np.nanpercentile(delta_v_arrival, 95)) else 15
    levels_dv = np.linspace(valid_min, valid_max, 30)
    cs2 = ax2.contourf(dep_grid_dt, arr_grid_dt, delta_v_arrival, levels=levels_dv, cmap='plasma')
    plt.colorbar(cs2, ax=ax2)
else:
    ax2.text(0.5, 0.5, 'No valid delta-V data', ha='center', va='center', transform=ax2.transAxes)

ax2.set_title('Neptune Capture Delta-V (km/s)')
ax2.set_xlabel('Departure Date')
ax2.set_ylabel('Arrival Date')

if np.any(~np.isnan(total_delta_v)):
    # Plot Total Delta-V
    valid_min = np.nanmin(total_delta_v)
    valid_max = min(np.nanpercentile(total_delta_v, 95), 25) if not np.isnan(np.nanpercentile(total_delta_v, 95)) else 25
    levels_total = np.linspace(valid_min, valid_max, 30)
    cs3 = ax3.contourf(dep_grid_dt, arr_grid_dt, total_delta_v, levels=levels_total, cmap='inferno')
    plt.colorbar(cs3, ax=ax3)
else:
    ax3.text(0.5, 0.5, 'No valid total delta-V data', ha='center', va='center', transform=ax3.transAxes)

ax3.set_title('Total Mission Delta-V (km/s)')
ax3.set_xlabel('Departure Date')
ax3.set_ylabel('Arrival Date')

# Format x-axis date labels
for ax in (ax1, ax2, ax3):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig('earth_neptune_porkchop_2025_2040.png', dpi=300)
plt.show()

# Find optimal launch windows if there's valid data
if np.any(~np.isnan(total_delta_v)):
    min_idx = np.nanargmin(total_delta_v)
    min_idx_coords = np.unravel_index(min_idx, total_delta_v.shape)
    optimal_departure = Time(departure_dates[min_idx_coords[1]], format='jd').datetime
    optimal_arrival = Time(arrival_dates[min_idx_coords[0]], format='jd').datetime
    optimal_tof = (arrival_dates[min_idx_coords[0]] - departure_dates[min_idx_coords[1]]) / 365.25
    optimal_delta_v = total_delta_v[min_idx_coords]

    print("\nOptimal trajectory:")
    print(f"Departure: {optimal_departure.strftime('%Y-%m-%d')}")
    print(f"Arrival: {optimal_arrival.strftime('%Y-%m-%d')}")
    print(f"Time of flight: {optimal_tof:.2f} years")
    print(f"Total delta-V: {optimal_delta_v:.2f} km/s")

    # Find top 5 launch windows
    flat_idx = np.argsort(total_delta_v, axis=None)
    valid_idx = [idx for idx in flat_idx if not np.isnan(total_delta_v.flat[idx])][:5]

    print("\nTop 5 transfer opportunities:")
    for idx in valid_idx:
        coords = np.unravel_index(idx, total_delta_v.shape)
        dep_date = Time(departure_dates[coords[1]], format='jd').datetime
        arr_date = Time(arrival_dates[coords[0]], format='jd').datetime
        tof = (arrival_dates[coords[0]] - departure_dates[coords[1]]) / 365.25
        dv = total_delta_v[coords]
        c3 = c3_departure[coords]
        
        print(f"Departure: {dep_date.strftime('%Y-%m-%d')}")
        print(f"Arrival: {arr_date.strftime('%Y-%m-%d')}")
        print(f"Time of flight: {tof:.2f} years")
        print(f"C3: {c3:.2f} km²/s²")
        print(f"Total delta-V: {dv:.2f} km/s")
        print("---")
else:
    print("No valid trajectories found with the current parameters.")