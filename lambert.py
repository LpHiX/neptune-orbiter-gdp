from skyfield.api import load
from skyfield.framelib import ecliptic_frame
from matplotlib import pyplot as plt
from datetime import timedelta
import numpy as np
from lambert_solver import lambert_solver
from tqdm import tqdm 

planets = load('de430t.bsp')
ts = load.timescale()

sun = planets['sun']
earth = planets['earth barycenter']
mars = planets['mars barycenter']

mu_sun = 1.32712440018e20

first_time = ts.utc(2025, 1, 1)
last_time = ts.utc(2040, 1, 1)

# Number of grid points
n = 1000

# Create a matrix to store delta_v values (initialize with NaN)
delta_v_matrix = np.full((n, n), np.nan)

# Generate date arrays for plotting
dates = [first_time + timedelta(days=d) for d in np.linspace(0, (last_time - first_time), num=n)]
date_labels = [date.utc_strftime('%Y-%m') for date in dates]

# Fill the delta_v matrix
for i in tqdm(range(n), desc="Computing transferes"):
    start_time = dates[i]
    
    for j in range(i+1, n):  # Only compute for j > i (upper triangle)
        end_time = dates[j]
        tof = (end_time - start_time) * 86400
        
        r1 = sun.at(start_time).observe(earth).frame_xyz(ecliptic_frame).m
        r2 = sun.at(end_time).observe(mars).frame_xyz(ecliptic_frame).m

        _, v1c = earth.at(start_time).frame_xyz_and_velocity(ecliptic_frame)
        v1c = v1c.m_per_s
        
        # Solve Lambert's problem
        try:
            v1, v2, a = lambert_solver(r1, r2, tof, mu=mu_sun)
            # Calculate delta_v and store it
            delta_v = np.linalg.norm(v1c - v1) / 1000  # Convert to km/s for better readability
            delta_v_matrix[i, j] = delta_v
        except ValueError as e:
            # Leave as NaN for invalid combinations
            pass

# ...existing code...

# Create the heatmap
plt.figure(figsize=(12, 10))

# Transpose the matrix to swap axes: x=start date, y=end date
delta_v_matrix_T = delta_v_matrix.T

# Create a log-scaled copy of the matrix for plotting
# We add a small value (1e-10) to avoid log(0) issues
log_delta_v = np.log10(delta_v_matrix_T + 1e-10)
log_delta_v[np.isnan(delta_v_matrix_T)] = np.nan  # Preserve NaN values

# Create the heatmap with log scale and inverted color scale (lower values = hotter colors)
im = plt.imshow(log_delta_v, origin='lower', cmap='viridis_r', interpolation='nearest')

# Custom colorbar with actual delta-V values (not log values)
cbar = plt.colorbar(im, label='Delta V (km/s)')

# Generate logarithmically spaced tick positions
min_val = np.nanmin(delta_v_matrix_T)
max_val = np.nanmax(delta_v_matrix_T)
log_ticks = np.logspace(np.log10(min_val), np.log10(max_val), 5)
log_tick_positions = np.interp(np.log10(log_ticks), 
                              [np.log10(min_val), np.log10(max_val)], 
                              [0, 1])
                              
# Set colorbar ticks to show actual delta-V values
cbar.set_ticks(np.interp(log_tick_positions, [0, 1], 
                         [np.nanmin(log_delta_v), np.nanmax(log_delta_v)]))
cbar.set_ticklabels([f'{val:.2f}' for val in log_ticks])

# Add labels with swapped meaning
plt.xlabel('Start Date')
plt.ylabel('End Date')

# Set tick positions and labels (show fewer ticks for readability)
tick_positions = np.arange(0, n, 4)
plt.xticks(tick_positions, [date_labels[pos] for pos in tick_positions], rotation=45)
plt.yticks(tick_positions, [date_labels[pos] for pos in tick_positions])

plt.title('Earth to Neptune Transfer Delta-V Requirements (Log Scale)')
plt.tight_layout()
plt.show()

# ...existing code...

# ...existing code...

print("Minimum Delta-V:", np.nanmin(delta_v_matrix), "km/s")
min_idx = np.unravel_index(np.nanargmin(delta_v_matrix), delta_v_matrix.shape)
print(f"Best launch: {dates[min_idx[0]].utc_strftime('%Y-%m-%d')}, arrival: {dates[min_idx[1]].utc_strftime('%Y-%m-%d')}")

