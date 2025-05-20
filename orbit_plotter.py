from skyfield.api import load
from skyfield.framelib import ecliptic_frame
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import timedelta
import numpy as np

# Load planetary data and timescale
planets = load('de430t.bsp')
ts = load.timescale()
start_time = ts.utc(2025, 5, 13)
duration = 365*200

sun = planets['sun']
earth = planets['earth barycenter']
neptune = planets['neptune barycenter']
venus = planets['venus barycenter']

# Prepare arrays to store positions
earth_positions = np.array([0, 0, 0])
neptune_positions = np.array([0, 0, 0])
venus_positions = np.array([0, 0, 0])

# Loop through each day to calculate positions in the ecliptic frame
for time in np.linspace(0, duration, num=100):
    current_time = start_time + timedelta(days=time)
    earth_pos = sun.at(start_time + timedelta(days=time/200)).observe(earth).frame_xyz(ecliptic_frame).au
    neptune_pos = sun.at(current_time).observe(neptune).frame_xyz(ecliptic_frame).au
    venus_pos = sun.at(start_time + timedelta(days=time/200)).observe(venus).frame_xyz(ecliptic_frame).au
    earth_positions = np.vstack((earth_positions, earth_pos))
    neptune_positions = np.vstack((neptune_positions, neptune_pos))
    venus_positions = np.vstack((venus_positions, venus_pos))

# Remove the initial placeholder row
earth_positions = earth_positions[1:]
neptune_positions = neptune_positions[1:]
venus_positions = venus_positions[1:]

# Plot the orbits
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot(earth_positions[:, 0], earth_positions[:, 1], earth_positions[:, 2], label='Earth', color='blue')
ax.plot(neptune_positions[:, 0], neptune_positions[:, 1], neptune_positions[:, 2], label='Neptune', color='red')
ax.plot(venus_positions[:, 0], venus_positions[:, 1], venus_positions[:, 2], label='Venus', color='green')

# Make the plot axes equal/square
all_positions = np.vstack([earth_positions, neptune_positions, venus_positions])
max_range = np.max(all_positions.max(axis=0) - all_positions.min(axis=0))
mid_x = (all_positions[:, 0].min() + all_positions[:, 0].max()) / 2
mid_y = (all_positions[:, 1].min() + all_positions[:, 1].max()) / 2
mid_z = (all_positions[:, 2].min() + all_positions[:, 2].max()) / 2

ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

# Alternative approach for newer matplotlib versions
try:
    ax.set_box_aspect([1, 1, 1])
except AttributeError:
    pass  # Older matplotlib version, already handled with set_xlim etc.

ax.set_xlabel('X (AU)')
ax.set_ylabel('Y (AU)')
ax.set_zlabel('Z (AU)')
ax.legend()

# Add Sun at center
# ax.scatter([0], [0], [0], color='yellow', s=100, label='Sun')

plt.tight_layout()
plt.show()