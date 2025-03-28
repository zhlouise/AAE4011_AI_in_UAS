import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

file_path1 = '/home/zh_louise/AAE4011_AI_in_UAS/lecture4/Solution1.csv'  # Replace with your first CSV file path
file_path2 = '/home/zh_louise/AAE4011_AI_in_UAS/lecture4/Solution2.csv'  # Replace with your second CSV file path

data1 = pd.read_csv(file_path1)
data2 = pd.read_csv(file_path2)
time1 = data1['Time']
data1['Time'] = (data1['Time'] - time1.iloc[0])/(1e09)
pos1 = data1[['PosX', 'PosY', 'PosZ']]

time2 = data2['Time']
data2['Time'] = (data2['Time'] - time2.iloc[0])/(1e09)
# data2['Time'] = data2['Time'] - time2.iloc[0]
pos2 = data2[['PosX', 'PosY', 'PosZ']]


# Create a figure with four subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# First subplot (3D plot)
ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.plot(pos1['PosX'], pos1['PosY'], pos1['PosZ'], marker='', linestyle='-', color='r', label='Baseline')
ax.plot(pos2['PosX'], pos2['PosY'], pos2['PosZ'], marker='', linestyle='-', color='b', label='Proposed')
ax.set_title('Trajectory (X-Y-Z)')
ax.set_xlabel('X direction')
ax.set_ylabel('Y direction')
ax.set_zlabel('Z direction')
ax.legend()
ax.grid(True)

# Second subplot
axs[0, 1].plot(data1['Time'], pos1['PosX'], marker='', linestyle='-', color='r',label='Baseline')
axs[0, 1].plot(data2['Time'], pos2['PosX'], marker='', linestyle='-', color='b',label='Proposed')
axs[0, 1].set_title('X direction')
axs[0, 1].set_xlabel('time')
axs[0, 1].set_ylabel('X')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Third subplot
axs[1, 0].plot(data1['Time'], pos1['PosY'], marker='', linestyle='-', color='r',label='Baseline')
axs[1, 0].plot(data2['Time'], pos2['PosY'], marker='', linestyle='-', color='b',label='Proposed')
axs[1, 0].set_title('Y direction')
axs[1, 0].set_xlabel('time')
axs[1, 0].set_ylabel('Y')
axs[1, 0].legend()
axs[1, 0].grid(True)

# Fourth subplot
axs[1, 1].plot(data1['Time'], pos1['PosZ'], marker='', linestyle='-', color='r',label='Baseline')
axs[1, 1].plot(data2['Time'], pos2['PosZ'], marker='', linestyle='-', color='b',label='Proposed')
axs[1, 1].set_title('Z direction')
axs[1, 1].set_xlabel('time')
axs[1, 1].set_ylabel('Z')
axs[1, 1].legend()
axs[1, 1].grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plot as an image file (optional)
#plt.savefig('four_subplots.png')

# Show the plot
plt.show()