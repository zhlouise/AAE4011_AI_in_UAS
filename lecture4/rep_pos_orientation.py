import numpy as np
from scipy.spatial.transform import Rotation as R

# Position represented as a 3D vector
position = np.array([1.0, 2.0, 3.0])

# Orientation represented as Euler angles (in radians)
euler_angles = np.array([np.pi / 4, np.pi / 4, np.pi / 4])  # Roll, Pitch, Yaw

# Convert Euler angles to a rotation matrix
rotation_matrix = R.from_euler('xyz', euler_angles).as_matrix()

# Convert Euler angles to a quaternion
quaternion = R.from_euler('xyz', euler_angles).as_quat()

# Print the representations
print("Position (3D vector):", position)
print("Orientation (Euler angles):", euler_angles)
print("Orientation (Rotation matrix):\n", rotation_matrix)
print("Orientation (Quaternion):", quaternion)

# Example of using the rotation matrix to transform a point
point = np.array([1.0, 0.0, 0.0])
transformed_point = rotation_matrix @ point + position
print("Transformed point:", transformed_point)