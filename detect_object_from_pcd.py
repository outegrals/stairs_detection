import open3d as o3d
import numpy as np
import sys
import matplotlib.pyplot as plt

#rotates the point cloud along the x axis
def rotate_point_cloud(pcd, angle_degrees):
    angle_radians = np.radians(angle_degrees)
    
    # Rotation matrix around X-axis
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(angle_radians), -np.sin(angle_radians)],
                    [0, np.sin(angle_radians), np.cos(angle_radians)]])
    
    # Rotate the point cloud
    pcd.rotate(R_x, center=(0, 0, 0))

# Check if the user has provided a command-line argument
if len(sys.argv) < 2:
    print("Usage: python script_name.py path_to_your_pcd_file.pcd")
    sys.exit(1)  # Exit the script if no argument is provided

# The first argument is the script name, so the second argument (index 1) is the file path
pcd_file_path = sys.argv[1]

# Load the PCD file
pcd = o3d.io.read_point_cloud(pcd_file_path)
rotate_point_cloud(pcd, 180)

plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

floor = pcd.select_by_index(inliers)
floor.paint_uniform_color([1.0, 0, 0])  # Red

objects = pcd.select_by_index(inliers, invert=True)
# Extract the point cloud data as an Nx3 numpy array
points = np.asarray(objects.points)
# Calculate the absolute distance of each point to the plane
distances_to_plane = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) 
distances_to_plane /= np.sqrt(a**2 + b**2 + c**2)
# Use the distance to filter out points that are above the plane
indices_above_plane = np.where(distances_to_plane > 0.1)[0]
objects_above_plane = pcd.select_by_index(indices_above_plane)


# Perform DBSCAN clustering
labels = np.array(objects_above_plane.cluster_dbscan(eps=0.05, min_points=10, print_progress=True))

# Create an array to hold bounding box sizes
cluster_sizes = []

# The max label is the label of the biggest cluster.
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")

# Assign a color for each cluster
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0  # noise points that are not in any cluster

objects_above_plane.colors = o3d.utility.Vector3dVector(colors[:, :3])

# Iterate over each cluster to calculate its size
for i in range(labels.max() + 1):
    # Select points belonging to the current cluster
    cluster_points = objects_above_plane.select_by_index(np.where(labels == i)[0])

    # Compute the axis-aligned bounding box of the cluster
    aabb = cluster_points.get_axis_aligned_bounding_box()
    
    # Get the extent (difference between max and min) along each axis
    extent = aabb.get_extent()
    
    # Store the size (extent) of the cluster
    cluster_sizes.append(extent)
    print(f"Cluster {i}: Size (width, depth, height) = {extent}")

# Visualize the point cloud
#o3d.visualization.draw_geometries([pcd])
o3d.visualization.draw_geometries([floor, objects_above_plane])

