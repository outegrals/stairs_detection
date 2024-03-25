import open3d as o3d
import sys

if len(sys.argv) < 2:
    print("Usage: python script.py <number>")
    sys.exit(1)

yes_stairs = sys.argv[1]
number = sys.argv[2]

# Load the point cloud from a PCD file
filename = f"Negative_PCD/{number}_False_XYZRGB.pcd"
if yes_stairs == 'true':
    filename = f"Positive_PCD/{number}_Cloud_XYZRGB.pcd"
pcd = o3d.io.read_point_cloud(filename)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])
