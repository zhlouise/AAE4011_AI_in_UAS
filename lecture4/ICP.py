import open3d as o3d
import numpy as np
import copy

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0, 0])  # Red
    target_temp.paint_uniform_color([0, 1, 0])  # Green
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

# Generate source and target point clouds
def generate_point_clouds():
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    
    # Generate random points for source
    source_points = np.random.rand(10000, 3)
    source.points = o3d.utility.Vector3dVector(source_points)
    
    # Generate random points for target and apply a transformation
    target_points = source_points + np.array([0.5, 0.5, 0.5])
    target.points = o3d.utility.Vector3dVector(target_points)
    
    return source, target

# Perform iterative ICP registration
def icp_registration(source, target):
    threshold = 0.08
    trans_init = np.eye(4)
    draw_registration_result(source, target, trans_init)
    
    print("Initial alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
    print(evaluation)
    
    max_iterations = 4
    current_transformation = trans_init
    
    for i in range(max_iterations):
        print(f"Iteration {i+1}")
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, current_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        
        current_transformation = reg_p2p.transformation
        print(reg_p2p)
        print("Transformation is:")
        print(current_transformation)
        print("")
    
    draw_registration_result(source, target, current_transformation)
    
    return current_transformation

if __name__ == "__main__":
    source, target = generate_point_clouds()
    transformation = icp_registration(source, target)