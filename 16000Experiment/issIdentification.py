import pandas as pd
import open3d as o3d
import numpy as np

def iss_identification(input_file, keypoints_file, mesh_file, simplified_mesh_file):
    # 读取数据
    df = pd.read_csv(input_file)
    points = df[['0', '1', '2']].values

    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 计算法向量
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # ISS 参数
    salient_radius = 0.5
    non_max_radius = 0.5
    gamma_21 = 0.975
    gamma_32 = 0.975
    min_neighbors = 5

    # 计算ISS关键点
    keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd, salient_radius=salient_radius,
                                                            non_max_radius=non_max_radius,
                                                            gamma_21=gamma_21, gamma_32=gamma_32,
                                                            min_neighbors=min_neighbors)

    # 保存关键点
    keypoints_df = pd.DataFrame(np.asarray(keypoints.points), columns=['0', '1', '2'])
    keypoints_df.to_csv(keypoints_file, index=False)

    # 简化网格
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=0.03)
    mesh = mesh.simplify_quadric_decimation(1000)
    o3d.io.write_triangle_mesh(simplified_mesh_file, mesh)

    # 保存网格
    o3d.io.write_triangle_mesh(mesh_file, mesh)
