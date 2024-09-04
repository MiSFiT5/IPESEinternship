import pandas as pd
import open3d as o3d
import numpy as np

def harris_identification(input_file, keypoints_file, mesh_file):
    # 读取 CSV 文件
    df_read = pd.read_csv(input_file)
    df = df_read[['0', '1', '2', 'Cluster_KMeans']]  # 包含聚类结果列
    
    # 确保数据是 n x 4 的格式
    assert df.shape[1] == 4, "输入的 CSV 文件必须有 4 列（PC1, PC2, PC3, cluster）。"
    
    # 将 DataFrame 转换为 numpy 数组
    points = df[['0', '1', '2']].values
    clusters = df['Cluster_KMeans'].values
    
    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 可视化原始点云并设置点大小
    def visualize_point_cloud_with_size(pcd, point_size=5.0):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.point_size = point_size
        vis.run()
        vis.destroy_window()
    
    visualize_point_cloud_with_size(pcd)
    
    # 可视化网格并显示线框
    def visualize_mesh(mesh):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(mesh)
        opt = vis.get_render_option()
        opt.mesh_show_wireframe = True
        vis.run()
        vis.destroy_window()
    
    # 下采样点云
    voxel_size = 0.02
    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # 可视化下采样后的点云并设置点大小
    visualize_point_cloud_with_size(down_pcd)
    
    # 计算法向量
    down_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    
    # 可视化点云和法向量并设置点大小
    visualize_point_cloud_with_size(down_pcd)
    
    # 获取法向量
    normals = np.asarray(down_pcd.normals)
    
    # Harris3D 参数
    radius = 0.05
    k = 0.04
    
    # 创建KDTree用于邻域搜索
    kd_tree = o3d.geometry.KDTreeFlann(down_pcd)
    
    # 计算 Harris 响应
    harris_response = []
    
    for i, point in enumerate(down_pcd.points):
        [_, idx, _] = kd_tree.search_radius_vector_3d(point, radius)
        if len(idx) < 3:
            harris_response.append(0)
            continue
        
        neighbors = np.asarray(down_pcd.points)[idx, :]
        neighbors = neighbors - np.mean(neighbors, axis=0)
        
        cov_matrix = np.cov(neighbors.T)
        eigenvalues, _ = np.linalg.eigh(cov_matrix)
        
        response = eigenvalues[0] * eigenvalues[1] - k * (eigenvalues[0] + eigenvalues[1]) ** 2
        harris_response.append(response)
    
    harris_response = np.array(harris_response)
    
    # 非极大值抑制
    max_response = np.max(harris_response)
    threshold = 0.01 * max_response
    
    keypoints = []
    
    for i, response in enumerate(harris_response):
        if response > threshold:
            [_, idx, _] = kd_tree.search_radius_vector_3d(down_pcd.points[i], radius)
            if response == np.max(harris_response[idx]):
                keypoints.append(down_pcd.points[i])
    
    # 将关键点转换为 PointCloud 对象并可视化
    keypoints_pcd = o3d.geometry.PointCloud()
    keypoints_pcd.points = o3d.utility.Vector3dVector(keypoints)
    visualize_point_cloud_with_size(keypoints_pcd, point_size=10.0)
    
    # 将关键点转换为 numpy 数组
    keypoints_np = np.asarray(keypoints)
    
    # 创建一个KDTree用于高效的最近邻搜索
    pcd_tree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)))
    
    # 找到关键点在原始点云数据中的索引
    keypoint_indices = []
    for keypoint in keypoints_np:
        [_, idx, _] = pcd_tree.search_knn_vector_3d(keypoint, 1)
        keypoint_indices.append(idx[0])
    
    # 将关键点的原始序号和聚类结果添加到 DataFrame 中
    keypoints_df = pd.DataFrame(keypoints_np, columns=["x", "y", "z"])
    keypoints_df['original_index'] = keypoint_indices
    keypoints_df['Cluster'] = [clusters[idx] for idx in keypoint_indices]
    
    # 去重处理
    keypoints_df = keypoints_df.drop_duplicates()
    
    # 保存包含原始序号和聚类结果的关键点数据
    keypoints_df.to_csv(keypoints_file, index=False)
    
    # 输出结果
    print(keypoints_df)
    
    # 从关键点创建网格并可视化
    keypoints_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    keypoints_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(keypoints_pcd, depth=9)
    bbox = keypoints_pcd.get_axis_aligned_bounding_box()
    keypoints_mesh = keypoints_mesh.crop(bbox)
    keypoints_mesh.paint_uniform_color([0.7, 0.7, 0.7])
    visualize_mesh(keypoints_mesh)
    
    keypoints_df.to_csv(mesh_file, index=False)
