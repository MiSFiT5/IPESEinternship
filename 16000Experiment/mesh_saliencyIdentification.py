# mesh_saliency_identification.py
import pandas as pd
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

def mesh_saliency_identification(input_file, output_file_saliency, output_file_red_region):
    # 读取 CSV 文件
    df_read = pd.read_csv(input_file)
    df = df_read[['0', '1', '2', 'Cluster_KMeans']]  # 包含聚类结果列
    assert df.shape[1] == 4, "输入的 CSV 文件必须有 4 列（PC1, PC2, PC3, cluster）。"

    # 将 DataFrame 转换为 numpy 数组
    points = df[['0', '1', '2']].values
    clusters = df['Cluster_KMeans'].values

    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 下采样点云
    voxel_size = 0.02
    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # 计算法向量
    down_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))

    # 使用泊松表面重建算法构建网格
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(down_pcd, depth=9)

    # 裁剪网格以移除可能的孤立区域或异常
    bbox = down_pcd.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)

    # 计算曲率
    def compute_curvature(mesh):
        mesh.compute_vertex_normals()
        vertices = np.asarray(mesh.vertices)
        normals = np.asarray(mesh.vertex_normals)

        curvature = np.zeros(len(vertices))
        for i in range(len(vertices)):
            curvature[i] = np.linalg.norm(normals[i])
        return curvature

    # 计算显著性
    def compute_saliency(mesh, curvature, sigma=1.0):
        vertices = np.asarray(mesh.vertices)
        n = len(vertices)
        saliency = np.zeros(n)

        for i in range(n):
            diff = curvature - curvature[i]
            dist = np.linalg.norm(vertices - vertices[i], axis=1)
            weight = np.exp(-dist**2 / (2 * sigma**2))
            saliency[i] = np.sum(weight * np.abs(diff))

        return saliency

    # 平滑显著性值
    def smooth_saliency(saliency, iterations=10, alpha=0.1):
        for _ in range(iterations):
            saliency = (1 - alpha) * saliency + alpha * np.mean(saliency)
        return saliency

    # 计算曲率和显著性
    curvature = compute_curvature(mesh)
    saliency = compute_saliency(mesh, curvature)
    smooth_saliency = smooth_saliency(saliency)

    # 显著性值归一化
    smooth_saliency = (smooth_saliency - np.min(smooth_saliency)) / (np.max(smooth_saliency) - np.min(smooth_saliency))

    # 设置阈值来确定红色区域
    threshold = 0.8  # 你可以调整这个阈值

    # 找到显著性值高于阈值的顶点
    red_region_indices = np.where(smooth_saliency > threshold)[0]
    red_region_vertices = np.asarray(mesh.vertices)[red_region_indices]

    # 打印红色区域的坐标
    print("红色区域的坐标：")
    for vertex in red_region_vertices:
        print(vertex)

    # 使用KD树在原始点云中找到最近邻点
    kd_tree = KDTree(points)
    _, original_indices = kd_tree.query(red_region_vertices)

    # 获取原始点云中对应的点和它们的序号
    corresponding_points = points[original_indices]
    corresponding_indices = original_indices

    # 将显著性结果保存到CSV文件
    vertices = np.asarray(mesh.vertices)
    saliency_df = pd.DataFrame(vertices, columns=["x", "y", "z"])
    saliency_df['saliency'] = smooth_saliency
    saliency_df.to_csv(output_file_saliency, index=False)

    # 保存映射回原始数据集的红色区域点及其序号
    red_region_df = pd.DataFrame(corresponding_points, columns=["x", "y", "z"])
    red_region_df['original_index'] = corresponding_indices
    red_region_df.to_csv(output_file_red_region, index=False)

    # 打印映射回原始数据集的红色区域点及其序号
    print("映射回原始数据集的红色区域点及其序号：")
    print(red_region_df)

    # 将显著性值映射到颜色上
    colors = plt.cm.jet(smooth_saliency)[:, :3]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # 可视化显著性结果
    o3d.visualization.draw_geometries([mesh])

    # 输出 DataFrame 以便用户查看
    print(saliency_df)
    print(red_region_df)

if __name__ == "__main__":
    input_file = "data_with_labels.csv"
    output_file_saliency = "mesh_saliency.csv"
    output_file_red_region = "original_data_red_region.csv"
    mesh_saliency_identification(input_file, output_file_saliency, output_file_red_region)
