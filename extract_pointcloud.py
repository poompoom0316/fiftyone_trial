import open3d as o3d
import numpy as np


def main():
    point_cloud_path = "analysis/exports/plant_pot2/featuresplatting/pointcloud.ply"
    ply_path = "analysis/exports/plant_pot2/featuresplatting/splat.ply"
    segment_path = "analysis/plant_pot2/feature/feature.ply"

    pc = o3d.io.read_point_cloud(point_cloud_path)
    segmentation = o3d.io.read_point_cloud(segment_path)

    pc_points = np.array(pc.points)
    pc_color = np.array(pc.colors)
    b1 = np.array(segmentation.points)
    b2 = np.array(segmentation.colors)

    opacity_filtered_loc = b2[:, 2]>0
    b1_filtered = b1[opacity_filtered_loc]
    b2_filtered = b2[opacity_filtered_loc]

    segmentation_filter = (b2_filtered[:, 0] > 0.9) & ((b2_filtered[:, 1] < 0.5))
    # segmentation_filter = (b2_filtered[:, 0] > 0.9)

    pc_filterd = o3d.geometry.PointCloud()
    pc_filterd.points = o3d.utility.Vector3dVector(pc_points[segmentation_filter])
    pc_filterd.colors = o3d.utility.Vector3dVector(pc_color[segmentation_filter])

    pc2 = o3d.geometry.PointCloud()
    pc2.points = o3d.utility.Vector3dVector(pc_points)
    pc2.colors = o3d.utility.Vector3dVector(pc_color)


    o3d.visualization.draw_geometries([pc_filterd])
    o3d.visualization.draw_geometries([pc2])




def main3():
    point_cloud_path = "analysis/exports/plant_pot2/featuresplatting3/pointcloud.ply"
    ply_path = "analysis/exports/plant_pot2/featuresplatting3/splat.ply"
    segment_path = "analysis/plant_pot2/feature/test3/feature.ply"

    pc = o3d.io.read_point_cloud(point_cloud_path)
    pc_splat = o3d.io.read_point_cloud(ply_path)
    segmentation = o3d.io.read_point_cloud(segment_path)

    a1 = np.array(pc.points)
    b1 = np.array(segmentation.points)
    b2 = np.array(segmentation.colors)
    c1 = np.array(pc_splat.points)
