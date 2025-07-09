import numpy as np
import open3d as o3d
import torch
import os


def create_output_dir(output_path, dataset_name):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print('Create output path successfully at %s' % output_path)
    if dataset_name == 'PCN':
        plane_path = os.path.join(output_path, '02691156')
        cabinet_path = os.path.join(output_path, '02933112')
        car_path = os.path.join(output_path, '02958343')
        chair_path = os.path.join(output_path, '03001627')
        lamp_path = os.path.join(output_path, '03636649')
        couch_path = os.path.join(output_path, '04256520')
        table_path = os.path.join(output_path, '04379243')
        watercraft_path = os.path.join(output_path, '04530566')
        category_path = [plane_path, cabinet_path, car_path, chair_path, lamp_path, couch_path,
                         table_path, watercraft_path]
        for path in category_path:
            if not os.path.exists(path):
                os.makedirs(path)


def savePointCloud(filePath, ptc):
    '''
    Save the point cloud file by open3d
    :param filePath: the path of output file
    :param ptc: point cloud,  ndarray [tensor] [numpy], size: N*3.
    :return:
    '''
    if torch.is_tensor(ptc):
        ptc_array = ptc.cpu().numpy()
    else:
        ptc_array = ptc
    ptc_o3d = o3d.geometry.PointCloud()
    ptc_o3d.points = o3d.utility.Vector3dVector(ptc_array)
    o3d.io.write_point_cloud(filePath, pointcloud=ptc_o3d)


def pointVisualize(ptcloud):
    '''
    点云可视化
    :param ptcloud: 点云数据 n*3 numpy h5py等
    '''
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=500, height=500, window_name='ptc')  # 创建窗口
    render_option: o3d.visualization.RenderOption = vis.get_render_option()  # 设置点云渲染参数
    render_option.background_color = np.array([255, 255, 255])  # 设置背景色（这里为黑色）
    render_option.point_size = 2.0  # 设置渲染点的大小
    ptc = o3d.geometry.PointCloud()
    ptc.points = o3d.utility.Vector3dVector(ptcloud)
    ptc.paint_uniform_color([0, 0, 0])
    vis.add_geometry(ptc)  # 添加点云
    vis.run()


def upSamplePoints(ptcloud, npoints):
    """
    upsample points
    :param ptcloud: numpy data point cloud
    :param npoints: output points' number
    :return: upsample points [numpy]
    """
    curr = ptcloud.shape[0]
    need = npoints - curr

    if need < 0:
        return ptcloud[np.random.permutation(npoints)]

    while curr <= need:
        ptcloud = np.tile(ptcloud, (2, 1))
        need -= curr
        curr *= 2

    choice = np.random.permutation(need)
    ptcloud = np.concatenate((ptcloud, ptcloud[choice]))

    return ptcloud


def toTensor(arr):
    return torch.from_numpy(arr.copy()).float()


def inferDataInit(ptc, model_in_points=2048):
    up_points = upSamplePoints(ptc, model_in_points)
    out_points = toTensor(up_points).unsqueeze(0).contiguous()
    return out_points
