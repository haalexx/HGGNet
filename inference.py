import torch
from utils.build_utils import load_model
from Datasets.readData import readData
from models.HGGNet import HGGNet
from utils.config import cfg_from_yaml_file
from utils.infer_utils import inferDataInit, pointVisualize
from pointFlowRender import pointRender

config = cfg_from_yaml_file("cfgs/model_configs/HGGNet.yaml")
model = HGGNet(config.model).to('cuda')
load_model(model, 'checkpoints/HGG-noise-21.pth')
model.eval()
dataPath = "./Datasets/ShapeNetCompletion/test/partial/04379243/974cc395f9684d47c955e5ed03ef3a2f/00.pcd"
gtPath = "./Datasets/ShapeNetCompletion/test/complete/04379243/974cc395f9684d47c955e5ed03ef3a2f.pcd"
# dataPath = "./Datasets/KITTI/cars/frame_0_car_0.pcd"
input_ptcloud = readData.get(dataPath)
gt_ptcloud = readData.get(gtPath)
model_input = inferDataInit(input_ptcloud).to('cuda')
with torch.no_grad():
    sparse_points, dense_points = model(model_input)

print(sparse_points.shape)
print(dense_points.shape)
dense_points = dense_points.squeeze(0).cpu().numpy()
# pointVisualize(input_ptcloud)
# pointVisualize(gt_ptcloud)
# pointVisualize(dense_points)
pointRender('input.xml', input_ptcloud, [32, 88, 103])
pointRender('gt.xml', gt_ptcloud, [32, 88, 103])

pointRender('output.xml', dense_points, [245, 157, 86])
