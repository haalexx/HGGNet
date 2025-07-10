import torch
from utils.build_utils import load_model
from Datasets.readData import readData
from models.HGGNet import HGGNet
from utils.config import cfg_from_yaml_file
from utils.infer_utils import inferDataInit, pointVisualize
from pointFlowRender import pointRender

dataPath = "./Datasets/Completion3D/train/partial/02691156/1a04e3eab45ca15dd86060f189eb133.h5"
gtPath = "./Datasets/Completion3D/train/gt/02691156/1a04e3eab45ca15dd86060f189eb133.h5"
testPath = "./Datasets/Completion3D/test/partial/all/0001.h5"
input_ptcloud = readData.get(dataPath)
gt_ptcloud = readData.get(gtPath)
test_ptc = readData.get(testPath)

print(input_ptcloud.shape)
print(gt_ptcloud.shape)
print(test_ptc.shape)

pointVisualize(input_ptcloud)
pointVisualize(gt_ptcloud)

