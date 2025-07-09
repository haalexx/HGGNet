import torch
from ptflops import get_model_complexity_info
from thop import profile
# from models import HGGNet
from models.HGGNet_7_11 import HGGNet
from models.PoinTr import PoinTr
from utils.config import cfg_from_yaml_file

config = cfg_from_yaml_file("cfgs/model_configs/HGGNet_7_4.yaml")
model = HGGNet(config.model).to('cuda')
MACs, params = get_model_complexity_info(model, (2048, 3), print_per_layer_stat=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', MACs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))

# x = torch.randn((1, 3, 2048)).to('cuda')
# flops, params = profile(model, inputs=(x,))
# print("%s | params: %.2fM | flops: %.2fG" % ('MSPG', params / (1000 ** 2), flops / (1000 ** 3)))
