import torch
import numpy as np
from pathlib import Path as pa
#---->from tools.utils import *

stage1_cfg = {
    'liver': {
        'VXM': './logs/liver/VXM/1/Apr01-022512_1_VXMx1_normal__',
        'VTN': './logs/liver/VTN/1/Apr02-150232_1_VTNx3_normal__',
        'TSM': './logs/liver/TSM/1/Apr03-140648_1_TSMx1_normal__',
        'CLM': './logs/liver/CLM/1/Mar04-011015_1_TSMx1_normal__',
    },
    'lung': {
        'VXM': './logs/brain/VXM/1/Apr08-160947_1_VXMx1_normal__',
        'VTN': './logs/brain/VTN/1/Apr08-070318_1_VTNx3_normal___',
        'TSM': './logs/brain/TSM/1/Apr08-211218_1_TSMx1_normal__',
        'CLM': './logs/brain/CLM/1/Apr08-200438_1_TSMx1_normal__',
    },
}
import os
for i in stage1_cfg:
    for j in stage1_cfg[i]:
        stage1_cfg[i][j] = os.path.realpath(stage1_cfg[i][j])

def build_precompute(model, dataset, cfg):
    # get template image and segmentation
    template                     = list(dataset.subset['temp'].values())[0]
    template_image, template_seg = template['volume'], template['segmentation']
    # convert to torch tensors and move to GPU
    template_image = torch.tensor(np.array(template_image).astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()/255.0
    template_seg   = torch.tensor(np.array(template_seg).astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
    # load pretrained model
    state_path = stage1_cfg[cfg.data_type][cfg.base_network]
    return model.build_preregister(template_image, template_seg, state_path, cfg.base_network)


def read_cfg(model_path):
    print('Model Path:', model_path)
    # Traverse up the directory tree until we find the folder where the model is saved
    while 'model_wts' in str(model_path):
        model_path = pa(model_path).parent
    # Get the model name
    name     = pa(model_path).stem.split('_')[-1]
    cfg_path = pa(model_path) / 'args.txt'
    # Read the first line of the file
    cfg = open(cfg_path).read().split('\n\n')[0]
    # Replace single quotes with double quotes
    cfg = cfg.replace("'", '"')
    # Convert the string to a dictionary
    cfg = eval(cfg)
    cfg['name'] = name
    # Convert the dictionary to a ConfigDict
    import ml_collections
    cfg = ml_collections.ConfigDict(cfg)
    print(cfg)
    return cfg