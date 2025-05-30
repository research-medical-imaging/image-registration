#%%
import json
import torch
import random
import pandas as pd
from torch.utils.data import Dataset
import sys
sys.path.append("..")
from processing.cts_processors import ScanProcessor
from processing.cts_operations import ReadVolume
from processing.cts_operations import ToNumpyArray
from processing.cts_operations import ToNumpyArrayAbdomen
from processing.cts_operations import ToLungWindowLevelNormalization
from processing.cts_operations import ToAbdomenWindowLevelNormalization

class RawDataLung():
    def __init__(self, json_file, root_dir, mode='train', transform=None):
        with open(json_file, 'r') as file:
            data_info = json.load(file)
        self.root_dir  = root_dir
        self.mode      = mode
        self.transform = transform
        self.inp_dtype = torch.float32
        self.scan_loader  = self.__init_scan_loader()
        self.label_loader = self.__init_label_loader()
        
        # We didn't consider registration_val since they are the three first elements of the training dataset
        # This 3 elements have landmarks information
        mode_mapping = {
            'train': 'training',
            'val':   'training',
            'test' : 'test'
        }
        
        if self.mode not in mode_mapping:
            raise ValueError('mode can only be train, val, or test')
        
        data = data_info[mode_mapping[self.mode]]
        
        if self.mode == 'test':
            self.data = self.get_pairs_with_gt_test(data)
        else:
            self.data = self.get_pairs_with_gt(data)
            if self.mode == 'val':
                self.data = self.data[:3]
            
        self.add_root_dir_to_paths()
    
    
    
    def get_pairs_with_gt(self, data):
        # Calculate the midpoint index
        midpoint_index = len(data) // 2
        pairs = []

        # Create pairs from the first half and the second half
        for i in range(midpoint_index):
            pair = {
                'fix': data[i],
                'mov': data[i + midpoint_index]
            }
            pairs.append(pair)
        return pairs
    
    
    def get_pairs_with_gt_test(self, data):
        pairs = []
        # Iterate over data to create pairs
        for i in range(0, len(data), 2):
            fix_image = data[i]
            mov_image = data[i + 1]
            
            pairs.append({
                'fix': fix_image,
                'mov': mov_image
            })
        return pairs
    
    
    def add_root_dir_to_paths(self):
        for pair in self.data:
            for key, value in pair.items():
                # Iterate through each key-value pair in the nested dictionary
                for item_key, item_value in value.items():
                    # Update the path with the new project path
                    pair[key][item_key] = self.root_dir + item_value.lstrip('./')
    
                    
    
    def read_keypoints(self, file):
        kps = pd.read_csv(file, header=None).values.astype(int)
        return kps
        
    def __init_label_loader(self):
        return ScanProcessor(
            ReadVolume(),
            ToNumpyArray()
        )
        
              
    def __init_scan_loader(self):
        return ScanProcessor(
            ReadVolume(),
            ToLungWindowLevelNormalization()
        )
    
    def __len__(self):
        return len(self.data)
    
    
    
    def __getitem__(self, idx: int):
        
        ret = {}

        voxel1        = torch.from_numpy(self.scan_loader(self.data[idx]['fix']['image'])).type(self.inp_dtype)
        voxel2        = torch.from_numpy(self.scan_loader(self.data[idx]['mov']['image'])).type(self.inp_dtype)
        segmentation1 = torch.from_numpy(self.label_loader(self.data[idx]['fix']['mask'])).type(self.inp_dtype)
        segmentation2 = torch.from_numpy(self.label_loader(self.data[idx]['mov']['mask'])).type(self.inp_dtype)
        kps1          = torch.from_numpy(self.read_keypoints(self.data[idx]['fix']['keypoints'])).type(self.inp_dtype)
        kps2          = torch.from_numpy(self.read_keypoints(self.data[idx]['mov']['keypoints'])).type(self.inp_dtype)
        
        ret['img1_path']     = self.data[idx]['fix']['image']
        ret['img2_path']     = self.data[idx]['mov']['image']
        ret['voxel1']        = voxel1[None, :]
        ret['voxel2']        = voxel2[None, :]
        ret['segmentation1'] = segmentation1[None, :]
        ret['segmentation2'] = segmentation2[None, :]
        ret['kps1']          = kps1
        ret['kps2']          = kps2
        
        if self.mode == 'val':
            lmks1 = torch.from_numpy(self.read_keypoints(self.data[idx]['fix']['landmarks']))
            lmks2 = torch.from_numpy(self.read_keypoints(self.data[idx]['mov']['landmarks']))
            ret['lmks1'] = lmks1
            ret['lmks2'] = lmks2
        return ret
            


class RawDataAbdomen():
    def __init__(self, json_file, root_dir, mode='train', transform=None):
        with open(json_file, 'r') as file:
            data_info = json.load(file)
        self.root_dir  = root_dir
        self.mode      = mode
        self.transform = transform
        self.inp_dtype = torch.float32
        self.scan_loader  = self.__init_scan_loader()
        self.label_loader = self.__init_label_loader()
        random.seed(2023)
        # We didn't consider registration_val since they are the three first elements of the training dataset
        # This 3 elements have landmarks information
        mode_mapping = {
            'train': 'training',
            'val':   'registration_val',
            'test' : 'registration_test'
        }
        
        if self.mode not in mode_mapping:
            raise ValueError('mode can only be train, val, or test')
        
        data      = data_info[mode_mapping[self.mode]]
        
        if self.mode == 'train':   
            data_temp = data_info[mode_mapping['val']]
            self.data = self.get_pairs_with_gt_train(data, data_temp)
        else:
            self.data = self.get_pairs_with_gt_test(data)
        self.add_root_dir_to_paths()
    
    
    
    def get_pairs_with_gt_train(self, data, data_temp):
        val_list  = sorted(list(set( [x['fixed'] for x in data_temp] + [x['moving'] for x in data_temp] )))
        training_ = [x for x in data if x['image'] not in val_list]
        pairs     = []
        
        for entry in training_:
            fixed_image  = entry['image']
            moving_image = random.choice(training_)['image']
            fixed_mask   = fixed_image.replace('imagesTr', 'labelsTr')
            
            for i in range(3):
                moving_mask  = moving_image.replace('imagesTr', 'labelsTr')
                pair = {
                    'fix': fixed_image,
                    'mov': moving_image,
                    'fix_seg': fixed_mask,
                    'mov_seg': moving_mask
                }
                pairs.append(pair)
        return pairs
    
    
    def get_pairs_with_gt_test(self, data):
        pairs     = []
        for entry in data:
            fixed_image  = entry['fixed']
            moving_image = entry['moving']
            fixed_mask   = fixed_image.replace('imagesTs', 'labelsTs')
            moving_mask  = moving_image.replace('imagesTs', 'labelsTs')
            
            pair = {
                'fix': fixed_image,
                'mov': moving_image,
                'fix_seg': fixed_mask,
                'mov_seg': moving_mask
            }
            pairs.append(pair)
        return pairs
    
    
    def add_root_dir_to_paths(self):
        for pair in self.data:
            for key, value in pair.items():
                pair[key] = self.root_dir + value.lstrip('./')
                    
                    
        
    def __init_label_loader(self):
        return ScanProcessor(
            ReadVolume(),
            ToNumpyArrayAbdomen()
        )
        
              
    def __init_scan_loader(self):
        return ScanProcessor(
            ReadVolume(),
            ToAbdomenWindowLevelNormalization()
        )
    
    def __len__(self):
        return len(self.data)
    
    
    
    def __getitem__(self, idx: int):
        
        ret = {}
        voxel1        = torch.from_numpy(self.scan_loader(self.data[idx]['fix'])).type(self.inp_dtype)
        voxel2        = torch.from_numpy(self.scan_loader(self.data[idx]['mov'])).type(self.inp_dtype)
        segmentation1 = torch.from_numpy(self.label_loader(self.data[idx]['fix_seg'])).type(self.inp_dtype)
        segmentation2 = torch.from_numpy(self.label_loader(self.data[idx]['mov_seg'])).type(self.inp_dtype)
        
        ret['img1_path']     = self.data[idx]['fix']
        ret['img2_path']     = self.data[idx]['mov']
        ret['voxel1']        = voxel1[None, :]
        ret['voxel2']        = voxel2[None, :]
        ret['segmentation1'] = segmentation1[None, :]
        ret['segmentation2'] = segmentation2[None, :]

        return ret



class RawData():  
    def __init__(self, args, **kwargs):    
        data_type      = 'Abdomen' if 'Abdomen' in args else 'lung'
        if data_type == 'lung':
            self.raw_data = RawDataLung(args, **kwargs)
        else:
            self.raw_data = RawDataAbdomen(args, **kwargs)
    
    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, idx):
        return self.raw_data[idx]
         
class Data(RawData, Dataset):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        #print('data - args: '. args)
        #print('data - kwargs: ', kwargs)
        

#%%
import sys 
sys.path.append('..')
from tools.utils         import show_img
from tools.visualization import plot_sample_data
from tools.visualization import plot_sample_data_and_kpts

import matplotlib.pyplot as plt
if __name__  == '__main__':
    # Lung dataset:
    data_file = './LungCT_dataset.json'
    root_dir  = './LungCT/'
    data      = Data(data_file, root_dir=root_dir, mode='val')
    plot_sample_data(data[0], slide=164, save_path='./164_.png')
    
    # Abdomen
    data_file = './AbdomenCTCT_dataset.json'
    root_dir  = './AbdomenCTCT/'
    data      = Data(data_file, root_dir=root_dir, mode='train')
    print(len(data))
    plot_sample_data(data[0], slide=164, save_path='./164_abd.png')
    #print(data[0])
