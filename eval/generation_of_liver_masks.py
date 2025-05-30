import os
import pandas  as pd
import os
import glob
#from cts_operations import ReadVolume
from   tqdm          import tqdm

class Total_Segmentator(object):
    def __init__(self):
        super(Total_Segmentator, self).__init__()
        
        
    def read_data(self, input_folder):
        if not input_folder.endswith(os.path.sep):
            input_folder += os.path.sep
        nii_gz_files = glob.glob(os.path.join(input_folder, '*.nii.gz'))
        return nii_gz_files


    def make_folder_to_save(self, output_folder):
        if not os.path.exists(output_folder):
            print('Creating folder...')
            os.makedirs(output_folder)
        else:
            print('This folder already exists :)!')
            
            
    def get_segmentations(self, data, output_folder):
        
        with tqdm(total=len(data)) as pbar:
            
            for scan in data:
                
                try:        
                    seg_name  = output_folder + scan.split('/')[-1].replace('.nii.gz', '')
                    #print(seg_name)
                    ts_cmd = f"TotalSegmentator -i {scan} -o {seg_name} --roi_subset liver --statistics"
                    os.system(ts_cmd)
                except:
                    print("--------------- PRIOR CT was not loaded! ---------------")
                    print(scan)
                pbar.update(1)



if __name__ == "__main__":
    totseg = Total_Segmentator()
    input_folder  = './AbdomenCTCT/images_ts/'
    output_folder = './AbdomenCTCT/labels_ts/'
    
    totseg.make_folder_to_save(output_folder)
    
    data = totseg.read_data(input_folder)
    
    totseg.get_segmentations(data, output_folder)