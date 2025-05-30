import os
import glob
import numpy     as np
import SimpleITK as sitk
from   tqdm      import tqdm
'''
Abdominal images are flipped horizontally, and when we run TotalSegmentator we did not get
proper segmentations of the liver. Therefore, this class, was done to get the data on the right view.
'''

class Horizontal_Flip(object):
    def __init__(self):
        super(Horizontal_Flip, self).__init__()
        
        
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
                
                image       = sitk.ReadImage(scan)
                image_array = sitk.GetArrayFromImage(image)

                # Flip the image array horizontally
                flipped_array = np.flip(image_array, axis=-1)
                flipped_image = sitk.GetImageFromArray(flipped_array)

                # Copy the metadata (origin, spacing, direction) from the original image
                flipped_image.SetOrigin(image.GetOrigin())
                flipped_image.SetSpacing(image.GetSpacing())
                flipped_image.SetDirection(image.GetDirection())

                # Save the flipped image to a new file
                new_path = output_folder + scan.split('/')[-1]
                sitk.WriteImage(flipped_image, new_path)
                
                pbar.update(1)



if __name__ == "__main__":
    horflip = Horizontal_Flip()
    input_folder  = './AbdomenCTCT/imagesTs/'
    output_folder = './AbdomenCTCT/images_ts/'
    
    horflip.make_folder_to_save(output_folder)
    data = horflip.read_data(input_folder)
    horflip.get_segmentations(data, output_folder)