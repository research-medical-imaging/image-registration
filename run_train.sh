#!/bin/bash
#SBATCH --time=6-00:00:00                        # Time limit hrs:min:sec
#SBATCH --job-name=train                         # Job name
#SBATCH --qos=a6000_qos
#SBATCH --partition=rtx8000                      # Partition
#SBATCH --nodelist=roentgen                      # Node name
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --output=./logs/train_%j.log   # Standard output and error log
pwd; hostname; date

# Activate conda environment pyenv
source /home/miniconda3/bin/activate pytorch
rsync -avv --info=progress2 --ignore-existing ./LungCT /processing/

# Run your command
python /projects/image-registration/train.py -base VXM -reg 0.1
python /projects/image-registration/train.py -base VXM -reg 1
python /projects/image-registration/train.py -base VXM -reg 10
python /projects/image-registration/train.py -base VTN -reg 0.1
python /projects/image-registration/train.py -base VTN -reg 1
python /projects/image-registration/train.py -base VTN -reg 10
python /projects/image-registration/train.py -base TSM -reg 0.1
python /projects/image-registration/train.py -base TSM -reg 1
python /projects/image-registration/train.py -base TSM -reg 10
python /projects/image-registration/train.py -base CLM -reg 0.1  
python /projects/image-registration/train.py -base CLM -reg 1    
python /projects/image-registration/train.py -base CLM -reg 10 