<h1 align="center">Medical Alignment Networks</h1>

## Introduction

This repository provides an implementation for the alignment of medical images. Medical Alignment Networks are designed to align pairs of images, whether they are from the same patient or different patients. Specifically, a moving image is aligned to a fixed image using both affine and deformable transformations.


⚠️ **Important**
This project is actively progressing, with ongoing refinements to the code. We are happy to hear any suggestions or feedback you might have. Don’t hesitate to open an issue or submit a pull request!




## Datasets

We use two evaluation datasets:

- The Learn2Reg challenge dataset, consisting of lung CT scans for inspiration and expiration.
- A private dataset from the Netherlands Cancer Institute, consisting of liver CT scans.

The Learn2Reg dataset can be downloaded from the following link:

- https://cloud.imi.uni-luebeck.de/s/o7LyCbJCie8fQ3B



## Models

In addition to our method, you can use the following baseline models to train and evaluate your dataset or to replicate our results:

- VoxelMorph (VXM)
- Volume Tweening Network (VTN)
- TransMorph (TSM)
- Contrastive Registration for Unsupervised Medical Image Segmentation (CLMorph) 
- Elastix
- ANTs

Note: Elastix and ANTs are used only for evaluation with their default configurations.





## Training

To train our model, you can use the following command:

```
python train.py -d datasets/LungCT_dataset.json 
```

To train the baseline models, specify the appropriate base name (VXM, VTN, TSM, or CLM). For example, to train the VoxelMorph model, use the following command:

```
python train.py -d datasets/LungCT_dataset.json --name vxm -base VXM
```

## Evaluation

To evaluate our model and baselines, use the following command:

```
python eval/eval.py -d datasets/LungCT_dataset.json -c path_to_checkpoint_dir
```

If you want to save the aligned images, specify it and add the directory where the images should be saved:

```
python eval/eval.py -d datasets/LungCT_dataset.json -c path_to_checkpoint_dir -sn -ndir path_to_save_images
```

For Elastix and ANTs, specify the model and assign a name for the experiment to create the output file and folder. For example, for Elastix, use the following command:

```
python eval/eval.py -d datasets/LungCT_dataset.json -c path_to_checkpoint_dir -ue -en elastix 
```

## Acknowledgments

We would like to thank the following repositories for their contributions and inspiration:

- https://github.com/microsoft/Recursive-Cascaded-Networks?tab=readme-ov-file
- https://github.com/ivan-jgr/recursive-cascaded-networks
- https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/tree/main/TransMorph
- https://github.com/dddraxxx/Medical-Reg-with-Volume-Preserving
