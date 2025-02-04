# SAM annotation refinement

 This repository uses [SAM](https://github.com/facebookresearch/segment-anything) to refine coarse annotations in a fully automated process. The approach is demonstrated using the Cityscapes dataset, which contains 2975 fine and 20000 coarse annotated images. 

<p align="center">
  <img src="assets/1_coarse.png" width="49%">
  <img src="assets/1_refined.png" width="49%">
</p>

## Installation

Install [SAM](https://github.com/facebookresearch/segment-anything) as mentioned in their repository: 

> The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`.  Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.
>  
> Install Segment Anything:  
>  
> ```bash
> pip install git+https://github.com/facebookresearch/segment-anything.git
> ```

After this, install the dependencies of this repository
```
pip install -r requirements.txt
```

## Dataset

To use the code the [Cityscapes](https://www.cityscapes-dataset.com/) is needed. Follow these instructions to prepare the necessary files: 

https://github.com/mcordts/cityscapesScripts


## Perform refinement

To perform the refinement you need the default [SAM checkpoint (ViT-H SAM model)](https://github.com/facebookresearch/segment-anything#model-checkpoints). 

Then run 
```
python sam_annotation_refinement.py --data_root /path/to/cityscapes --checkpoint /path/to/sam_vit_h_4b8939.pth
```
with the **required** arguments: 

- `--data_root` → Path to the Cityscapes dataset.
- `--checkpoint` → Path to the default SAM checkpoint file.

## Visualize refinement

After the refinement is performed, the results for an image with `ID` can be visualized by running

```
python visualize_refinement.py --data_root /path/to/cityscapes --id ID
```

The visualization will be stored in the `plots/` folder.

## Acknowledgements

[SAM](https://github.com/facebookresearch/segment-anything)

```bibtex
@article{kirillov2023segany,
title={Segment Anything}, 
author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
journal={arXiv:2304.02643},
year={2023}
}
```