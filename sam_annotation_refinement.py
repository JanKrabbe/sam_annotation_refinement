import argparse
from typing import Tuple, Optional

import numpy as np
import torch
import torchmetrics
import torchvision.transforms as T
from tqdm import tqdm

from skimage.measure import label, regionprops
from torchvision.datasets import Cityscapes
from segment_anything import SamPredictor, sam_model_registry

def parse_args():
    parser = argparse.ArgumentParser(description="Refine coarse Cityscapes annotaions with SAM.")

    parser.add_argument("--data_root", type=str,
                        help="Path to the Cityscapes dataset")
    
    parser.add_argument("--checkpoint", type=str,
                        help="Path to the SAM checkpoint")
    
    parser.add_argument("--n_points", type=int, default=9,
                        help="Number of points to use as prompts")
    
    parser.add_argument("--box_factor", type=float, default=0.4,
                        help="Factor for expanding the bounding box (-1 for no box)")
    
    args = parser.parse_args()

    return args

def expand_bounding_box(
    bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
    enlargement_factor: float = 0.075
) -> np.ndarray:
    """
    Expand the given bounding box by a percentage.

    Arguments:
        bbox (tuple(int, int, int, int)): Original bounding box as (min_row, min_col, max_row,
          max_col).
        image_shape (tuple(int, int)): Dimensions of the image as (rows, cols).
        enlargement_factor (float): Factor to expand the bounding box.

    Returns:
        (np.array): The expanded bounding box as a NumPy array.
    """
    min_row, min_col, max_row, max_col = bbox
    rows, cols = image_shape
    expansion_rows, expansion_cols = np.rint(
        np.array([max_row - min_row, max_col - min_col]) * enlargement_factor / 2
    ).astype(int)
    return np.array([
        max(0, min_row - expansion_rows),
        max(0, min_col - expansion_cols),
        min(rows, max_row + expansion_rows),
        min(cols, max_col + expansion_cols)
    ])

def refine_instance(
    predictor: SamPredictor,
    instance_mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    n_points: int,
    box_factor: float,
) -> torch.Tensor:
    """
    Refine a single segmentation instance using the SAM predictor.

    Arguments:
        predictor (SamPredictor): The SAM predictor.
        instance_mask (np.ndarray): Binary mask of the instance as a NumPy array in HxW format, 
          where (H, W) is the image size.
        bbox (tuple(int, int, int, int)): Bounding box containing the instance (min_row, min_col,
          max_row, max_col).
        n_points (int): Number of prompt points.
        box_factor (float): Factor for expanding the bounding box (-1 for no box).

    Returns:
        (torch.Tensor): Refined mask as a Torch tensor in HxW format, where (H, W) is the image 
          size.
    """
    point_coords: Optional[np.ndarray] = None
    point_labels: Optional[np.ndarray] = None
    prompt_box: Optional[np.ndarray] = None

    if n_points > 0:
        pos_indices = torch.nonzero(torch.tensor(instance_mask))
        if pos_indices.size(0) > 0:
            rand_perm = torch.randperm(pos_indices.size(0))
            sampled_indices = rand_perm[:n_points]
            pos_sampled_points = pos_indices[sampled_indices][:, [1, 0]]  # (x, y) order
            pos_labels = torch.ones(pos_sampled_points.shape[0])
            point_coords = pos_sampled_points.numpy()
            point_labels = pos_labels.numpy()

    if box_factor != -1:
        expanded_box = expand_bounding_box(bbox, instance_mask.shape, box_factor)
        # Convert order and add batch dimension
        prompt_box = expanded_box[[1, 0, 3, 2]][None, :]

    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=prompt_box,
        mask_input=None,
        multimask_output=True
    )

    best_mask = torch.tensor(masks[scores.argmax()])
    
    return best_mask

def refine_sample(
    img,
    target_coarse,
    predictor: SamPredictor,
    n_points: int,
    box_factor: float,
) -> torch.Tensor:
    """
    Process a single image and refine the coarse annotation using SAM.

    Arguments:
        img: The PIL image for which the annotation is to be refined.
        target_coarse: Coarse annotation as PIL image.
        predictor (SamPredictor): The SAM predictor.
        n_points (int): Number of positive points for prompts.
        box_factor (float): Factor for expanding the bounding box.

    Returns:
        (torch.Tensor): Refined annotation for the processed image.
    """
    predictor.set_image(np.array(img))
    
    target_coarse = torch.tensor(np.array(target_coarse))
    refined_target = target_coarse.clone()

    class_ids = torch.unique(target_coarse)
    class_ids = class_ids[class_ids != 0]
    for class_id in class_ids:
        if class_id > 3:  # Skip the first 4 classes, since they are irrelevant
            mask = (target_coarse == class_id).numpy()
            labeled_mask = label(mask)
            for region_label in np.unique(labeled_mask)[1:]:
                instance_mask = labeled_mask == region_label
                props = regionprops(instance_mask.astype(int))
                if not props:
                    continue
                bbox = props[0].bbox  # (min_row, min_col, max_row, max_col)
                refined_instance_mask = refine_instance(
                    predictor, instance_mask, bbox,
                    n_points, box_factor
                )
                unlabeled_pixels = refined_target == 0
                refined_target[unlabeled_pixels] = refined_instance_mask[unlabeled_pixels] * class_id
    return refined_target

def main():
    args = parse_args()

    dataset_fine = Cityscapes(args.data_root, split='train', mode='fine', target_type='semantic')
    dataset_coarse = Cityscapes(args.data_root, split='train', mode='coarse', target_type='semantic')

    sam = sam_model_registry["default"](checkpoint=args.checkpoint).cuda()
    predictor = SamPredictor(sam)

    to_pil_image = T.ToPILImage()

    iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=34, ignore_index=255)

    for i in tqdm(range(len(dataset_coarse)), desc="Processing images"):
        img_fine, target_fine = dataset_fine[i]
        img_coarse, target_coarse = dataset_coarse[i]

        target_fine = torch.tensor(np.array(target_fine))
        
        refined_target = refine_sample(
            img_fine, target_coarse,
            predictor, args.n_points, args.box_factor
        )

        iou.update(target_fine, refined_target)


        refined_label_image = to_pil_image(refined_target)
        target_path = dataset_coarse.targets[i][0]
        save_path = target_path.replace("_gtCoarse", "_gtRefined")
        
        refined_label_image.save(save_path)

    mIoU = iou.compute()
    
    print(f"Refinement score with n_points={args.n_points} and box_factor={args.box_factor} is mIoU: {mIoU}")

if __name__ == '__main__':
    main()
