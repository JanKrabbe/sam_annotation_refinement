
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from torchvision.datasets import Cityscapes

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize refined annotation.")

    parser.add_argument("--data_root", type=str,
                        help="Path to the Cityscapes dataset")
    
    parser.add_argument("--id", type=int,
                        default=0,
                        help="Id of the image to visualize in the Cityscapes Dataset.")
    
    parser.add_argument("--no_overlay", action="store_false", help="Plot without background image.")
     
    args = parser.parse_args()

    return args

def create_plot(image, target, file_name):
    plt.figure()
    plt.axis("off")
    if image is not None:
        plt.imshow(image)
        plt.imshow(target, alpha=(np.array(target) != 0)*0.5)
    else:
        plt.imshow(target)

    plt.savefig(file_name, bbox_inches='tight', pad_inches=0, dpi=413)

def main():
    args = parse_args()

    dataset_coarse = Cityscapes(args.data_root, split='train', mode='coarse', target_type='semantic')
    img, target_coarse = dataset_coarse[args.id]

    target_path = dataset_coarse.targets[args.id][0]
    target_refined = Image.open(target_path.replace("_gtCoarse", "_gtRefined"))

    image_plot = None
    if args.no_overlay:
        image_plot = img

    if not os.path.isdir("plots"):
        os.makedirs(os.path.join(os.getcwd(),"plots"))

    create_plot(image_plot, target_coarse, f"plots/{args.id}_coarse.png")
    create_plot(image_plot, target_refined, f"plots/{args.id}_refined.png")

if __name__ == '__main__':
    main()