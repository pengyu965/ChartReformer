from skimage import io
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
from utils import resize_and_pad
from PIL import Image
import os
from tqdm import tqdm


def SSIM_score(image1, image2):
    # Ensure both images are of the same shape
    assert image1.shape == image2.shape, "Images must have the same dimensions and channels"

    # Initialize a list to hold SSIM values for each channel
    ssim_values = []

    # Loop through each channel (R, G, B)
    for i in range(3):
        channel1 = image1[:,:,i].astype(np.float64)
        channel2 = image2[:,:,i].astype(np.float64)
        ssim_index, _ = compare_ssim(channel1, channel2, data_range=255, full=True)
        ssim_values.append(ssim_index)

    # Compute the average SSIM across all channels
    average_ssim = np.mean(ssim_values)
    return average_ssim

def main():
    vis_path = "./visualizations/finetune_fixed_data_laterwork/"
    ground_truth_path = "../../data/fixed_data_laterwork/test/edited_images/"

    category_dict = {}

    chartid_list = []
    for file in tqdm(os.listdir(vis_path)):
        if file.endswith(".txt"):
            chartid_list.append(file[:-4])

    for chartid in tqdm(chartid_list):
        category = chartid.split("_")[0]
        image1 = resize_and_pad(Image.open(ground_truth_path + chartid + ".jpg").convert('RGB'),(800,800))
        image2 = resize_and_pad(Image.open(vis_path + chartid + "_edited.jpg").convert('RGB'),(800,800))

        ssim = SSIM_score(np.array(image1), np.array(image2))

        if category not in category_dict:
            category_dict[category] = []
        category_dict[category].append(ssim)

    for category in category_dict:
        print("Category: ", category)
        print("Average SSIM: ", np.mean(category_dict[category]))

if __name__ == "__main__":
    main()