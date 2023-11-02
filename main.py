from utils import *
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from IPython.display import Image, display
from PIL import Image
import matplotlib.pyplot as plt
import argparse

def main(class_choice, img_path, model_name):


    PATH_OF_DATA = "data/"

    elephant_folder = "African_elephant/"
    bear_folder = "black_bear/"

    if class_choice == "bear":

        class_folder = bear_folder
        target_cls = 295

    elif class_choice == "elephant":
        target_cls = 386

        class_folder = elephant_folder
    else:
        raise NotImplementedError("Local dataset allows only bear and elephant arguments..")



    img_path = PATH_OF_DATA + class_folder + img_path
    img = Image.open(img_path)

    if model_name == "resnet":
        feature_dim = 2048
    elif model_name == "alexnet":
        feature_dim = 256
    else:
        raise NotImplementedError("For the time of the push, there is only resnet50 and alexnet models available.")

    model = get_model(model_name)
    transform = get_transforms(model_name)

    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)


    gradients = get_gradients(model_name, batch_t,target_cls)
    feature_maps = get_feature_maps(model,batch_t)

    # Expand tensor1 to match the shape of tensor2
    gradients_expanded = gradients.view(1, feature_dim, 1, 1)

    # Perform element-wise multiplication
    weigted_feature_maps = gradients_expanded * feature_maps

    # Sum along the second dimension (2048)
    weigted_feature_maps = weigted_feature_maps.sum(dim=1)
    weigted_feature_maps = weigted_feature_maps.unsqueeze(0)

    weigted_feature_maps = F.interpolate(weigted_feature_maps, size=(img.size[1], img.size[0]), mode='bilinear', align_corners=False)
    weigted_feature_maps = F.relu(weigted_feature_maps)
    weigted_feature_maps = weigted_feature_maps.squeeze(0).squeeze(0)

    # Normalize the grayscale heatmap to be in the range [0, 1]
    heatmap_tensor = (weigted_feature_maps - weigted_feature_maps.min()) / (weigted_feature_maps.max() - weigted_feature_maps.min())

    # Convert the heatmap to a colormap image
    heatmap_colormap = plt.get_cmap('jet')(heatmap_tensor)


    # Convert the PIL image to a NumPy array
    pil_array = np.array(img)

    # Blend the images
    alpha = 0.7  # Adjust the blending strength
    blended_image = (1 - alpha) * pil_array + alpha * heatmap_colormap[:, :, :3] * 255

    # Ensure values are in the valid range for displaying as an RGB image
    blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)

    # Convert the blended NumPy array to a PIL image
    result_pil_image = Image.fromarray(blended_image)

    # Save the final blended image
    result_pil_image.save('result.png')


if __name__ == "__main__":
    # Define command-line arguments with default values
    parser = argparse.ArgumentParser(description='Image Heatmap Generator')
    parser.add_argument('--class_choice', type=str, default="elephant", help='Choose the class ("bear" or "elephant")')
    parser.add_argument('--img_path', type=str, default="ILSVRC2012_val_00025941.JPEG", help='Path to the image file')
    parser.add_argument('--model_name', type=str, default="resnet", help='Choose the model ("resnet" or "alexnet")')

    args = parser.parse_args()

    # Call the main function with user-specified or default values
    main(args.class_choice, args.img_path, args.model_name)
