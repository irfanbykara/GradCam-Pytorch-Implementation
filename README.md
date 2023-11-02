# GradCam-Pytorch-Implementation

This is a PyTorch project that implements Grad-CAM (Gradient-weighted Class Activation Mapping) from scratch. For the sake of simplicity, the project uses pretrained models from ImageNet, including AlexNet and ResNet50.

## Installation

To set up the project on your local machine, follow these steps:

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/your-username/GradCam-Pytorch-Implementation.git
    ```

2. Navigate to the project directory:

    ```bash
    cd GradCam-Pytorch-Implementation
    ```

3. Create a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use: venv\Scripts\activate
    ```

4. Install the required Python packages using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

To generate heatmaps for images using Grad-CAM, you can run the project with the following command-line arguments:

```bash
python main.py --class_choice <class_name> --img_path <image_path> --model_name <model_name>
Where:

class_choice: Choose the target class name (e.g., "elephant" or "bear").
img_path: Provide the path to the image file you want to visualize.
model_name: Choose the model for generating heatmaps ("resnet" or "alexnet").
By default, the script uses the following values:

class_choice: "elephant"
img_path: "ILSVRC2012_val_00025941.JPEG"
model_name: "resnet"
You can adjust these command-line arguments according to your specific use case.

