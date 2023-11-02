from torchvision import models
import torch
from torchvision import transforms
import numpy as np

def get_model(model_name):

    if model_name == "alexnet":
        model = models.alexnet(pretrained=True)
    elif model_name == "resnet":
        model = models.resnet50(pretrained=True)
    else:
        raise NotImplementedError("The model you are looking for is not implemented.")
    return model


def get_transforms(model_name):
    if model_name == "alexnet":

        # Define the input transformation
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    elif model_name == "resnet":
        transform = transforms.Compose([  # [1]
            transforms.Resize(256),  # [2]
            transforms.CenterCrop(224),  # [3]
            transforms.ToTensor(),  # [4]
            transforms.Normalize(  # [5]
                mean=[0.485, 0.456, 0.406],  # [6]
                std=[0.229, 0.224, 0.225]  # [7]
            )])
    else:
        raise NotImplementedError("The model you are looking for is not implemented.")

    return transform

# Create a hook to store gradients
gradients = None

def hook_fn(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

def get_gradients(model_name,batch,target_cls):

    # Register the hook to the last convolutional layer in AlexNet
    model = get_model(model_name)
    if model_name == "alexnet":

        last_conv_layer = model.features[-2]

    else:
        last_conv_layer = model.layer4[-1]

    hook = last_conv_layer.register_backward_hook(hook_fn)


    # Forward pass
    output = model(batch)

    pred_index = np.argmax(output.detach().numpy())  # we will explain for this specific class

    # Define a target class (the class for which you want to compute gradients)
    target_class = target_cls  # Replace with the desired class index


    # Calculate the loss as the negative log probability of the target class
    loss = output[0, target_class]

    # Backpropagation to compute gradients
    loss.backward()
    if model_name == "alexnet":
        sum_tensor = gradients.view(1, 256, -1).sum(dim=2).view(1, 256, 1)
        sum_tensor /= 289
    else:
        sum_tensor = gradients.view(1, 2048, -1).sum(dim=2).view(1, 2048, 1)
        sum_tensor /= 49


    # The gradients are now stored in the 'gradients' variable
    if gradients is  None:
        print("Gradients not available.")

    return sum_tensor

def get_feature_maps(model,batch_t):

    model_without_fc = torch.nn.Sequential(*list(model.children())[:-2])
    # Pass the image through the model to get the feature map
    with torch.no_grad():
        feature_map = model_without_fc(batch_t)
    return feature_map




