'''
Sample code demonstrating:
- TorchScript tracing and scripting
- TensorRT compilation

Requirements:
conda install cudatoolkit
pip install --upgrade setuptools pip
pip install nvidia-pyindex
pip install --upgrade nvidia-tensorrt
sudo apt-get install python3-libnvinfer-dev

Rowel Atienza
github.com/roatienza
2023
'''

import torch
from PIL import Image
import numpy as np
import torch_tensorrt
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
import urllib
import os

# Download the ImageNet1k label file
filename = "imagenet1000_labels.txt"
url = "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"

# Download the file if it does not exist
if not os.path.isfile(filename):
    urllib.request.urlretrieve(url, filename)

with open(filename) as f:
    idx2label = eval(f.read())
        
# Load the ResNet50 pre-trained model
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).cuda()
model.eval()

# Load the sample image
image = Image.open("wonder_cat.jpg")

# Define the transformation
transform = transforms.Compose([
    transforms.ToTensor()
])

# Apply the transformation to the image
input_tensor = transform(image).unsqueeze(0).cuda()

# Print the shape of the input tensor
print(input_tensor.shape)

# Generate the traced TorchScript module
traced_model = torch.jit.trace(model, example_inputs=torch.randn(1, 3, 224, 224).cuda())
# Save the traced TorchScript module
traced_model.save("traced_model.pt")
# Test prediction of the traced model
outputs = traced_model(input_tensor)
print(outputs.shape)
argmax_output = torch.argmax(outputs, dim=1).cpu().numpy()[0]
print("Traced model label index:", argmax_output)
print("Traced model label:", idx2label[argmax_output])

# Generate the scripted TorchScript module
scripted_model = torch.jit.script(model)
# Save the scripted TorchScript module
scripted_model.save("scripted_model.pt")
# Test prediction of the scripted model
outputs = scripted_model(input_tensor)
argmax_output = torch.argmax(outputs, dim=1).cpu().numpy()[0]
print("Scripted model label index:", argmax_output)
print("Scripted model label:", idx2label[argmax_output])

# Compile the traced TorchScript module using TensorRT (trt)
compiled_model = torch_tensorrt.compile(traced_model, 
                                        inputs = [input_tensor],
                                        truncate_long_and_double = True,)

# Save the compiled trt model
compiled_model.save("trt_model.pt")
outputs = compiled_model(input_tensor).cpu().numpy()
print(outputs.shape)
# Test prediction of the compiled trt model
argmax_output = np.argmax(outputs)
print("TensorRT label index:", argmax_output)
print("TensorRT label:", idx2label[argmax_output])