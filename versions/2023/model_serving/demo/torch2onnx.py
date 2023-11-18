
'''
Sample code demonstrating:
- ONNX export of a PyTorch model (pre-trained ResNet50)

Requirements:
pip install onnx
pip install onnxruntime-gpu
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
from torchvision.models import resnet50, ResNet50_Weights
import onnx

# Create a dummy input tensor
dummy_input = torch.randn(1, 3, 224, 224, device="cuda")
# Load the model
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).cuda()
model.eval()

# input name - optional
input_names = [ "input1" ]
# output name - optional
output_names = [ "output1" ]

torch.onnx.export(model, dummy_input, "resnet50.onnx", 
                  verbose=True, input_names=input_names, 
                  output_names=output_names)


# Load the ONNX model
model = onnx.load("resnet50.onnx")

# Check that the model is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))