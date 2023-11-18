'''
Sample code demonstrating:
- ONNX inference using onnxruntime (CPU, CUDA, TensorRT)

Rowel Atienza
github.com/roatienza
2023
'''


import onnx
import onnxruntime 
import numpy as np
from PIL import Image
import numpy as np
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
    
# Load the ONNX model
model = onnx.load("resnet50.onnx")

# Check that the model is well formed
onnx.checker.check_model(model)

# List all the available providers
print("Available providers:", onnxruntime.get_available_providers())

# Load the image
image = Image.open("wonder_cat.jpg")

# Convert the image to a numpy array
# Note that ONNX expects numpy inputs with batch size and channel dimension first
image_array = np.array(image)
# Add batch dimension to image_array
image_array = np.expand_dims(image_array, axis=0)
# Permute the channel axis to the 2nd dimension
image_array = np.transpose(image_array, (0, 3, 1, 2)).astype(np.float32)
image_array /= 255.0

# Choose the device to run the model on
device = 'cpu'
if device == 'cpu':
    providers = ['CPUExecutionProvider']
elif device == 'cuda':
    providers = ['CUDAExecutionProvider']
else: # use all including tensorrt
    providers = onnxruntime.get_available_providers()

# Perform inference using onnxruntime
print("Using providers:", providers)
ort_session = onnxruntime.InferenceSession("resnet50.onnx", providers=providers,)
outputs = ort_session.run( None, {"input1": image_array},)[0]
print(outputs.shape)
argmax_output = np.argmax(outputs)
print("Class label index:", argmax_output)
print("Predicted label:", idx2label[argmax_output])


