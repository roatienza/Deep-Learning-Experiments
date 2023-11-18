from ultralytics import YOLO
import torch
import cv2
import numpy as np

model = YOLO("checkpoints/yolov8x.pt")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model.to(device)
#results = model("../sam/images/dog_car.jpg")
image = cv2.imread("../sam/images/dog_car.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(type(image), np.amax(image))
results = model(image)
print(type(results))
for i, item in enumerate(results):
    print(i, type(item))
    print(item.boxes.data)
    for data in item.boxes.data:
        print(data)
        data = data.detach().cpu().numpy()
        idx = int(data[5])
        prob = data[4]
        bbox = data[:4]
        name = item.names[idx]
        print(idx , name, bbox)
