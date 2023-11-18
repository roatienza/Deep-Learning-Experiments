'''
PyTriton Yolo server

Usage:
    python3 server.py

Author:
    Rowel Atienza
    rowel@eee.upd.edu.ph

'''

import torch
import numpy as np
import os
import logging
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")
logger = logging.getLogger("Yolov8x")

checkpoint = os.path.join("checkpoints", "yolov8x.pt")
yolov8x = YOLO(checkpoint)

@batch
def infer_yolov8x(**image):
    image = image["image"][0]
    result = yolov8x(image)[0]
    bboxes = []
    probs = []
    names = ""
    for data in result.boxes.data:
        data = data.detach().cpu().numpy()
        idx = int(data[5])
        prob = data[4]
        bbox = data[:4]
        name = result.names[idx]
        bboxes.append(bbox)
        probs.append(prob)
        names += name + "|" 

    bboxes = np.array(bboxes, dtype=np.float32)
    probs = np.array(probs, dtype=np.float32)
    bboxes = np.expand_dims(bboxes, axis=0)
    probs = np.expand_dims(probs, axis=0)
    names = np.frombuffer(names.encode('utf-32'), dtype=np.uint32)
    names = np.expand_dims(names, axis=0)
    return { "bboxes": bboxes, "probs": probs, "names" : names }


# Connecting inference callback with Triton Inference Server
config = TritonConfig(http_port=8000, grpc_port=8001, metrics_port=8002,)
with Triton(config=config) as triton:
    # Load model into Triton Inference Server
    logger.debug("Loading Yolov8x.")
    triton.bind(
        model_name="Yolov8x",
        infer_func=infer_yolov8x,
        inputs=[
            Tensor(name="image", dtype=np.uint8, shape=(-1,-1,3)),
        ],
        outputs=[
            Tensor(name="bboxes", dtype=np.float32, shape=(-1,4)),
            Tensor(name="probs", dtype=np.float32, shape=(-1,1)),
            Tensor(name="names", dtype=np.uint32, shape=(-1,-1)),
        ],
        config=ModelConfig(max_batch_size=1)
    )
    triton.serve()
