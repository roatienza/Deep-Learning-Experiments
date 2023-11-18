'''
PyTriton SAM server

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
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")
logger = logging.getLogger("SAM logger")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sam_checkpoint_h = os.path.join("checkpoints", "sam_vit_h_4b8939.pth")
model_type_h = "vit_h"
sam_h = sam_model_registry[model_type_h](checkpoint=sam_checkpoint_h)
sam_h.to(device=device)
sam_masks_gen_h = SamAutomaticMaskGenerator(sam_h)

sam_checkpoint_l = os.path.join("checkpoints", "sam_vit_l_0b3195.pth")
model_type_l = "vit_l"
sam_l = sam_model_registry[model_type_l](checkpoint=sam_checkpoint_l)
sam_l.to(device=device)
sam_masks_gen_l = SamAutomaticMaskGenerator(sam_l)

sam_checkpoint_b = os.path.join("checkpoints", "sam_vit_b_01ec64.pth")
model_type_b = "vit_b"
sam_b = sam_model_registry[model_type_b](checkpoint=sam_checkpoint_b)
sam_b.to(device=device)
sam_masks_gen_b = SamAutomaticMaskGenerator(sam_b)

@batch
def infer_sam_masks_h(**image):
    image = image["image"][0]
    logger.debug(f"Image data: {image.shape} ({image.size})") 
    masks = sam_masks_gen_h.generate(image)
    return format_outputs(masks)

@batch
def infer_sam_masks_l(**image):
    image = image["image"][0]
    logger.debug(f"Image data: {image.shape} ({image.size})") 
    masks = sam_masks_gen_l.generate(image)
    return format_outputs(masks)

@batch
def infer_sam_masks_b(**image):
    image = image["image"][0]
    logger.debug(f"Image data: {image.shape} ({image.size})") 
    masks = sam_masks_gen_b.generate(image)
    return format_outputs(masks)

def format_outputs(masks):
    outputs = {}
    outputs["segmentation"] = []
    outputs["area"] = []
    outputs["bbox"] = []
    outputs["predicted_iou"] = []
    outputs["stability_score"] = []
    for i, mask in enumerate(masks):
        for k,v in mask.items():
            if k in outputs:
                outputs[k].append(v)

    segmentation = np.array(outputs["segmentation"]).astype(np.bool_)
    segmentation = np.expand_dims(segmentation, axis=0)

    bbox = np.array(outputs["bbox"]).astype(np.intc)
    bbox = np.expand_dims(bbox, axis=0)
    
    predicted_iou = np.array(outputs["predicted_iou"]).astype(np.float32)
    predicted_iou = np.expand_dims(predicted_iou, axis=0)

    stability_score = np.array(outputs["stability_score"]).astype(np.float32)
    stability_score = np.expand_dims(stability_score, axis=0)

    area = np.array(outputs["area"]).astype(np.intc)
    area = np.expand_dims(area, axis=0)
    
    return { "segmentation" : segmentation, "area" : area, "bbox" : bbox, "predicted_iou" : predicted_iou, "stability_score" : stability_score  }


# Connecting inference callback with Triton Inference Server
config = TritonConfig(http_port=8010, grpc_port=8011, metrics_port=8012,)
with Triton(config=config) as triton:
    # Load model into Triton Inference Server
    logger.debug("Loading SAM_h.")
    triton.bind(
        model_name="SAM_h",
        infer_func=infer_sam_masks_h,
        inputs=[
            Tensor(name="image", dtype=np.uint8, shape=(-1,-1,3)),
        ],
        outputs=[
            Tensor(name="segmentation", dtype=np.bool_, shape=(-1,-1,-1)),
            Tensor(name="area", dtype=np.intc, shape=(-1,)),
            Tensor(name="bbox", dtype=np.intc, shape=(-1,4)),
            Tensor(name="predicted_iou", dtype=np.float32, shape=(-1,)),
            Tensor(name="stability_score", dtype=np.float32, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=1)
    )
    triton.bind(
        model_name="SAM_l",
        infer_func=infer_sam_masks_l,
        inputs=[
            Tensor(name="image", dtype=np.uint8, shape=(-1,-1,3)),
        ],
        outputs=[
            Tensor(name="segmentation", dtype=np.bool_, shape=(-1,-1,-1)),
            Tensor(name="area", dtype=np.intc, shape=(-1,)),
            Tensor(name="bbox", dtype=np.intc, shape=(-1,4)),
            Tensor(name="predicted_iou", dtype=np.float32, shape=(-1,)),
            Tensor(name="stability_score", dtype=np.float32, shape=(-1,)), 
        ],
        config=ModelConfig(max_batch_size=1)
    )
    triton.bind(
        model_name="SAM_b",
        infer_func=infer_sam_masks_b,
        inputs=[
            Tensor(name="image", dtype=np.uint8, shape=(-1,-1,3)),
        ],
        outputs=[
            Tensor(name="segmentation", dtype=np.bool_, shape=(-1,-1,-1)),
            Tensor(name="area", dtype=np.intc, shape=(-1,)),
            Tensor(name="bbox", dtype=np.intc, shape=(-1,4)),
            Tensor(name="predicted_iou", dtype=np.float32, shape=(-1,)),
            Tensor(name="stability_score", dtype=np.float32, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=1)
    )
    triton.serve()
