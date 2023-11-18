'''
PyTriton OpenClip server

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
import open_clip
import urllib
import pathlib
from PIL import Image
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")
logger = logging.getLogger("OpenClip and Cocalogger")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

openclip_b32, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
openclip_b32.to(device)

filename = "imagenet1000_labels.txt"
url = "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"

# Download the file if it does not exist
if not os.path.isfile(filename):
    urllib.request.urlretrieve(url, filename)

with open(filename) as f:
    idx2label = eval(f.read())

imagenet_labels = list(idx2label.values())
text = tokenizer(imagenet_labels)
text = text.to(device)

coca_l14, _, transform = open_clip.create_model_and_transforms(
  model_name="coca_ViT-L-14",
  pretrained="mscoco_finetuned_laion2B-s13B-b90k"
)

coca_l14.to(device)


@batch
def infer_openclip_b32(**image):
    image = image["image"][0]
    image = Image.fromarray(image)
    image = preprocess(image).unsqueeze(0)
    image = image.to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = openclip_b32.encode_image(image)
        text_features = openclip_b32.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    index = np.argmax(text_probs.cpu().numpy())
    label = imagenet_labels[index]

    index = np.array([index]).astype(np.int32)
    index = np.expand_dims(index, axis=0)

    label = np.frombuffer(label.encode('utf-32'), dtype=np.uint32)
    label = np.expand_dims(label, axis=0)

    return { "index": index , "label": label }

@batch
def infer_coca_l14(**image):
    image = image["image"][0]
    image = Image.fromarray(image)
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0)
    image = image.to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        generated = coca_l14.generate(image)

    label = open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")

    label = np.frombuffer(label.encode('utf-32'), dtype=np.uint32)
    label = np.expand_dims(label, axis=0)

    return { "label": label }


# Connecting inference callback with Triton Inference Server
model_repo = pathlib.Path("/data/triton/models")
config = TritonConfig(http_port=8010, grpc_port=8011, metrics_port=8012, model_repository=model_repo)
with Triton(config=config) as triton:
    # Load model into Triton Inference Server
    logger.debug("Loading OpenClip.")
    triton.bind(
        model_name="OpenClip_b32",
        infer_func=infer_openclip_b32,
        inputs=[
            Tensor(name="image", dtype=np.uint8, shape=(-1,-1,3)),
        ],
        outputs=[
            Tensor(name="index", dtype=np.int32, shape=(-1,)),
            Tensor(name="label", dtype=np.uint32, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=1)
    )
    logger.debug("Loading Coca.")
    triton.bind(
        model_name="CoCa_l14",
        infer_func=infer_coca_l14,
        inputs=[
            Tensor(name="image", dtype=np.uint8, shape=(-1,-1,3)),
        ],
        outputs=[
            Tensor(name="label", dtype=np.uint32, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=1)
    )
    triton.serve()
