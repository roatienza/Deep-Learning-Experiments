# Open CLIP and CoCa

OpenAI's [CLIP](https://github.com/OpenAI/CLIP) is useful for zero-shot image classification and image to text generation but not free. In  this example, ML Foundation's [OpenClip](https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/model.py) is used. OpenCLIP has a performance that is comparable to CLIP.

This example also shows how to serve [CoCa](https://arxiv.org/ba/2205.01917).

### Install

Install `open_clip` and its dependencies:

```
pip install open_clip_torch --upgrade
pip install torch --upgrade
pip install torchvision --upgrade
```

```
cd triton/openclip
```

In this example, Open CLIP that is trained on LAION 2B using ViT-B-32 is used.

### Run the server

```
python server.py
```

The default http, grpc and metrics ports can be changed by modifying:

```
config = TritonConfig(http_port=8000, grpc_port=8001, metrics_port=8002)
```

### Run the client

Open a new terminal, then run:

```
python client.py
```

The result:

```
2023-05-15 13:32:09,751 - INFO - OpenClip & CoCa: Running inference requests
index [284] (1,)
Siamese cat, Siamese
2023-05-15 13:32:09,888 - INFO - OpenClip & CoCa: Running inference requests
an orange and white cat with a red collar .
```

The default http port can be changed through the url option:

```
--url http://localhost:8000
```
