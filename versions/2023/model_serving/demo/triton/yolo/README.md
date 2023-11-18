# Yolov8

[Yolov8](https://github.com/ultralytics/ultralytics) is a fast object detection model from [Ultralytics](https://docs.ultralytics.com/). 
In this example, a pytriton server/client example is illustrated.

### Install

Before using the example, install `ultralytics`:

```
pip install ultralytics
```

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
2023-05-16 09:23:12,311 - INFO - Yolov8x: Running inference requests
bboxes [[2.2844419e+00 1.4254788e+02 1.5097408e+03 1.2124508e+03]
 [5.9323590e+02 8.3716437e+02 1.1737468e+03 1.3993324e+03]
 [5.9113712e+01 2.4727133e-01 3.4345667e+02 1.4267696e+02]
 [1.4533229e+03 9.9240184e-01 1.5112069e+03 1.0208485e+02]] (4, 4)
probs [0.9795568 0.9494859 0.6166969 0.3918449] (4,)
['car', 'dog', 'car', 'chair']
```

### Client Notebook

The client is also available as a [Jupyter Notebook](https://github.com/roatienza/mlops/blob/main/triton/yolo/client.ipynb). Be sure to modify the `url` to point to your server ip. 
