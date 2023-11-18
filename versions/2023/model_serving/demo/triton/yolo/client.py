"""
OpenClip Client

Usage: 
    python3 client --image <path>

Author:
    Rowel Atienza rowel@eee.upd.edu.ph

"""
import argparse
import logging
import cv2
import numpy as np
import urllib.request
import validators

from pytriton.client import ModelClient

logger = logging.getLogger("Yolov8x")

def infer_model(args):
    with ModelClient(args.url, args.model, init_timeout_s=args.init_timeout_s) as client:
        if validators.url(args.image):
            with urllib.request.urlopen(args.image) as url_response:
                img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
                image = cv2.imdecode(img_array, -1)
        else:
            image = cv2.imread(args.image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        logger.info(f"Running inference requests")
        outputs = client.infer_sample(image)
        for k, v in outputs.items():
            if k == "names":
                names = v.tobytes().decode('utf-32').split("|")
                print(names[:-1])
            else:
                print(k, v, v.shape)
                

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        default="../sam/images/dog_car.jpg",
        help=(
            "Path to image can filesystem path or url path"
        ),
    )
    choices = ["Yolov8x"]
    parser.add_argument(
        "--model",
        default=choices[0],
        choices=choices,
        help=(
            "Yolo8 model" 
        ),
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help=(
            "Url to Triton server (ex. grpc://localhost:8001)."
            "HTTP protocol with default port is used if parameter is not provided"
        ),
        required=False,
    )
    parser.add_argument(
        "--init-timeout-s",
        type=float,
        default=600.0,
        help="Server and model ready state timeout in seconds",
        required=False,
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="Number of concurrent requests.",
        required=False,
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of requests per client.",
        required=False,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

    infer_model(args)


if __name__ == "__main__":
    main()
