"""Which physical GPU does a Ray actor with one GPU get? (a8-rootcause RC-3: printed after the GPU order swap)"""
import os

import ray
import torch

ray.init(address="auto", logging_level="ERROR")


@ray.remote(num_gpus=1)
def probe():
    props = torch.cuda.get_device_properties(0)
    return {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "uuid": str(getattr(props, "uuid", "?"))}


print("gpu_map", ray.get(probe.remote()))
