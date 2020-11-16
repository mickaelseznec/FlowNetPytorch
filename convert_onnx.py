import argparse
import onnx
import torch
import flow_transforms
import torch.onnx as torch_onnx
import torchvision.transforms as transforms
import models
import numpy as np

from imageio import imread, imwrite
from torch.autograd import Variable
from onnx import numpy_helper

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("input_image_1")
    parser.add_argument("input_image_2")
    args = parser.parse_args()

    # Data loading code
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
        transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
    ])

    input_h, input_w, _ = np.asarray(imread(args.input_image_1)).shape

    model_raw = torch.load(args.model_path)
    model_arch = model_raw['arch']
    model = models.__dict__[model_arch](model_raw)
    model.train(False)

    model_onnx_path = model_arch + ".onnx"
    input = torch.cat([input_transform(imread(args.input_image_1)),
                       input_transform(imread(args.input_image_2))]).unsqueeze(0)

    with open("input.pb", "wb") as f:
        f.write(numpy_helper.from_array(input.cpu().numpy()).SerializeToString())

    inputs = ['input']
    outputs = ['output']
    dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0:'batch_size'}}
    out = torch.onnx.export(model,
                            input,
                            model_onnx_path,
                            opset_version=10,
                            input_names=inputs,
                            output_names=outputs,
                            dynamic_axes=dynamic_axes)

    output = model(input)

    with open("output.pb", "wb") as f:
        f.write(numpy_helper.from_array(output.cpu().numpy()).SerializeToString())

if __name__=='__main__':
    main()
