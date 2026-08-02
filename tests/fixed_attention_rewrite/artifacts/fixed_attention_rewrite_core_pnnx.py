# pnnx model stat
# model inputshape = [4,32,32]f32
# FLOPS = 1.688M
# memory OPS = 151.68K

import os
import numpy as np
import tempfile, zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchvision
    import torchaudio
except:
    pass

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.qkv = nn.Linear(bias=True, in_features=32, out_features=96)
        self.output = nn.Linear(bias=True, in_features=32, out_features=32)

        archive = zipfile.ZipFile('D:\AIProjects\seedvr2-ncnn\tests\fixed_attention_rewrite\artifacts\fixed_attention_rewrite_core.pnnx.bin', 'r')
        self.qkv.bias = self.load_pnnx_bin_as_parameter(archive, 'qkv.bias', (96), 'float32')
        self.qkv.weight = self.load_pnnx_bin_as_parameter(archive, 'qkv.weight', (96,32), 'float32')
        self.output.bias = self.load_pnnx_bin_as_parameter(archive, 'output.bias', (32), 'float32')
        self.output.weight = self.load_pnnx_bin_as_parameter(archive, 'output.weight', (32,32), 'float32')
        archive.close()

    def load_pnnx_bin_as_parameter(self, archive, key, shape, dtype, requires_grad=True):
        return nn.Parameter(self.load_pnnx_bin_as_tensor(archive, key, shape, dtype), requires_grad)

    def load_pnnx_bin_as_tensor(self, archive, key, shape, dtype):
        fd, tmppath = tempfile.mkstemp()
        with os.fdopen(fd, 'wb') as tmpf, archive.open(key) as keyfile:
            tmpf.write(keyfile.read())
        m = np.memmap(tmppath, dtype=dtype, mode='r', shape=shape).copy()
        os.remove(tmppath)
        return torch.from_numpy(m)

    def forward(self, v_0):
        v_1 = v_0.reshape(128, 32)
        v_2 = self.qkv(v_1)
        v_3 = v_2.reshape(4, 32, 96)
        v_4, v_5, v_6 = torch.split(tensor=v_3, dim=-1, split_size_or_sections=32)
        v_7, v_8, v_9, v_10 = torch.split(tensor=v_4, dim=-1, split_size_or_sections=8)
        v_11, v_12, v_13, v_14 = torch.split(tensor=v_5, dim=-1, split_size_or_sections=8)
        v_15, v_16, v_17, v_18 = torch.split(tensor=v_6, dim=-1, split_size_or_sections=8)
        v_19 = torch.transpose(v_11, dim0=-2, dim1=-1)
        v_20 = torch.matmul(v_7, other=v_19)
        v_21 = (v_20 * 0.353553385)
        v_22 = F.softmax(v_21, dim=-1)
        v_23 = torch.matmul(v_22, other=v_15)
        v_24 = torch.transpose(v_12, dim0=-2, dim1=-1)
        v_25 = torch.matmul(v_8, other=v_24)
        v_26 = (v_25 * 0.353553385)
        v_27 = F.softmax(v_26, dim=-1)
        v_28 = torch.matmul(v_27, other=v_16)
        v_29 = torch.transpose(v_13, dim0=-2, dim1=-1)
        v_30 = torch.matmul(v_9, other=v_29)
        v_31 = (v_30 * 0.353553385)
        v_32 = F.softmax(v_31, dim=-1)
        v_33 = torch.matmul(v_32, other=v_17)
        v_34 = torch.transpose(v_14, dim0=-2, dim1=-1)
        v_35 = torch.matmul(v_10, other=v_34)
        v_36 = (v_35 * 0.353553385)
        v_37 = F.softmax(v_36, dim=-1)
        v_38 = torch.matmul(v_37, other=v_18)
        v_39 = torch.cat((v_23, v_28, v_33, v_38), dim=-1)
        v_40 = v_39.reshape(128, 32)
        v_41 = self.output(v_40)
        v_42 = v_41.reshape_as(other=v_0, )
        return v_42

def export_torchscript():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(4, 32, 32, dtype=torch.float)

    mod = torch.jit.trace(net, v_0)
    mod.save("D:\AIProjects\seedvr2-ncnn\tests\fixed_attention_rewrite\artifacts\fixed_attention_rewrite_core_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(4, 32, 32, dtype=torch.float)

    torch.onnx.export(net, v_0, "D:\AIProjects\seedvr2-ncnn\tests\fixed_attention_rewrite\artifacts\fixed_attention_rewrite_core_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0'], output_names=['out0'])

def export_pnnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(4, 32, 32, dtype=torch.float)

    import pnnx
    pnnx.export(net, "D:\AIProjects\seedvr2-ncnn\tests\fixed_attention_rewrite\artifacts\fixed_attention_rewrite_core_pnnx.py.pt", v_0)

def export_ncnn():
    export_pnnx()

@torch.no_grad()
def test_inference():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(4, 32, 32, dtype=torch.float)

    return net(v_0)

if __name__ == "__main__":
    print(test_inference())
