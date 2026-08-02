# pnnx model stat
# model inputshape = [16,48,32]f32,[16,48,48]f32
# FLOPS = 12.141M
# memory OPS = 1.331M

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

        archive = zipfile.ZipFile('core_344_grid_1x4x4.pnnx.bin', 'r')
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

    def forward(self, v_0, v_1):
        v_2 = v_0.reshape(768, 32)
        v_3 = self.qkv(v_2)
        v_4 = v_3.reshape(16, 48, 96)
        v_5, v_6, v_7 = torch.split(tensor=v_4, dim=-1, split_size_or_sections=32)
        v_8, v_9, v_10, v_11 = torch.split(tensor=v_5, dim=-1, split_size_or_sections=8)
        v_12, v_13, v_14, v_15 = torch.split(tensor=v_6, dim=-1, split_size_or_sections=8)
        v_16, v_17, v_18, v_19 = torch.split(tensor=v_7, dim=-1, split_size_or_sections=8)
        v_20 = torch.transpose(v_12, dim0=-2, dim1=-1)
        v_21 = torch.matmul(v_8, other=v_20)
        v_22 = ((v_21 * 0.353553385) + v_1)
        v_23 = F.softmax(v_22, dim=-1)
        v_24 = torch.matmul(v_23, other=v_16)
        v_25 = torch.transpose(v_13, dim0=-2, dim1=-1)
        v_26 = torch.matmul(v_9, other=v_25)
        v_27 = ((v_26 * 0.353553385) + v_1)
        v_28 = F.softmax(v_27, dim=-1)
        v_29 = torch.matmul(v_28, other=v_17)
        v_30 = torch.transpose(v_14, dim0=-2, dim1=-1)
        v_31 = torch.matmul(v_10, other=v_30)
        v_32 = ((v_31 * 0.353553385) + v_1)
        v_33 = F.softmax(v_32, dim=-1)
        v_34 = torch.matmul(v_33, other=v_18)
        v_35 = torch.transpose(v_15, dim0=-2, dim1=-1)
        v_36 = torch.matmul(v_11, other=v_35)
        v_37 = ((v_36 * 0.353553385) + v_1)
        v_38 = F.softmax(v_37, dim=-1)
        v_39 = torch.matmul(v_38, other=v_19)
        v_40 = torch.cat((v_24, v_29, v_34, v_39), dim=-1)
        v_41 = v_40.reshape(768, 32)
        v_42 = self.output(v_41)
        v_43 = v_42.reshape_as(other=v_0, )
        return v_43

def export_torchscript():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(16, 48, 32, dtype=torch.float)
    v_1 = torch.rand(16, 48, 48, dtype=torch.float)

    mod = torch.jit.trace(net, (v_0, v_1))
    mod.save("core_344_grid_1x4x4_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(16, 48, 32, dtype=torch.float)
    v_1 = torch.rand(16, 48, 48, dtype=torch.float)

    torch.onnx.export(net, (v_0, v_1), "core_344_grid_1x4x4_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0', 'in1'], output_names=['out0'])

def export_pnnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(16, 48, 32, dtype=torch.float)
    v_1 = torch.rand(16, 48, 48, dtype=torch.float)

    import pnnx
    pnnx.export(net, "core_344_grid_1x4x4_pnnx.py.pt", (v_0, v_1))

def export_ncnn():
    export_pnnx()

@torch.no_grad()
def test_inference():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(16, 48, 32, dtype=torch.float)
    v_1 = torch.rand(16, 48, 48, dtype=torch.float)

    return net(v_0, v_1)

if __name__ == "__main__":
    print(test_inference())
