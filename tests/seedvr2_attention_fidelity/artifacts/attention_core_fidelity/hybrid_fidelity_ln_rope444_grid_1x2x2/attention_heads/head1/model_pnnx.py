# pnnx model stat
# model inputshape = [64,32]f32,[64,64]f32,[64,4]f32,[64,4]f32
# FLOPS = 571.392K
# memory OPS = 52.352K

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
        self.ln_0 = nn.LayerNorm(elementwise_affine=True, eps=1.000000e-5, normalized_shape=(8,))
        self.ln_1 = nn.LayerNorm(elementwise_affine=True, eps=1.000000e-5, normalized_shape=(8,))

        archive = zipfile.ZipFile('model.pnnx.bin', 'r')
        self.qkv.bias = self.load_pnnx_bin_as_parameter(archive, 'qkv.bias', (96), 'float32')
        self.qkv.weight = self.load_pnnx_bin_as_parameter(archive, 'qkv.weight', (96,32), 'float32')
        self.ln_0.bias = self.load_pnnx_bin_as_parameter(archive, 'ln_0.bias', (8), 'float32')
        self.ln_0.weight = self.load_pnnx_bin_as_parameter(archive, 'ln_0.weight', (8), 'float32')
        self.ln_1.bias = self.load_pnnx_bin_as_parameter(archive, 'ln_1.bias', (8), 'float32')
        self.ln_1.weight = self.load_pnnx_bin_as_parameter(archive, 'ln_1.weight', (8), 'float32')
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

    def forward(self, v_0, v_1, v_2, v_3):
        v_4 = self.qkv(v_0)
        v_5 = v_4.reshape(1, 64, 96)
        v_6, v_7, v_8 = torch.split(tensor=v_5, dim=-1, split_size_or_sections=32)
        v_9 = v_6[:,:,8:16]
        v_10 = v_7[:,:,8:16]
        v_11 = v_8[:,:,8:16]
        v_12 = self.ln_0(v_9)
        v_13 = self.ln_1(v_10)
        v_14 = v_2.repeat(1, 1)
        v_15 = v_3.repeat(1, 1)
        v_16 = v_12.reshape(64, 1, 8)
        v_17 = v_16.reshape(64, 1, 4, 2)
        v_18 = v_14.unsqueeze(1)
        v_19 = v_15.unsqueeze(1)
        v_20, v_21 = torch.unbind(v_17, dim=3)
        v_22 = ((v_20 * v_18) - (v_21 * v_19))
        v_23 = ((v_20 * v_19) + (v_21 * v_18))
        v_24 = torch.stack((v_22, v_23), dim=-1)
        v_25 = v_24.reshape_as(other=v_16, )
        v_26 = v_25.reshape_as(other=v_12, )
        v_27 = v_13.reshape(64, 1, 8)
        v_28 = v_27.reshape(64, 1, 4, 2)
        v_29, v_30 = torch.unbind(v_28, dim=3)
        v_31 = ((v_29 * v_18) - (v_30 * v_19))
        v_32 = ((v_29 * v_19) + (v_30 * v_18))
        v_33 = torch.stack((v_31, v_32), dim=-1)
        v_34 = v_33.reshape_as(other=v_27, )
        v_35 = v_34.reshape_as(other=v_13, )
        v_36 = torch.transpose(v_35, dim0=-2, dim1=-1)
        v_37 = torch.matmul(v_26, other=v_36)
        v_38 = ((v_37 * 0.353553385) + v_1)
        v_39 = F.softmax(v_38, dim=-1)
        v_40 = torch.matmul(v_39, other=v_11)
        return v_40

def export_torchscript():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(64, 32, dtype=torch.float)
    v_1 = torch.rand(64, 64, dtype=torch.float)
    v_2 = torch.rand(64, 4, dtype=torch.float)
    v_3 = torch.rand(64, 4, dtype=torch.float)

    mod = torch.jit.trace(net, (v_0, v_1, v_2, v_3))
    mod.save("model_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(64, 32, dtype=torch.float)
    v_1 = torch.rand(64, 64, dtype=torch.float)
    v_2 = torch.rand(64, 4, dtype=torch.float)
    v_3 = torch.rand(64, 4, dtype=torch.float)

    torch.onnx.export(net, (v_0, v_1, v_2, v_3), "model_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0', 'in1', 'in2', 'in3'], output_names=['out0'])

def export_pnnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(64, 32, dtype=torch.float)
    v_1 = torch.rand(64, 64, dtype=torch.float)
    v_2 = torch.rand(64, 4, dtype=torch.float)
    v_3 = torch.rand(64, 4, dtype=torch.float)

    import pnnx
    pnnx.export(net, "model_pnnx.py.pt", (v_0, v_1, v_2, v_3))

def export_ncnn():
    export_pnnx()

@torch.no_grad()
def test_inference():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(64, 32, dtype=torch.float)
    v_1 = torch.rand(64, 64, dtype=torch.float)
    v_2 = torch.rand(64, 4, dtype=torch.float)
    v_3 = torch.rand(64, 4, dtype=torch.float)

    return net(v_0, v_1, v_2, v_3)

if __name__ == "__main__":
    print(test_inference())
