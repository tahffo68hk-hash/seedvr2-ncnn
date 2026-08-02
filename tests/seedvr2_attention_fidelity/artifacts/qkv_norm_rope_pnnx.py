# pnnx model stat
# model inputshape = [4,64,32]f32,[64,4]f32
# FLOPS = 1.802M
# memory OPS = 164.224K

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
        self.norm_q = nn.LayerNorm(elementwise_affine=True, eps=1.000000e-5, normalized_shape=(8,))
        self.norm_k = nn.LayerNorm(elementwise_affine=True, eps=1.000000e-5, normalized_shape=(8,))

        archive = zipfile.ZipFile('qkv_norm_rope.pnnx.bin', 'r')
        self.qkv.bias = self.load_pnnx_bin_as_parameter(archive, 'qkv.bias', (96), 'float32')
        self.qkv.weight = self.load_pnnx_bin_as_parameter(archive, 'qkv.weight', (96,32), 'float32')
        self.norm_q.bias = self.load_pnnx_bin_as_parameter(archive, 'norm_q.bias', (8), 'float32')
        self.norm_q.weight = self.load_pnnx_bin_as_parameter(archive, 'norm_q.weight', (8), 'float32')
        self.norm_k.bias = self.load_pnnx_bin_as_parameter(archive, 'norm_k.bias', (8), 'float32')
        self.norm_k.weight = self.load_pnnx_bin_as_parameter(archive, 'norm_k.weight', (8), 'float32')
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
        v_2 = v_0.reshape(256, 32)
        v_3 = self.qkv(v_2)
        v_4 = v_3.reshape(4, 64, 3, 4, 8)
        v_5, v_6, v_7 = torch.unbind(v_4, dim=2)
        v_8 = self.norm_q(v_5)
        v_9 = self.norm_k(v_6)
        v_10 = v_1.repeat(4, 1)
        v_11 = v_8.reshape(256, 4, 8)
        v_12 = v_11.reshape(256, 4, 4, 2)
        v_13 = torch.cos(v_10)
        v_14 = v_13.unsqueeze(1)
        v_15 = torch.sin(v_10)
        v_16 = v_15.unsqueeze(1)
        v_17, v_18 = torch.unbind(v_12, dim=3)
        v_19 = ((v_17 * v_14) - (v_18 * v_16))
        v_20 = ((v_17 * v_16) + (v_18 * v_14))
        v_21 = torch.stack((v_19, v_20), dim=-1)
        v_22 = v_21.reshape_as(other=v_11, )
        v_23 = v_9.reshape(256, 4, 8)
        v_24 = v_23.reshape(256, 4, 4, 2)
        v_25, v_26 = torch.unbind(v_24, dim=3)
        v_27 = ((v_25 * v_14) - (v_26 * v_16))
        v_28 = ((v_25 * v_16) + (v_26 * v_14))
        v_29 = torch.stack((v_27, v_28), dim=-1)
        v_30 = v_29.reshape_as(other=v_23, )
        v_31 = v_7.reshape(256, 4, 8)
        v_32 = (v_22, v_30, v_31, )
        return v_32

def export_torchscript():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(4, 64, 32, dtype=torch.float)
    v_1 = torch.rand(64, 4, dtype=torch.float)

    mod = torch.jit.trace(net, (v_0, v_1))
    mod.save("qkv_norm_rope_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(4, 64, 32, dtype=torch.float)
    v_1 = torch.rand(64, 4, dtype=torch.float)

    torch.onnx.export(net, (v_0, v_1), "qkv_norm_rope_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0', 'in1'], output_names=['out0'])

def export_pnnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(4, 64, 32, dtype=torch.float)
    v_1 = torch.rand(64, 4, dtype=torch.float)

    import pnnx
    pnnx.export(net, "qkv_norm_rope_pnnx.py.pt", (v_0, v_1))

def export_ncnn():
    export_pnnx()

@torch.no_grad()
def test_inference():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(4, 64, 32, dtype=torch.float)
    v_1 = torch.rand(64, 4, dtype=torch.float)

    return net(v_0, v_1)

if __name__ == "__main__":
    print(test_inference())
