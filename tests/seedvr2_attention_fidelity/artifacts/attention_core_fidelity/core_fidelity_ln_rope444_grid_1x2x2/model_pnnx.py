# pnnx model stat
# model inputshape = [4,64,32]f32,[4,64,64]f32,[64,4]f32,[64,4]f32
# FLOPS = 4.882M
# memory OPS = 621.216K

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
        self.output = nn.Linear(bias=True, in_features=32, out_features=32)

        archive = zipfile.ZipFile('model.pnnx.bin', 'r')
        self.qkv.bias = self.load_pnnx_bin_as_parameter(archive, 'qkv.bias', (96), 'float32')
        self.qkv.weight = self.load_pnnx_bin_as_parameter(archive, 'qkv.weight', (96,32), 'float32')
        self.norm_q.bias = self.load_pnnx_bin_as_parameter(archive, 'norm_q.bias', (8), 'float32')
        self.norm_q.weight = self.load_pnnx_bin_as_parameter(archive, 'norm_q.weight', (8), 'float32')
        self.norm_k.bias = self.load_pnnx_bin_as_parameter(archive, 'norm_k.bias', (8), 'float32')
        self.norm_k.weight = self.load_pnnx_bin_as_parameter(archive, 'norm_k.weight', (8), 'float32')
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

    def forward(self, v_0, v_1, v_2, v_3):
        v_4 = v_0.reshape(256, 32)
        v_5 = self.qkv(v_4)
        v_6 = v_5.reshape(4, 64, 3, 4, 8)
        v_7, v_8, v_9 = torch.unbind(v_6, dim=2)
        v_10 = self.norm_q(v_7)
        v_11 = self.norm_k(v_8)
        v_12 = v_2.repeat(4, 1)
        v_13 = v_3.repeat(4, 1)
        v_14 = v_10.reshape(256, 4, 8)
        v_15 = v_14.reshape(256, 4, 4, 2)
        v_16 = v_12.unsqueeze(1)
        v_17 = v_13.unsqueeze(1)
        v_18, v_19 = torch.unbind(v_15, dim=3)
        v_20 = ((v_18 * v_16) - (v_19 * v_17))
        v_21 = ((v_18 * v_17) + (v_19 * v_16))
        v_22 = torch.stack((v_20, v_21), dim=-1)
        v_23 = v_22.reshape_as(other=v_14, )
        v_24 = v_23.reshape_as(other=v_10, )
        v_25 = v_11.reshape(256, 4, 8)
        v_26 = v_25.reshape(256, 4, 4, 2)
        v_27, v_28 = torch.unbind(v_26, dim=3)
        v_29 = ((v_27 * v_16) - (v_28 * v_17))
        v_30 = ((v_27 * v_17) + (v_28 * v_16))
        v_31 = torch.stack((v_29, v_30), dim=-1)
        v_32 = v_31.reshape_as(other=v_25, )
        v_33 = v_32.reshape_as(other=v_11, )
        v_34 = v_24.permute(dims=(0,2,1,3))
        v_35 = v_33.permute(dims=(0,2,3,1))
        v_36 = torch.matmul(v_34, other=v_35)
        v_37 = v_1.unsqueeze(1)
        v_38 = ((v_36 * 0.353553385) + v_37)
        v_39 = F.softmax(v_38, dim=-1)
        v_40 = v_9.permute(dims=(0,2,1,3))
        v_41 = torch.matmul(v_39, other=v_40)
        v_42 = v_41.permute(dims=(0,2,1,3))
        v_43 = v_42.reshape(256, 32)
        v_44 = self.output(v_43)
        v_45 = v_44.reshape_as(other=v_0, )
        return v_45

def export_torchscript():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(4, 64, 32, dtype=torch.float)
    v_1 = torch.rand(4, 64, 64, dtype=torch.float)
    v_2 = torch.rand(64, 4, dtype=torch.float)
    v_3 = torch.rand(64, 4, dtype=torch.float)

    mod = torch.jit.trace(net, (v_0, v_1, v_2, v_3))
    mod.save("model_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(4, 64, 32, dtype=torch.float)
    v_1 = torch.rand(4, 64, 64, dtype=torch.float)
    v_2 = torch.rand(64, 4, dtype=torch.float)
    v_3 = torch.rand(64, 4, dtype=torch.float)

    torch.onnx.export(net, (v_0, v_1, v_2, v_3), "model_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0', 'in1', 'in2', 'in3'], output_names=['out0'])

def export_pnnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(4, 64, 32, dtype=torch.float)
    v_1 = torch.rand(4, 64, 64, dtype=torch.float)
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
    v_0 = torch.rand(4, 64, 32, dtype=torch.float)
    v_1 = torch.rand(4, 64, 64, dtype=torch.float)
    v_2 = torch.rand(64, 4, dtype=torch.float)
    v_3 = torch.rand(64, 4, dtype=torch.float)

    return net(v_0, v_1, v_2, v_3)

if __name__ == "__main__":
    print(test_inference())
