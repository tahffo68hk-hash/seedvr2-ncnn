# pnnx model stat
# model inputshape = [4,32,32]f32
# FLOPS = 798.72K
# memory OPS = 19.552K

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

        self.proj = nn.Linear(bias=True, in_features=32, out_features=96)

        archive = zipfile.ZipFile('rewrite.pnnx.bin', 'r')
        self.proj.bias = self.load_pnnx_bin_as_parameter(archive, 'proj.bias', (96), 'float32')
        self.proj.weight = self.load_pnnx_bin_as_parameter(archive, 'proj.weight', (96,32), 'float32')
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
        v_2 = self.proj(v_1)
        v_3 = v_2.reshape(4, 32, 96)
        return v_3

def export_torchscript():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(4, 32, 32, dtype=torch.float)

    mod = torch.jit.trace(net, v_0)
    mod.save("rewrite_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(4, 32, 32, dtype=torch.float)

    torch.onnx.export(net, v_0, "rewrite_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0'], output_names=['out0'])

def export_pnnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(4, 32, 32, dtype=torch.float)

    import pnnx
    pnnx.export(net, "rewrite_pnnx.py.pt", v_0)

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
