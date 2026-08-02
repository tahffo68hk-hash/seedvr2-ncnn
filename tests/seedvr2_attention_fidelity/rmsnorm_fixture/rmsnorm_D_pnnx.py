# pnnx model stat
# model inputshape = [4,64,8]f32
# FLOPS = 20.736K
# memory OPS = 10.76K

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


        archive = zipfile.ZipFile('rmsnorm_D.pnnx.bin', 'r')
        self.weight_data = self.load_pnnx_bin_as_parameter(archive, 'weight.data', (8,), 'float32')
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
        v_1 = self.weight_data
        v_2 = (v_0 * v_0)
        v_3 = torch.mean(v_2, dim=(-1,), keepdim=True)
        v_4 = ((v_0 * torch.reciprocal(torch.sqrt((v_3 + 1.000000e-5)))) * v_1)
        return v_4

def export_torchscript():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(4, 64, 8, dtype=torch.float)

    mod = torch.jit.trace(net, v_0)
    mod.save("rmsnorm_D_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(4, 64, 8, dtype=torch.float)

    torch.onnx.export(net, v_0, "rmsnorm_D_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0'], output_names=['out0'])

def export_pnnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(4, 64, 8, dtype=torch.float)

    import pnnx
    pnnx.export(net, "rmsnorm_D_pnnx.py.pt", v_0)

def export_ncnn():
    export_pnnx()

@torch.no_grad()
def test_inference():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(4, 64, 8, dtype=torch.float)

    return net(v_0)

if __name__ == "__main__":
    print(test_inference())
