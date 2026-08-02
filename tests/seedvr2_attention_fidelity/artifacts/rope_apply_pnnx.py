# pnnx model stat
# model inputshape = [64,4,8]f32,[64,4]f32
# FLOPS = 8.192K
# memory OPS = 12.288K

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


        archive = zipfile.ZipFile('rope_apply.pnnx.bin', 'r')
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
        v_2 = v_0.reshape(64, 4, 4, 2)
        v_3 = torch.cos(v_1)
        v_4 = v_3.unsqueeze(1)
        v_5 = torch.sin(v_1)
        v_6 = v_5.unsqueeze(1)
        v_7, v_8 = torch.unbind(v_2, dim=3)
        v_9 = ((v_7 * v_4) - (v_8 * v_6))
        v_10 = ((v_7 * v_6) + (v_8 * v_4))
        v_11 = torch.stack((v_9, v_10), dim=-1)
        v_12 = v_11.reshape_as(other=v_0, )
        return v_12

def export_torchscript():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(64, 4, 8, dtype=torch.float)
    v_1 = torch.rand(64, 4, dtype=torch.float)

    mod = torch.jit.trace(net, (v_0, v_1))
    mod.save("rope_apply_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(64, 4, 8, dtype=torch.float)
    v_1 = torch.rand(64, 4, dtype=torch.float)

    torch.onnx.export(net, (v_0, v_1), "rope_apply_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0', 'in1'], output_names=['out0'])

def export_pnnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(64, 4, 8, dtype=torch.float)
    v_1 = torch.rand(64, 4, dtype=torch.float)

    import pnnx
    pnnx.export(net, "rope_apply_pnnx.py.pt", (v_0, v_1))

def export_ncnn():
    export_pnnx()

@torch.no_grad()
def test_inference():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(64, 4, 8, dtype=torch.float)
    v_1 = torch.rand(64, 4, dtype=torch.float)

    return net(v_0, v_1)

if __name__ == "__main__":
    print(test_inference())
