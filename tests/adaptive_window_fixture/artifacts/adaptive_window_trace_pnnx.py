# pnnx model stat
# model inputshape = [1,2,8,8,32]f32
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

        self.attention_wq = nn.Linear(bias=True, in_features=32, out_features=32)
        self.attention_wk = nn.Linear(bias=True, in_features=32, out_features=32)
        self.attention_wv = nn.Linear(bias=True, in_features=32, out_features=32)
        self.attention_wo = nn.Linear(bias=True, in_features=32, out_features=32)

        archive = zipfile.ZipFile('adaptive_window_trace.pnnx.bin', 'r')
        self.attention_wq.bias = self.load_pnnx_bin_as_parameter(archive, 'attention.wq.bias', (32), 'float32')
        self.attention_wq.weight = self.load_pnnx_bin_as_parameter(archive, 'attention.wq.weight', (32,32), 'float32')
        self.attention_wk.bias = self.load_pnnx_bin_as_parameter(archive, 'attention.wk.bias', (32), 'float32')
        self.attention_wk.weight = self.load_pnnx_bin_as_parameter(archive, 'attention.wk.weight', (32,32), 'float32')
        self.attention_wv.bias = self.load_pnnx_bin_as_parameter(archive, 'attention.wv.bias', (32), 'float32')
        self.attention_wv.weight = self.load_pnnx_bin_as_parameter(archive, 'attention.wv.weight', (32,32), 'float32')
        self.attention_wo.bias = self.load_pnnx_bin_as_parameter(archive, 'attention.wo.bias', (32), 'float32')
        self.attention_wo.weight = self.load_pnnx_bin_as_parameter(archive, 'attention.wo.weight', (32,32), 'float32')
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
        v_1 = v_0.reshape(1, 1, 2, 2, 4, 2, 4, 32)
        v_2 = v_1.permute(dims=(0,1,3,5,2,4,6,7))
        v_3 = v_2.reshape(4, 32, 32)
        v_4 = self.attention_wq(v_3)
        v_5 = v_4.reshape(4, 32, 4, 8)
        v_6 = torch.transpose(v_5, dim0=1, dim1=2)
        v_7 = self.attention_wk(v_3)
        v_8 = v_7.reshape(4, 32, 4, 8)
        v_9 = torch.transpose(v_8, dim0=1, dim1=2)
        v_10 = self.attention_wv(v_3)
        v_11 = v_10.reshape(4, 32, 4, 8)
        v_12 = torch.transpose(v_11, dim0=1, dim1=2)
        v_13 = v_6.reshape(16, 32, 8)
        v_14 = v_9.reshape(16, 32, 8)
        v_15 = v_12.reshape(16, 32, 8)
        v_16 = torch.transpose(v_14, dim0=-2, dim1=-1)
        v_17 = torch.matmul(v_13, other=v_16)
        v_18 = (v_17 * 0.353553385)
        v_19 = F.softmax(v_18, dim=-1)
        v_20 = torch.matmul(v_19, other=v_15)
        v_21 = v_20.reshape(4, 4, 32, 8)
        v_22 = torch.transpose(v_21, dim0=1, dim1=2)
        v_23 = v_22.reshape(4, 32, 32)
        v_24 = self.attention_wo(v_23)
        v_25 = v_24.reshape(1, 1, 2, 2, 2, 4, 4, 32)
        v_26 = v_25.permute(dims=(0,1,4,2,5,3,6,7))
        v_27 = v_26.reshape(1, 2, 8, 8, 32)
        return v_27

def export_torchscript():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 2, 8, 8, 32, dtype=torch.float)

    mod = torch.jit.trace(net, v_0)
    mod.save("adaptive_window_trace_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 2, 8, 8, 32, dtype=torch.float)

    torch.onnx.export(net, v_0, "adaptive_window_trace_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0'], output_names=['out0'])

def export_pnnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 2, 8, 8, 32, dtype=torch.float)

    import pnnx
    pnnx.export(net, "adaptive_window_trace_pnnx.py.pt", v_0)

def export_ncnn():
    export_pnnx()

@torch.no_grad()
def test_inference():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 2, 8, 8, 32, dtype=torch.float)

    return net(v_0)

if __name__ == "__main__":
    print(test_inference())
