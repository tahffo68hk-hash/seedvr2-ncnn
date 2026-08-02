# pnnx model stat
# model inputshape = [64,512]f32,[64,64]f32,[64,64]f32,[64,64]f32
# FLOPS = 103.084M
# memory OPS = 1.194M

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

        self.qkv = nn.Linear(bias=True, in_features=512, out_features=1536)

        archive = zipfile.ZipFile('model.pnnx.bin', 'r')
        self.qkv.bias = self.load_pnnx_bin_as_parameter(archive, 'qkv.bias', (1536), 'float32')
        self.qkv.weight = self.load_pnnx_bin_as_parameter(archive, 'qkv.weight', (1536,512), 'float32')
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
        v_5 = v_4.reshape(1, 64, 1536)
        v_6, v_7, v_8 = torch.split(tensor=v_5, dim=-1, split_size_or_sections=512)
        v_9 = v_6[:,:,384:512]
        v_10 = v_7[:,:,384:512]
        v_11 = v_8[:,:,384:512]
        v_12 = v_9.reshape(64, 128)
        v_13 = v_10.reshape(64, 128)
        v_14 = (v_12 * v_12)
        v_15 = torch.mean(v_14, dim=(-1,), keepdim=True)
        v_16 = (v_13 * v_13)
        v_17 = torch.mean(v_16, dim=(-1,), keepdim=True)
        v_18 = (v_12 * torch.reciprocal(torch.sqrt((v_15 + 1.000000e-5))))
        v_19 = v_18.reshape_as(other=v_9, )
        v_20 = (v_13 * torch.reciprocal(torch.sqrt((v_17 + 1.000000e-5))))
        v_21 = v_20.reshape_as(other=v_10, )
        v_22 = v_2.repeat(1, 1)
        v_23 = v_3.repeat(1, 1)
        v_24 = v_19.reshape(64, 1, 128)
        v_25 = v_24.reshape(64, 1, 64, 2)
        v_26 = v_22.unsqueeze(1)
        v_27 = v_23.unsqueeze(1)
        v_28, v_29 = torch.unbind(v_25, dim=3)
        v_30 = ((v_28 * v_26) - (v_29 * v_27))
        v_31 = ((v_28 * v_27) + (v_29 * v_26))
        v_32 = torch.stack((v_30, v_31), dim=-1)
        v_33 = v_32.reshape_as(other=v_24, )
        v_34 = v_33.reshape_as(other=v_19, )
        v_35 = v_21.reshape(64, 1, 128)
        v_36 = v_35.reshape(64, 1, 64, 2)
        v_37, v_38 = torch.unbind(v_36, dim=3)
        v_39 = ((v_37 * v_26) - (v_38 * v_27))
        v_40 = ((v_37 * v_27) + (v_38 * v_26))
        v_41 = torch.stack((v_39, v_40), dim=-1)
        v_42 = v_41.reshape_as(other=v_35, )
        v_43 = v_42.reshape_as(other=v_21, )
        v_44 = torch.transpose(v_43, dim0=-2, dim1=-1)
        v_45 = torch.matmul(v_34, other=v_44)
        v_46 = ((v_45 * 8.83883461e-2) + v_1)
        v_47 = F.softmax(v_46, dim=-1)
        v_48 = torch.matmul(v_47, other=v_11)
        return v_48

def export_torchscript():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(64, 512, dtype=torch.float)
    v_1 = torch.rand(64, 64, dtype=torch.float)
    v_2 = torch.rand(64, 64, dtype=torch.float)
    v_3 = torch.rand(64, 64, dtype=torch.float)

    mod = torch.jit.trace(net, (v_0, v_1, v_2, v_3))
    mod.save("model_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(64, 512, dtype=torch.float)
    v_1 = torch.rand(64, 64, dtype=torch.float)
    v_2 = torch.rand(64, 64, dtype=torch.float)
    v_3 = torch.rand(64, 64, dtype=torch.float)

    torch.onnx.export(net, (v_0, v_1, v_2, v_3), "model_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0', 'in1', 'in2', 'in3'], output_names=['out0'])

def export_pnnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(64, 512, dtype=torch.float)
    v_1 = torch.rand(64, 64, dtype=torch.float)
    v_2 = torch.rand(64, 64, dtype=torch.float)
    v_3 = torch.rand(64, 64, dtype=torch.float)

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
    v_0 = torch.rand(64, 512, dtype=torch.float)
    v_1 = torch.rand(64, 64, dtype=torch.float)
    v_2 = torch.rand(64, 64, dtype=torch.float)
    v_3 = torch.rand(64, 64, dtype=torch.float)

    return net(v_0, v_1, v_2, v_3)

if __name__ == "__main__":
    print(test_inference())
