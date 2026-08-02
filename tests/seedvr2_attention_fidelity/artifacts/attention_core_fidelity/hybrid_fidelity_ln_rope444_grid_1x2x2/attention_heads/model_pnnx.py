# pnnx model stat
# model inputshape = [4,64,32]f32,[4,64,64]f32,[64,4]f32,[64,4]f32
# FLOPS = 4.35M
# memory OPS = 677.6K

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
        self.ln_2 = nn.LayerNorm(elementwise_affine=True, eps=1.000000e-5, normalized_shape=(8,))
        self.ln_3 = nn.LayerNorm(elementwise_affine=True, eps=1.000000e-5, normalized_shape=(8,))
        self.ln_4 = nn.LayerNorm(elementwise_affine=True, eps=1.000000e-5, normalized_shape=(8,))
        self.ln_5 = nn.LayerNorm(elementwise_affine=True, eps=1.000000e-5, normalized_shape=(8,))
        self.ln_6 = nn.LayerNorm(elementwise_affine=True, eps=1.000000e-5, normalized_shape=(8,))
        self.ln_7 = nn.LayerNorm(elementwise_affine=True, eps=1.000000e-5, normalized_shape=(8,))

        archive = zipfile.ZipFile('model.pnnx.bin', 'r')
        self.qkv.bias = self.load_pnnx_bin_as_parameter(archive, 'qkv.bias', (96), 'float32')
        self.qkv.weight = self.load_pnnx_bin_as_parameter(archive, 'qkv.weight', (96,32), 'float32')
        self.ln_0.bias = self.load_pnnx_bin_as_parameter(archive, 'ln_0.bias', (8), 'float32')
        self.ln_0.weight = self.load_pnnx_bin_as_parameter(archive, 'ln_0.weight', (8), 'float32')
        self.ln_1.bias = self.load_pnnx_bin_as_parameter(archive, 'ln_1.bias', (8), 'float32')
        self.ln_1.weight = self.load_pnnx_bin_as_parameter(archive, 'ln_1.weight', (8), 'float32')
        self.ln_2.bias = self.load_pnnx_bin_as_parameter(archive, 'ln_2.bias', (8), 'float32')
        self.ln_2.weight = self.load_pnnx_bin_as_parameter(archive, 'ln_2.weight', (8), 'float32')
        self.ln_3.bias = self.load_pnnx_bin_as_parameter(archive, 'ln_3.bias', (8), 'float32')
        self.ln_3.weight = self.load_pnnx_bin_as_parameter(archive, 'ln_3.weight', (8), 'float32')
        self.ln_4.bias = self.load_pnnx_bin_as_parameter(archive, 'ln_4.bias', (8), 'float32')
        self.ln_4.weight = self.load_pnnx_bin_as_parameter(archive, 'ln_4.weight', (8), 'float32')
        self.ln_5.bias = self.load_pnnx_bin_as_parameter(archive, 'ln_5.bias', (8), 'float32')
        self.ln_5.weight = self.load_pnnx_bin_as_parameter(archive, 'ln_5.weight', (8), 'float32')
        self.ln_6.bias = self.load_pnnx_bin_as_parameter(archive, 'ln_6.bias', (8), 'float32')
        self.ln_6.weight = self.load_pnnx_bin_as_parameter(archive, 'ln_6.weight', (8), 'float32')
        self.ln_7.bias = self.load_pnnx_bin_as_parameter(archive, 'ln_7.bias', (8), 'float32')
        self.ln_7.weight = self.load_pnnx_bin_as_parameter(archive, 'ln_7.weight', (8), 'float32')
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
        v_6 = v_5.reshape(4, 64, 96)
        v_7, v_8, v_9 = torch.split(tensor=v_6, dim=-1, split_size_or_sections=32)
        v_10, v_11, v_12, v_13 = torch.tensor_split(v_7, dim=2, indices=(8,16,24))
        v_14, v_15, v_16, v_17 = torch.tensor_split(v_8, dim=2, indices=(8,16,24))
        v_18, v_19, v_20, v_21 = torch.tensor_split(v_9, dim=2, indices=(8,16,24))
        v_22 = self.ln_0(v_10)
        v_23 = self.ln_1(v_14)
        v_24 = v_2.repeat(4, 1)
        v_25 = v_3.repeat(4, 1)
        v_26 = v_22.reshape(256, 1, 8)
        v_27 = v_26.reshape(256, 1, 4, 2)
        v_28 = v_24.unsqueeze(1)
        v_29 = v_25.unsqueeze(1)
        v_30, v_31 = torch.unbind(v_27, dim=3)
        v_32 = ((v_30 * v_28) - (v_31 * v_29))
        v_33 = ((v_30 * v_29) + (v_31 * v_28))
        v_34 = torch.stack((v_32, v_33), dim=-1)
        v_35 = v_34.reshape_as(other=v_26, )
        v_36 = v_35.reshape_as(other=v_22, )
        v_37 = v_23.reshape(256, 1, 8)
        v_38 = v_37.reshape(256, 1, 4, 2)
        v_39, v_40 = torch.unbind(v_38, dim=3)
        v_41 = ((v_39 * v_28) - (v_40 * v_29))
        v_42 = ((v_39 * v_29) + (v_40 * v_28))
        v_43 = torch.stack((v_41, v_42), dim=-1)
        v_44 = v_43.reshape_as(other=v_37, )
        v_45 = v_44.reshape_as(other=v_23, )
        v_46 = torch.transpose(v_45, dim0=-2, dim1=-1)
        v_47 = torch.matmul(v_36, other=v_46)
        v_48 = ((v_47 * 0.353553385) + v_1)
        v_49 = F.softmax(v_48, dim=-1)
        v_50 = torch.matmul(v_49, other=v_18)
        v_51 = self.ln_2(v_11)
        v_52 = self.ln_3(v_15)
        v_53 = v_51.reshape(256, 1, 8)
        v_54 = v_53.reshape(256, 1, 4, 2)
        v_55, v_56 = torch.unbind(v_54, dim=3)
        v_57 = ((v_55 * v_28) - (v_56 * v_29))
        v_58 = ((v_55 * v_29) + (v_56 * v_28))
        v_59 = torch.stack((v_57, v_58), dim=-1)
        v_60 = v_59.reshape_as(other=v_53, )
        v_61 = v_60.reshape_as(other=v_51, )
        v_62 = v_52.reshape(256, 1, 8)
        v_63 = v_62.reshape(256, 1, 4, 2)
        v_64, v_65 = torch.unbind(v_63, dim=3)
        v_66 = ((v_64 * v_28) - (v_65 * v_29))
        v_67 = ((v_64 * v_29) + (v_65 * v_28))
        v_68 = torch.stack((v_66, v_67), dim=-1)
        v_69 = v_68.reshape_as(other=v_62, )
        v_70 = v_69.reshape_as(other=v_52, )
        v_71 = torch.transpose(v_70, dim0=-2, dim1=-1)
        v_72 = torch.matmul(v_61, other=v_71)
        v_73 = ((v_72 * 0.353553385) + v_1)
        v_74 = F.softmax(v_73, dim=-1)
        v_75 = torch.matmul(v_74, other=v_19)
        v_76 = self.ln_4(v_12)
        v_77 = self.ln_5(v_16)
        v_78 = v_76.reshape(256, 1, 8)
        v_79 = v_78.reshape(256, 1, 4, 2)
        v_80, v_81 = torch.unbind(v_79, dim=3)
        v_82 = ((v_80 * v_28) - (v_81 * v_29))
        v_83 = ((v_80 * v_29) + (v_81 * v_28))
        v_84 = torch.stack((v_82, v_83), dim=-1)
        v_85 = v_84.reshape_as(other=v_78, )
        v_86 = v_85.reshape_as(other=v_76, )
        v_87 = v_77.reshape(256, 1, 8)
        v_88 = v_87.reshape(256, 1, 4, 2)
        v_89, v_90 = torch.unbind(v_88, dim=3)
        v_91 = ((v_89 * v_28) - (v_90 * v_29))
        v_92 = ((v_89 * v_29) + (v_90 * v_28))
        v_93 = torch.stack((v_91, v_92), dim=-1)
        v_94 = v_93.reshape_as(other=v_87, )
        v_95 = v_94.reshape_as(other=v_77, )
        v_96 = torch.transpose(v_95, dim0=-2, dim1=-1)
        v_97 = torch.matmul(v_86, other=v_96)
        v_98 = ((v_97 * 0.353553385) + v_1)
        v_99 = F.softmax(v_98, dim=-1)
        v_100 = torch.matmul(v_99, other=v_20)
        v_101 = self.ln_6(v_13)
        v_102 = self.ln_7(v_17)
        v_103 = v_101.reshape(256, 1, 8)
        v_104 = v_103.reshape(256, 1, 4, 2)
        v_105, v_106 = torch.unbind(v_104, dim=3)
        v_107 = ((v_105 * v_28) - (v_106 * v_29))
        v_108 = ((v_105 * v_29) + (v_106 * v_28))
        v_109 = torch.stack((v_107, v_108), dim=-1)
        v_110 = v_109.reshape_as(other=v_103, )
        v_111 = v_110.reshape_as(other=v_101, )
        v_112 = v_102.reshape(256, 1, 8)
        v_113 = v_112.reshape(256, 1, 4, 2)
        v_114, v_115 = torch.unbind(v_113, dim=3)
        v_116 = ((v_114 * v_28) - (v_115 * v_29))
        v_117 = ((v_114 * v_29) + (v_115 * v_28))
        v_118 = torch.stack((v_116, v_117), dim=-1)
        v_119 = v_118.reshape_as(other=v_112, )
        v_120 = v_119.reshape_as(other=v_102, )
        v_121 = torch.transpose(v_120, dim0=-2, dim1=-1)
        v_122 = torch.matmul(v_111, other=v_121)
        v_123 = ((v_122 * 0.353553385) + v_1)
        v_124 = F.softmax(v_123, dim=-1)
        v_125 = torch.matmul(v_124, other=v_21)
        v_126 = (v_50, v_75, v_100, v_125, )
        return v_126

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
