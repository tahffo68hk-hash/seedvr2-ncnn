import numpy as np
import ncnn
import torch

def test_inference():
    torch.manual_seed(0)
    in0 = torch.rand(4, 64, 32, dtype=torch.float)
    in1 = torch.rand(4, 64, 64, dtype=torch.float)
    in2 = torch.rand(64, 4, dtype=torch.float)
    in3 = torch.rand(64, 4, dtype=torch.float)
    out = []

    with ncnn.Net() as net:
        net.load_param("model.ncnn.param")
        net.load_model("model.ncnn.bin")

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(in0.numpy(), batch_index=233).clone())
            ex.input("in1", ncnn.Mat(in1.numpy(), batch_index=0).clone())
            ex.input("in2", ncnn.Mat(in2.numpy(), batch_index=233).clone())
            ex.input("in3", ncnn.Mat(in3.numpy(), batch_index=233).clone())

            _, out0 = ex.extract("out0")
            out.append(torch.from_numpy(out0.numpy(batch_index=0)))
            _, out1 = ex.extract("out1")
            out.append(torch.from_numpy(out1.numpy(batch_index=0)))
            _, out2 = ex.extract("out2")
            out.append(torch.from_numpy(out2.numpy(batch_index=0)))
            _, out3 = ex.extract("out3")
            out.append(torch.from_numpy(out3.numpy(batch_index=0)))

    if len(out) == 1:
        return out[0]
    else:
        return tuple(out)

if __name__ == "__main__":
    print(test_inference())
