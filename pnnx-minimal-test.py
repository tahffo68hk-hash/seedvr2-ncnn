import torch
import torch.nn as nn


class LinearRelu(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 4)

    def forward(self, x):
        return torch.relu(self.linear(x))


torch.manual_seed(0)
model = LinearRelu().eval()
example = torch.ones(1, 10)
traced = torch.jit.trace(model, example)
traced.save("linear_relu.pt")
print("saved linear_relu.pt")
print("output_shape", tuple(traced(example).shape))
