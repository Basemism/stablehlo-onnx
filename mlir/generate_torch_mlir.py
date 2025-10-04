import torch
from torch.export import export
import torch.nn as nn
import torchax as tx
import torchax.export
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Define a toy model using add, relu, and matmul (dot_general)
class ToyModel(nn.Module):
    def forward(self, x, y, z):
        a = x + y           # add
        b = torch.relu(a)   # relu
        c = torch.matmul(b, z)  # dot_general (matmul)
        return c

model = ToyModel()
model.eval()

x = torch.randn(2, 4)
y = torch.randn(2, 4)
z = torch.randn(4, 3)


exported = export(model, (x, y, z))

weights, stablehlo = torchax.export.exported_program_to_stablehlo(exported)

print(stablehlo.mlir_module())

with open("PoC_torch.mlir", "w") as f:
    f.write(stablehlo.mlir_module())

print("Saved PoC_torch.mlir \n\n")
