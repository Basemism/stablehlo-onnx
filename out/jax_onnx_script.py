from onnxscript import script
from onnxscript.onnx_opset import opset18 as op
import onnx

@script()
def model(arg0, arg1, arg2, arg3, arg4):
    v0 = op.MatMul(arg0, arg1)
    v1 = op.Identity(arg2)
    v2 = op.Identity(v1)
    v3 = op.Add(v0, v2)
    v4 = op.Relu(v3)
    v5 = op.MatMul(v4, arg3)
    v6 = op.Identity(arg4)
    v7 = op.Identity(v6)
    v8 = op.Add(v5, v7)
    return v8
