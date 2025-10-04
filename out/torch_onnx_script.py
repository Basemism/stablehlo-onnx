from onnxscript import script
from onnxscript.onnx_opset import opset18 as op
import onnx

@script()
def model(arg0, arg1, arg2):
    cst = op.Constant(value_float=1.0)
    v0 = op.Identity(cst)
    v1 = op.Mul(arg1, v0)
    v2 = op.Add(arg0, v1)
    v3 = op.Relu(v2)
    v4 = op.MatMul(v3, arg2)
    return v4
