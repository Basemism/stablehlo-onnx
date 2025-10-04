from onnxscript import script
from onnxscript.onnx_opset import opset18 as op
import onnx

@script()
def model(arg0, arg1, arg2):
    v0 = op.Add(arg0, arg1)
    v1 = op.Relu(v0)
    v2 = op.MatMul(v1, arg2)
    return v2
