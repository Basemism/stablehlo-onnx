from typing import Dict, List, Optional
from .ir import Module, Function, Operation
from .registry import REGISTRY, Context
from .utils import parse_tensor_type
from onnxscript import script
from onnxscript.onnx_opset import opset18 as op
import onnx
from onnx import helper, TensorProto, shape_inference

_DTYPE_MAP = {
    "f32": TensorProto.FLOAT,
    "f16": TensorProto.FLOAT16,
    "bf16": TensorProto.BFLOAT16,
    "f64": TensorProto.DOUBLE,
    "i64": TensorProto.INT64,
    "i32": TensorProto.INT32,
    "i16": TensorProto.INT16,
    "i8":  TensorProto.INT8,
    "u64": TensorProto.UINT64,
    "u32": TensorProto.UINT32,
    "u16": TensorProto.UINT16,
    "u8":  TensorProto.UINT8,
    "bool": TensorProto.BOOL,
}

def _to_tensorproto_dtype(elem: str):
    return _DTYPE_MAP.get(elem, TensorProto.FLOAT)

class OnnxBuilder:
    def __init__(self):
        self._name_map: Dict[str, str] = {}
        self._ops: List[str] = []
        self._inputs: List[str] = []
        self._input_shapes: Dict[str, List[int]] = {}
        self._returns: Optional[str] = None
        self._output_shape: Optional[List[int]] = None

    def get_tensor_name(self, ssa: str) -> str:
        if ssa not in self._name_map:
            if ssa.startswith("%arg"):
                self._name_map[ssa] = ssa[1:]                  # %arg0 -> "arg0"
            elif ssa.startswith("%") and ssa[1:].isalpha():
                self._name_map[ssa] = ssa[1:]                  # %cst  -> "cst"
            elif ssa.startswith("%"):
                self._name_map[ssa] = "v" + ssa[1:]            # %1    -> "v1"
            else:
                self._name_map[ssa] = ssa
        return self._name_map[ssa]


    def add(self, a: str, b: str, hint: Optional[str]=None) -> str:
        out = self.get_tensor_name(hint) if hint else f"add_{len(self._ops)}"
        self._ops.append(f"    {out} = op.Add({a}, {b})")
        return out

    def relu(self, x: str, hint: Optional[str]=None) -> str:
        out = self.get_tensor_name(hint) if hint else f"relu_{len(self._ops)}"
        self._ops.append(f"    {out} = op.Relu({x})")
        return out

    def max(self, a: str, b: str, hint: Optional[str]=None) -> str:
        out = self.get_tensor_name(hint) if hint else f"max_{len(self._ops)}"
        self._ops.append(f"    {out} = op.Max({a}, {b})")
        return out

    def mul(self, a: str, b: str, hint: Optional[str]=None) -> str:
        out = self.get_tensor_name(hint) if hint else f"mul_{len(self._ops)}"
        self._ops.append(f"    {out} = op.Mul({a}, {b})")
        return out

    def broadcast(self, x: str, hint: Optional[str]=None) -> str:
        out = self.get_tensor_name(hint) if hint else x
        self._ops.append(f"    {out} = op.Identity({x})")
        return out

    def constant(self, op: Operation, hint: Optional[str]=None) -> str:
        out = self.get_tensor_name(hint) if hint else f"const_{len(self._ops)}"
        lit = (op.attrs or {}).get("dense", None)

        elem_type = None

        if op.type_str:
            ttype = parse_tensor_type(op.type_str)
            if ttype and ttype.dtype:
                elem_type = ttype.dtype

        # print(f"Constant {out} with type {op.type_str} -> {elem_type} and literal {lit}")
        # print(type(lit))        

        # Check if lit is a scalar
        if lit is not None and not ("[" in lit or "]" in lit):
            # print(f"  Scalar constant {lit} of type {elem_type}")
            try:
                if elem_type.startswith("i"):
                    ival = int(float(lit))
                    self._ops.append(f"    {out} = op.Constant(value_int={ival})")
                    return out
                else:
                    fval = float(lit)
                    self._ops.append(f"    {out} = op.Constant(value_float={fval})")
                    return out
            except ValueError:
                pass

        # Fallback just emit Constant() and ignore literal for now
        self._ops.append(f"    {out} = op.Constant()")
        return out

    def matmul(self, a: str, b: str, hint: Optional[str]=None) -> str:
        out = self.get_tensor_name(hint) if hint else f"matmul_{len(self._ops)}"
        self._ops.append(f"    {out} = op.MatMul({a}, {b})")
        return out

    # Build the ONNX model from a Function IR
    def build(self, func: Function, opset: int = 18) -> str:
        for v in func.args:
            name = self.get_tensor_name(v.name)
            self._inputs.append(name)
            dims = []
            if v.type and v.type.dims is not None:
                dims = [d if d is not None else -1 for d in v.type.dims]
            self._input_shapes[name] = dims

            if not hasattr(self, "_input_dtypes"):
                self._input_dtypes = {}
            self._input_dtypes[name] = (v.type.dtype if v.type else "f32")

        for opn in func.ops:
            if opn.opname == "return":
                self._returns = self.get_tensor_name(opn.operands[0])
                continue
            handler = REGISTRY.get(opn.opname) or REGISTRY.get("stablehlo." + opn.opname)
            if handler is None:
                raise NotImplementedError(f"No handler for op: {opn.opname}")
            ctx = Context(self)
            handler(opn, ctx)

        if self._returns is None and func.ops and func.ops[-1].result:
            self._returns = self.get_tensor_name(func.ops[-1].result)

        # Capture output shape if present in function type
        if func.rets and func.rets[0] and func.rets[0].dims is not None:
            self._output_shape = [d if d is not None else -1 for d in func.rets[0].dims]

        if func.rets and func.rets[0]:
            self._output_dtype = func.rets[0].dtype or "f32"
        
        # Generate the ONNX Script source
        header = ("from onnxscript import script\n"
               "from onnxscript.onnx_opset import opset18 as op\n"
               "import onnx\n"
               "\n")
        signature = ("@script()\n"
               "def model(" + ", ".join(self._inputs) + "):\n")
        body = "\n".join(self._ops) + "\n"
        ret = f"    return {self._returns}\n"
        return header + signature + body + ret

# Prefer @main as entry point
def _pick_entry(mod: Module) -> Function:
    for f in mod.functions:
        if f.name == "main":
            return f
    return mod.functions[0]

# Write script to file & import (onnxscript needs real source file)
import importlib.util, sys, tempfile, os

# Convert Module to ONNX ModelProto
def module_to_onnx(mod: Module, opset: int = 18):
    func = _pick_entry(mod)
    b = OnnxBuilder()
    script_src = b.build(func, opset=opset)

    with tempfile.TemporaryDirectory() as td:
        script_path = os.path.join(td, "gen_model.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_src)

        spec = importlib.util.spec_from_file_location("gen_model", script_path)
        modpy = importlib.util.module_from_spec(spec)
        sys.modules["gen_model"] = modpy
        spec.loader.exec_module(modpy)

        model = modpy.model.to_model_proto()

    # Update inputs/outputs with shapes & types
    for name, dims in b._input_shapes.items():
        onnx_dims = [d if (isinstance(d, int) and d >= 0) else None for d in dims]
        dtype = _to_tensorproto_dtype(getattr(b, "_input_dtypes", {}).get(name, "f32"))
        vi = helper.make_tensor_value_info(name, dtype, onnx_dims or None)
        if not any(i.name == name for i in model.graph.input):
            model.graph.input.append(vi)
        
        found = False
        for i, old in enumerate(model.graph.input):
            if old.name == name:
                model.graph.input[i].CopyFrom(vi)
                found = True
                break
        if not found:
            model.graph.input.append(vi)

    # Update outputs with shapes & types
    if b._returns:
        out_name = b._returns
        out_shape = getattr(b, "_output_shape", None)
        out_dtype = _to_tensorproto_dtype(getattr(b, "_output_dtype", "f32"))
        onnx_out_dims = None
        if out_shape is not None:
            onnx_out_dims = [d if (isinstance(d, int) and d >= 0) else None for d in out_shape]

        vo = helper.make_tensor_value_info(out_name, out_dtype, onnx_out_dims)

        # clear & set the single output
        while len(model.graph.output):
            model.graph.output.pop()
        model.graph.output.append(vo)
        
    model = shape_inference.infer_shapes(model)

    return model, script_src