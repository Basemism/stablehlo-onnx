from .ir import TensorType

def dtype_to_onnx(dtype):
    return {
        'f32': 'FLOAT',
        'f16': 'FLOAT16',
        'bf16': 'BFLOAT16',
        'i64': 'INT64',
        'i32': 'INT32',
        'i8': 'INT8',
        'u8': 'UINT8',
    }.get(dtype, 'FLOAT')

def strip_loc(s: str) -> str:
    out = []
    i = 0
    while i < len(s):
        if s.startswith(" loc(", i):
            depth = 1
            i += 5
            while i < len(s) and depth > 0:
                if s[i] == '(':
                    depth += 1
                elif s[i] == ')':
                    depth -= 1
                i += 1
            continue
        out.append(s[i])
        i += 1
    return "".join(out)

def parse_tensor_type(t: str):
    t = t.strip()
    if not t.startswith("tensor<") or not t.endswith(">"):
        return None
    inner = t[len("tensor<"):-1]
    parts = inner.split("x")
    if len(parts) == 1:
        dtype = parts[0]
        dims = []
    else:
        dtype = parts[-1]
        dims = []
        for p in parts[:-1]:
            p = p.strip()
            if p == "?":
                dims.append(None)
            else:
                try:
                    dims.append(int(p))
                except ValueError:
                    dims.append(None)
    return TensorType(dims=dims, dtype=dtype)