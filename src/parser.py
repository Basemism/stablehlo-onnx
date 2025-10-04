from typing import List, Dict, Optional, Any
from .ir import Module, Function, Operation, Value
from .utils import strip_loc, parse_tensor_type

def _clean(line: str) -> str:
    return strip_loc(line).strip()

def parse_module(text: str) -> Module:
    lines = [ln.rstrip("\n") for ln in text.splitlines() if ln.strip() != ""]
    name = None
    functions: List[Function] = []
    i = 0
    while i < len(lines):
        line = _clean(lines[i])
        if line.startswith("module"):
            toks = line.split()
            if len(toks) > 1 and toks[1].startswith("@"):
                name = toks[1][1:]
            i += 1
            continue
        if line.startswith("func.func"):
            func, j = _parse_function(lines, i)
            functions.append(func)
            i = j
            continue
        i += 1
    return Module(name=name, functions=functions)

def _parse_function(lines: List[str], start: int):
    sig = _clean(lines[start])

    at = sig.find("@")
    lp = sig.find("(", at)
    fname = sig[at + 1 : lp]

    rp = sig.find(")", lp)
    args_seg = sig[lp + 1 : rp]
    args: List[Value] = []
    if args_seg.strip():
        for tok in args_seg.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if ":" in tok:
                vname, tstr = tok.split(":", 1)
                vname = vname.strip()
                t = parse_tensor_type(tstr.strip())
                args.append(Value(name=vname, type=t))
            else:
                args.append(Value(name=tok.strip()))

    rets: List = []
    arr = sig.find("->", rp)
    if arr != -1:
        ret_seg = sig[arr + 2 :].strip()
        if ret_seg.startswith("("):
            inner = ret_seg[1 : ret_seg.rfind(")")]
            parts = [p.strip() for p in inner.split(",") if p.strip()]
            for p in parts:
                tstr = p.split("{", 1)[0].strip()
                rets.append(parse_tensor_type(tstr))
        else:
            tstr = ret_seg.split("{", 1)[0].strip()
            rets.append(parse_tensor_type(tstr))
    i = start
    if "{" in _clean(lines[i]):
        open_line = i
        depth = _clean(lines[i]).count("{") - _clean(lines[i]).count("}")
    else:
        depth = 0
        i += 1
        while i < len(lines):
            if "{" in _clean(lines[i]):
                open_line = i
                depth = _clean(lines[i]).count("{") - _clean(lines[i]).count("}")
                break
            i += 1

    j = open_line + 1
    while j < len(lines) and depth > 0:
        cj = _clean(lines[j])
        depth += cj.count("{")
        depth -= cj.count("}")
        j += 1

    body_start = open_line + 1
    body_end_excl = j - 1 if j - 1 >= body_start else body_start - 1

    ops: List[Operation] = []
    k = body_start
    while k <= body_end_excl:
        line = _clean(lines[k])
        if not line or line == "{" or line == "}":
            k += 1
            continue

        if line.startswith("%") and "=" in line:
            res, rhs = line.split("=", 1)
            res = res.strip()
            rhs = rhs.strip()
            first_space = rhs.find(" ")
            op_token = rhs if first_space == -1 else rhs[:first_space]
            rest = "" if first_space == -1 else rhs[first_space + 1 :]

            if op_token == "call":
                # %1 = call @relu(%0) : (...) -> ...
                at2 = rest.find("@")
                lp2 = rest.find("(", at2)
                target = rest[at2 + 1 : lp2].strip()
                args_seg2 = rest[lp2 + 1 : rest.find(")", lp2)]
                operands = [a.strip() for a in args_seg2.split(",") if a.strip()]
                ops.append(Operation(result=res, opname=f"call@{target}", operands=operands))
            else:
                operands, attrs, type_str = _parse_operands_attrs_types(rest)
                ops.append(Operation(result=res, opname=op_token, operands=operands, attrs=attrs, type_str=type_str))

        elif line.startswith("return "):
            vals = line[len("return "):].split(":", 1)[0]
            operands = [v.strip() for v in vals.split(",") if v.strip()]
            ops.append(Operation(result=None, opname="return", operands=operands))

        k += 1

    return Function(name=fname, args=args, rets=rets, ops=ops), j

def _parse_operands_attrs_types(rest: str):
    type_str = None
    before = rest
    if " : " in rest:
        before, type_str = rest.rsplit(" : ", 1)

    operands: List[str] = []
    idx = 0
    while idx < len(before):
        if before[idx] == "%":
            j = idx + 1
            while j < len(before) and (before[j].isalnum() or before[j] == "_"):
                j += 1
            operands.append(before[idx:j])
            idx = j
        else:
            idx += 1

    attrs: Dict[str, Any] = {}

    if "dense<" in before:
        frag = before.split("dense<", 1)[1]
        lit = frag.split(">", 1)[0]
        attrs["dense"] = lit.strip()

    if "contracting_dims" in before:
        try:
            frag = before.split("contracting_dims", 1)[1]
            l1 = frag.find("["); r1 = frag.find("]")
            left = frag[l1+1:r1]
            frag2 = frag[r1+1:]
            l2 = frag2.find("["); r2 = frag2.find("]")
            right = frag2[l2+1:r2]
            attrs["contracting_dims"] = (
                [int(x) for x in left.replace(" ", "").split(",") if x != ""],
                [int(x) for x in right.replace(" ", "").split(",") if x != ""],
            )
        except Exception:
            pass

    return operands, attrs, type_str
