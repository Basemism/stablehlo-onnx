from typing import Callable, Dict, Optional
from .ir import Operation

class Context:
    def __init__(self, builder):
        self.b = builder

    def val(self, ssa_name: str) -> str:
        return self.b.get_tensor_name(ssa_name)
    
def handle_add(op: Operation, ctx: Context) -> str:
    a, b = op.operands
    return ctx.b.add(ctx.val(a), ctx.val(b), hint=op.result)

def handle_relu_direct(op: Operation, ctx: Context) -> str:
    x = op.operands[0]
    return ctx.b.relu(ctx.val(x), hint=op.result)

def handle_maximum(op: Operation, ctx: Context) -> str:
    a, b = op.operands
    return ctx.b.max(ctx.val(a), ctx.val(b), hint=op.result)

def handle_multiply(op: Operation, ctx: Context) -> str:
    a, b = op.operands
    return ctx.b.mul(ctx.val(a), ctx.val(b), hint=op.result)

def handle_broadcast_in_dim(op: Operation, ctx: Context) -> str:
    src = op.operands[0]
    return ctx.b.broadcast(ctx.val(src), hint=op.result)

def handle_constant(op: Operation, ctx: Context) -> str:
    return ctx.b.materialize_constant(op, hint=op.result)

def handle_dot_general(op: Operation, ctx: Context) -> str:
    a, b = op.operands
    cd = op.attrs.get("contracting_dims")
    if cd is not None:
        left, right = cd
        if left == [1] and right == [0]:
            return ctx.b.matmul(ctx.val(a), ctx.val(b), hint=op.result)
    return ctx.b.matmul(ctx.val(a), ctx.val(b), hint=op.result)

REGISTRY: Dict[str, Callable[[Operation, Context], str]] = {
    "stablehlo.add": handle_add,
    "stablehlo.relu": handle_relu_direct,
    "stablehlo.maximum": handle_maximum,
    "stablehlo.multiply": handle_multiply,
    "stablehlo.broadcast_in_dim": handle_broadcast_in_dim,
    "stablehlo.constant": handle_constant,
    "stablehlo.dot_general": handle_dot_general,
}