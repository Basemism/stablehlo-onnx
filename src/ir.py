from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class TensorType:
    dims: List[Optional[int]]
    dtype: str

@dataclass
class Value:
    name: str
    type: Optional[TensorType] = None
    data: Optional[Any] = None

@dataclass
class Operation:
    result: Optional[str]
    opname: str
    operands: List[str]
    attrs: Dict[str, Any] = field(default_factory=dict)
    type_str: Optional[str] = None
    loc: Optional[str] = None

@dataclass
class Function:
    name: str
    args: List[Value]
    rets: List[TensorType]
    ops: List[Operation] = field(default_factory=list)

@dataclass
class Module:
    name: Optional[str]
    functions: List[Function]
