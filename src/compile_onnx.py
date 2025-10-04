import argparse, pathlib
from .parser import parse_module
from .onnx_builder import module_to_onnx
import onnx

def main():
    ap = argparse.ArgumentParser(description="StableHLO -> ONNX PoC")
    ap.add_argument("input", help="Input StableHLO text file")
    ap.add_argument("-o", "--output", default="model.onnx", help="Output ONNX path")
    ap.add_argument("--save_script", default=None, help="dump the generated ONNX script to this path")
    ap.add_argument("--opset", type=int, default=18, help="ONNX opset (default: 18)")
    args = ap.parse_args()

    text = pathlib.Path(args.input).read_text(encoding="utf-8")
    mod = parse_module(text)
    # print(mod)
    model, script_src = module_to_onnx(mod, opset=args.opset)

    onnx.save_model(model, args.output)
    print(f"Saved {args.output}")
    if args.save_script:
        pathlib.Path(args.save_script).write_text(script_src, encoding="utf-8")
        print(f"Wrote ONNX Script to {args.save_script}")

if __name__ == "__main__":
    main()

# python -m src.compile_onnx ./mlir/PoC_jax.mlir -o out/jax.onnx --save_script out/jax_onnx_script.py
# python -m src.compile_onnx ./mlir/PoC_torch.mlir -o out/torch.onnx --save_script out/torch_onnx_script.py