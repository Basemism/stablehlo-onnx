# StableHLO to ONNX Proof of Concept

This is a PoC that demonstrates a prototype translator that converts machine learning models expressed in the StableHLO dialect (MLIR) into the ONNX format.

## Getting Started

1. **Create a virtual environment**  
    Python 3.11 is recommended.

    Using `venv`:
    ```bash
    python3.11 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

    Or with `conda`:
    ```bash
    conda create -n stablehlo-onnx-env python=3.11
    conda activate stablehlo-onnx-env
    ```

2. **Clone the repository**
    ```bash
    git clone https://github.com/Basemism/stablehlo-onnx.git
    ```

3. **Navigate to the project directory**
    ```bash
    cd stablehlo-onnx
    ```

4. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the translator, use the following command:
```bash
python -m src.compile_onnx <input_mlir_file> -o <output_onnx_file> \
       --save_script <output_script_file> --opset <opset_version>
```

### Example Commands

```bash
python -m src.compile_onnx ./mlir/PoC_jax.mlir -o out/jax.onnx --save_script out/jax_onnx_script.py
``` 

## Supported Ops
| StableHLO op | ONNX op | Notes |
|---------------|----------|-------|
| `stablehlo.add` | `Add` | Implicit broadcasting supported |
| `stablehlo.maximum` | `Max` | Max pooling operation |
| `stablehlo.multiply` | `Mul` | Basic scalar or tensor multiply |
| `stablehlo.dot_general` | `MatMul` | 2-D case `[1] x [0]` |
| `stablehlo.broadcast_in_dim` | `Identity` | Name-binding only |
| `stablehlo.constant` | `Constant` | Parses numeric payload from `dense<…>` |
| `call @relu` | `Relu` | Direct mapping from private `@relu` func |
| `return` | `return` | Preserves SSA result name |
