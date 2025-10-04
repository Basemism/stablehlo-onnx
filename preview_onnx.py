import onnx

model = onnx.load("out/jax.onnx")

print(f"The model is:\n{model}")
print("Model inputs:", [input.name for input in model.graph.input])
print("Model outputs:", [output.name for output in model.graph.output])
print("Model nodes:", [node.op_type for node in model.graph.node])