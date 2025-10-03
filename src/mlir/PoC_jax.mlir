module @jit_toy attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>, %arg2: tensor<4x3xf32>) -> (tensor<2x3xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<2x4xf32>
    %1 = call @relu(%0) : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %2 = stablehlo.dot_general %1, %arg2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4xf32>, tensor<4x3xf32>) -> tensor<2x3xf32>
    return %2 : tensor<2x3xf32>
  }
  func.func private @relu(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<2x4xf32>
    return %1 : tensor<2x4xf32>
  }
}
