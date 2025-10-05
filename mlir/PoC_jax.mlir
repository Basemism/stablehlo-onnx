module @jit_simple_mlp attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x4xf32>, %arg1: tensor<4x8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8x3xf32>, %arg4: tensor<3xf32>) -> (tensor<2x3xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4xf32>, tensor<4x8xf32>) -> tensor<2x8xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<8xf32>) -> tensor<1x8xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<2x8xf32>
    %3 = stablehlo.add %0, %2 : tensor<2x8xf32>
    %4 = call @relu(%3) : (tensor<2x8xf32>) -> tensor<2x8xf32>
    %5 = stablehlo.dot_general %4, %arg3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x8xf32>, tensor<8x3xf32>) -> tensor<2x3xf32>
    %6 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<3xf32>) -> tensor<1x3xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<1x3xf32>) -> tensor<2x3xf32>
    %8 = stablehlo.add %5, %7 : tensor<2x3xf32>
    return %8 : tensor<2x3xf32>
  }
  func.func private @relu(%arg0: tensor<2x8xf32>) -> tensor<2x8xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x8xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<2x8xf32>
    return %1 : tensor<2x8xf32>
  }
}
