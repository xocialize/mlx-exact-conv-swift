# mlx-exact-conv-swift

`MLXExactConv` keeps MLX convolutions exact on Apple GPUs where mlx's Winograd conv2d is lossy
(M5 / NAX: its batched GEMM runs TF32, relL2 ~6.4e-3 per conv in fp32).

- `ExactConv.conv2d(_:weight:bias:...route:)`: routes 3×3 convs inside mlx's Winograd window.
- `ExactConv.conv3d(_:weight:...)`: keeps small-kernel-depth conv3d on implicit GEMM (mlx-swift
  0.32.x splits it into Winograd conv2d, ml-explore/mlx#3785).
- `RoutedConv2d`: drop-in `Conv2d` subclass with the same parameter keys.

The exact path pads one zero input channel, which sends the conv down mlx's pad-and-slice
implicit-GEMM path on both 0.31.x and 0.32.x (~+10% over the old kT = 1 conv3d route).
Upstream opt-out request: ml-explore/mlx#4595. `swift test` / `xcodebuild test` run GPU-vs-CPU
probes that fail if a future mlx moves the exact path back onto a lossy kernel.
