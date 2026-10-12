# mlx-exact-conv-swift

`MLXExactConv` keeps MLX convolutions exact on Apple GPUs where mlx's Winograd conv2d is lossy
(M5 / NAX: its batched GEMM runs TF32, relL2 ~6.4e-3 per conv in fp32).

- `ExactConv.conv2d(_:weight:bias:...route:)`: routes 3×3 convs inside mlx's Winograd window.
- `ExactConv.conv3d(_:weight:...)`: keeps small-kernel-depth conv3d on implicit GEMM (mlx-swift
  0.32.x splits it into Winograd conv2d, ml-explore/mlx#3785).
- `RoutedConv2d`: drop-in `Conv2d` subclass with the same parameter keys.

The exact path keeps the conv on mlx's 3D implicit GEMM: a kT = 1 conv3d with depth kernel dilation
2 (a no-op on one tap that 0.32.x doesn't split per frame), or, for kernel depth > 1, 16 zero
channels appended in the graph.

**Never mlx's channel-pad helpers (0.1.1, AB-T-0214).** In mlx 0.32.2, `pad_and_slice_conv_3D_gpu`
(conv3d with channels not a multiple of 16) and `pad_in_channels_conv_2D_gpu` (stride-1 conv2d with
5+ input channels not a multiple of 16) free their zero fill value before the command buffer that
reads it has run. With mlx's buffer cache at its limit the buffer is released at once and the GPU
faults: `[METAL] Command buffer execution failed: Invalid Resource
(kIOGPUCommandBufferCallbackErrorInvalidResource)`, a process abort. 0.1.0's exact path padded one
zero channel to reach the first of them, so every exact conv could trip it. No conv routed through
`ExactConv` reaches either helper now: shapes that would get their channels padded to a multiple of
16 in the graph instead. `testNoConvFreesABufferInFlight` aborts on 0.1.0.
Upstream opt-out request: ml-explore/mlx#4595. `swift test` / `xcodebuild test` run GPU-vs-CPU
probes that fail if a future mlx moves the exact path back onto a lossy kernel.
