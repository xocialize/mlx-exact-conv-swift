// Exact-class convolutions on Apple GPUs.
//
// mlx's Metal conv2d runs a Winograd F(6x6,3x3) kernel for 3x3 / stride 1 / dilation 1 / groups 1
// convs with C % 32 == 0, O % 32 == 0, C + O >= 256 and N*H*W >= 4096. On NAX GPUs (M5) its batched
// GEMM runs TF32 (MLX_ENABLE_TF32 defaults on), so each conv loses precision: relL2 vs an exact
// reference ~6.4e-3 in fp32, ~7.4e-3 in fp16, ~5.8e-2 in bf16. The implicit-GEMM path is exact-class.
//
// mlx-swift 0.31.x: a 3x3 conv2d expressed as conv3d with kT = 1 took implicit GEMM (exact).
// mlx-swift 0.32.x (mlx#3785): small-kernel-depth conv3d with N = 1, depth padding 0 and channels
// % 16 == 0 is split into per-frame conv2d, so it reaches Winograd again. That also hits real
// small-kD conv3d (video-VAE causal convs).
//
// Both versions send conv3d with channels % 16 != 0 to `pad_and_slice_conv_3D_gpu` first, which pads
// channels and runs implicit GEMM. So one zero input channel (input and weight) keeps a conv exact on
// 0.31.x and 0.32.x, at ~+10% over the old kT = 1 route. Tracking a real opt-out: ml-explore/mlx#4595.
// The probes in MLXExactConvTests fail if a future mlx changes this dispatch.

import Foundation
import MLX
import MLXNN

/// How a conv inside mlx's Winograd window runs.
public enum ExactConvRoute: String, Sendable, CaseIterable {
    /// mlx's default Winograd kernel: fastest, lossy on NAX GPUs.
    case winograd
    /// Implicit-GEMM path: exact-class. Raw value "conv3d" is kept for old env overrides.
    case exact = "conv3d"
    /// Half-precision input upcast to fp32 for Winograd, result cast back (less loss than bf16).
    case fp32Winograd

    /// Reads an override such as `MYPKG_CONV_ROUTE=winograd|conv3d|exact|fp32Winograd`.
    public static func environment(_ name: String) -> ExactConvRoute? {
        guard let raw = getenv(name).map({ String(cString: $0) }) else { return nil }
        return raw == "exact" ? .exact : ExactConvRoute(rawValue: raw)
    }
}

public enum ExactConv {

    /// mlx's 2D Winograd predicate (mlx/backend/metal/conv.cpp, `dispatch_conv_2D_gpu`), NHWC input.
    public static func takesWinograd2D(
        input x: MLXArray, weight: MLXArray, stride: (Int, Int) = (1, 1),
        dilation: (Int, Int) = (1, 1), groups: Int = 1
    ) -> Bool {
        guard x.ndim == 4, groups == 1, stride == (1, 1), dilation == (1, 1),
            weight.dim(1) == 3, weight.dim(2) == 3
        else { return false }
        let (c, o) = (x.dim(3), weight.dim(0))
        return c % 32 == 0 && o % 32 == 0 && c + o >= 256 && x.dim(0) * x.dim(1) * x.dim(2) >= 4096
    }

    /// True when a conv3d would be decomposed into Winograd-eligible conv2d (mlx-swift 0.32.x) or is
    /// otherwise worth forcing onto implicit GEMM: groups 1 and channels already a multiple of 16.
    public static func needsChannelPad3D(input x: MLXArray, weight: MLXArray, groups: Int = 1) -> Bool {
        x.ndim == 5 && groups == 1 && x.dim(4) % 16 == 0
    }

    /// Appends one zero channel on the last (channel) axis. Use for inputs and OHWC / ODHWC weights.
    public static func appendingZeroChannel(_ a: MLXArray) -> MLXArray {
        var shape = a.shape
        shape[shape.count - 1] = 1
        return concatenated([a, MLXArray.zeros(shape, dtype: a.dtype)], axis: -1)
    }

    /// The kT = 1 conv3d weight with one zero input channel, as taken by `conv2d(...exactWeight:)`.
    public static func exactWeight2D(_ weight: MLXArray) -> MLXArray {
        appendingZeroChannel(weight.expandedDimensions(axis: 1))
    }

    /// A 2D conv (NHWC input, OHWC weight) that honours `route` for shapes in the Winograd window.
    /// Out-of-window shapes always take plain conv2d, which is already exact-class.
    public static func conv2d(
        _ x: MLXArray, weight: MLXArray, bias: MLXArray? = nil,
        stride: (Int, Int) = (1, 1), padding: (Int, Int) = (0, 0),
        dilation: (Int, Int) = (1, 1), groups: Int = 1,
        route: ExactConvRoute, exactWeight: MLXArray? = nil
    ) -> MLXArray {
        func plain() -> MLXArray {
            let y = MLX.conv2d(
                x, weight, stride: .init(stride), padding: .init(padding),
                dilation: .init(dilation), groups: groups)
            return bias.map { y + $0 } ?? y
        }
        guard route != .winograd,
            takesWinograd2D(input: x, weight: weight, stride: stride, dilation: dilation, groups: groups)
        else { return plain() }
        if route == .fp32Winograd {
            guard x.dtype != .float32 else { return plain() }
            var y = MLX.conv2d(
                x.asType(.float32), weight.asType(.float32), stride: .init(stride),
                padding: .init(padding), dilation: .init(dilation), groups: groups)
            if let bias { y = y + bias.asType(.float32) }
            return y.asType(x.dtype)
        }
        let w = exactWeight ?? exactWeight2D(weight)
        var y = MLX.conv3d(
            appendingZeroChannel(x.expandedDimensions(axis: 1)), w,
            stride: [1, 1, 1], padding: [0, padding.0, padding.1]
        ).squeezed(axis: 1)
        if let bias { y = y + bias }
        return y
    }

    /// A conv3d (NDHWC input, ODHWC weight) kept on implicit GEMM when 0.32.x would split it into
    /// Winograd conv2d. Pass a cached `appendingZeroChannel(weight)` as `exactWeight` when possible.
    public static func conv3d(
        _ x: MLXArray, weight: MLXArray, bias: MLXArray? = nil,
        stride: (Int, Int, Int) = (1, 1, 1), padding: (Int, Int, Int) = (0, 0, 0),
        dilation: (Int, Int, Int) = (1, 1, 1), groups: Int = 1,
        exact: Bool = true, exactWeight: MLXArray? = nil
    ) -> MLXArray {
        let usePad = exact && needsChannelPad3D(input: x, weight: weight, groups: groups)
        var y = MLX.conv3d(
            usePad ? appendingZeroChannel(x) : x,
            usePad ? (exactWeight ?? appendingZeroChannel(weight)) : weight,
            stride: [stride.0, stride.1, stride.2], padding: [padding.0, padding.1, padding.2],
            dilation: [dilation.0, dilation.1, dilation.2], groups: groups)
        if let bias { y = y + bias }
        return y
    }
}

/// Drop-in `Conv2d` that routes Winograd-window shapes. Parameter keys are unchanged (`weight`,
/// `bias`). The padded weight is rebuilt per call: mlx-swift updates parameters in place, so a cache
/// keyed on the array would go stale, and the pad is small next to the conv itself.
open class RoutedConv2d: Conv2d {
    public var route: ExactConvRoute

    public init(
        inputChannels: Int, outputChannels: Int, kernelSize: IntOrPair, stride: IntOrPair = 1,
        padding: IntOrPair = 0, dilation: IntOrPair = 1, groups: Int = 1, bias: Bool = true,
        route: ExactConvRoute = .exact
    ) {
        self.route = route
        super.init(
            inputChannels: inputChannels, outputChannels: outputChannels, kernelSize: kernelSize,
            stride: stride, padding: padding, dilation: dilation, groups: groups, bias: bias)
    }

    open override func callAsFunction(_ x: MLXArray) -> MLXArray {
        ExactConv.conv2d(
            x, weight: weight, bias: bias, stride: stride, padding: padding, dilation: dilation,
            groups: groups, route: route)
    }
}
