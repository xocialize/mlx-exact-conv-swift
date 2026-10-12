// Exact-class convolutions on Apple GPUs.
//
// mlx's Metal conv2d runs a Winograd F(6x6,3x3) kernel for 3x3 / stride 1 / dilation 1 / groups 1
// convs with C % 32 == 0, O % 32 == 0, C + O >= 256 and N*H*W >= 4096. On NAX GPUs (M5) its batched
// GEMM runs TF32 (MLX_ENABLE_TF32 defaults on), so each conv loses precision: relL2 vs an exact
// reference ~6.4e-3 in fp32, ~7.4e-3 in fp16, ~5.8e-2 in bf16. The implicit-GEMM path is exact-class.
//
// mlx-swift 0.31.x: a 3x3 conv2d expressed as conv3d with kT = 1 took implicit GEMM (exact).
// mlx-swift 0.32.x (mlx#3785): small-kernel-depth conv3d with N = 1, depth padding 0, depth dilation 1
// and channels % 16 == 0 is split into per-frame conv2d, so it reaches Winograd again. That also hits
// real small-kD conv3d (video-VAE causal convs). Tracking a real opt-out: ml-explore/mlx#4595.
//
// mlx's channel-pad helpers must not run (mlx 0.32.2, AB-T-0214). `pad_and_slice_conv_3D_gpu` (conv3d,
// channels not a multiple of 16) and `pad_in_channels_conv_2D_gpu` (stride-1 conv2d, 5+ input channels
// not a multiple of 16) build their zero fill value as a local array, bind it to the fill kernel and
// never register it as a temporary, so it is freed before the command buffer runs. Once mlx's buffer
// cache is at its limit (an engine caps it at 2 GiB) the freed buffer is released to the OS at once,
// and the GPU faults on it: "[METAL] Command buffer execution failed: Invalid Resource
// (kIOGPUCommandBufferCallbackErrorInvalidResource)", which aborts the process. The 0.1.0 exact path
// appended one zero channel precisely to reach `pad_and_slice_conv_3D_gpu`. Now no conv that goes
// through ExactConv reaches either helper:
//   - exact path, kernel depth 1 (conv2d, kT = 1 conv3d): depth kernel dilation 2. It's a no-op on a
//     1-tap depth kernel, but outside 0.32.x's per-frame split, so the conv stays on 3D implicit GEMM
//     with its own channels: nothing is padded, and it costs less than the old pad;
//   - exact path, kernel depth > 1: 16 zero channels appended in the graph. C % 32 == 16 is outside the
//     Winograd window, and the channels stay a multiple of 16, so mlx pads nothing itself;
//   - any other shape a helper would take: its channels are padded to a multiple of 16 in the graph,
//     where the zeros are real graph inputs that mlx keeps alive.
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
        return winograd2D(c: x.dim(3), o: weight.dim(0), pixels: x.dim(0) * x.dim(1) * x.dim(2))
    }

    /// True when mlx-swift 0.32.x would split this conv3d into per-frame conv2d that lands in the
    /// Winograd window (mlx#3785: groups 1, N 1, kD ≤ 7, depth stride/dilation 1, depth padding 0),
    /// so it needs the exact path. Shapes outside the window are already exact and are left alone
    /// (e.g. a 1024² 96-channel decoder stage: C + O < 256).
    public static func takesWinograd3D(
        input x: MLXArray, weight: MLXArray, stride: (Int, Int, Int) = (1, 1, 1),
        padding: (Int, Int, Int) = (0, 0, 0), dilation: (Int, Int, Int) = (1, 1, 1), groups: Int = 1
    ) -> Bool {
        splitsPerFrame3D(
            input: x, weight: weight, stride: stride, padding: padding, dilation: dilation,
            groups: groups)
            && winograd2D(c: x.dim(4), o: weight.dim(0), pixels: framePixels3D(input: x, weight: weight))
    }

    /// True when mlx 0.32.x would run this conv3d through `pad_and_slice_conv_3D_gpu` (groups 1, input
    /// channels not a multiple of 16, or more than 16 output channels not a multiple of 16).
    public static func takesPadHelper3D(input x: MLXArray, weight: MLXArray, groups: Int = 1) -> Bool {
        guard x.ndim == 5, groups == 1 else { return false }
        return !channelsAligned(c: x.dim(4), o: weight.dim(0))
    }

    /// True when mlx 0.32.x would run this conv2d through `pad_in_channels_conv_2D_gpu`: groups 1,
    /// stride 1, a kernel of 9+ taps, N·OH·OW ≥ 256, aligned output channels, and input channels that
    /// are neither ≤ 4 nor a multiple of 16.
    public static func takesPadHelper2D(
        input x: MLXArray, weight: MLXArray, stride: (Int, Int) = (1, 1),
        padding: (Int, Int) = (0, 0), dilation: (Int, Int) = (1, 1), groups: Int = 1
    ) -> Bool {
        guard x.ndim == 4, groups == 1, stride == (1, 1), weight.dim(1) * weight.dim(2) >= 9
        else { return false }
        let (c, o) = (x.dim(3), weight.dim(0))
        let oh = x.dim(1) + 2 * padding.0 - dilation.0 * (weight.dim(1) - 1)
        let ow = x.dim(2) + 2 * padding.1 - dilation.1 * (weight.dim(2) - 1)
        return c > 4 && c % 16 != 0 && (o <= 16 || o % 16 == 0) && x.dim(0) * oh * ow >= 256
    }

    /// The non-channel half of `takesWinograd3D`: a 3x3 conv3d that 0.32.x splits into per-frame
    /// conv2d once its channels are a multiple of 16.
    static func splitsPerFrame3D(
        input x: MLXArray, weight: MLXArray, stride: (Int, Int, Int), padding: (Int, Int, Int),
        dilation: (Int, Int, Int), groups: Int
    ) -> Bool {
        x.ndim == 5 && groups == 1 && x.dim(0) == 1 && weight.dim(1) <= 7
            && stride.0 == 1 && dilation.0 == 1 && padding.0 == 0
            && weight.dim(2) == 3 && weight.dim(3) == 3 && stride.1 == 1 && stride.2 == 1
            && dilation.1 == 1 && dilation.2 == 1
    }

    static func framePixels3D(input x: MLXArray, weight: MLXArray) -> Int {
        (x.dim(1) - weight.dim(1) + 1) * x.dim(2) * x.dim(3)
    }

    static func winograd2D(c: Int, o: Int, pixels: Int) -> Bool {
        c % 32 == 0 && o % 32 == 0 && c + o >= 256 && pixels >= 4096
    }

    /// mlx's "specialized channels" test: anything else takes a pad helper (groups 1).
    static func channelsAligned(c: Int, o: Int) -> Bool {
        c % 16 == 0 && (o <= 16 || o % 16 == 0)
    }

    static func roundUp16(_ n: Int) -> Int { (n + 15) / 16 * 16 }

    /// Appends one zero channel on the last (channel) axis. Use for inputs and OHWC / ODHWC weights.
    public static func appendingZeroChannel(_ a: MLXArray) -> MLXArray {
        appendingZeroChannels(a, 1)
    }

    /// Appends `count` zero channels on the last (channel) axis.
    public static func appendingZeroChannels(_ a: MLXArray, _ count: Int) -> MLXArray {
        guard count > 0 else { return a }
        var shape = a.shape
        shape[shape.count - 1] = count
        return concatenated([a, MLXArray.zeros(shape, dtype: a.dtype)], axis: -1)
    }

    /// Appends `count` zero filters (output channels) on the first axis of an OHWC / ODHWC weight.
    static func appendingZeroFilters(_ w: MLXArray, _ count: Int) -> MLXArray {
        guard count > 0 else { return w }
        var shape = w.shape
        shape[0] = count
        return concatenated([w, MLXArray.zeros(shape, dtype: w.dtype)], axis: 0)
    }

    /// The kT = 1 conv3d weight the exact conv2d path takes, as accepted by `conv2d(...exactWeight:)`.
    public static func exactWeight2D(_ weight: MLXArray) -> MLXArray {
        weight.expandedDimensions(axis: 1)
    }

    /// A 2D conv (NHWC input, OHWC weight) that honours `route` for shapes in the Winograd window.
    /// Out-of-window shapes take plain conv2d, which is already exact-class; a shape mlx would send
    /// through its channel-pad helper gets its input channels padded here instead.
    ///
    /// `exactWeight` is `exactWeight2D(weight)`. A weight in 0.1.0's layout (one extra zero input
    /// channel) is still accepted.
    public static func conv2d(
        _ x: MLXArray, weight: MLXArray, bias: MLXArray? = nil,
        stride: (Int, Int) = (1, 1), padding: (Int, Int) = (0, 0),
        dilation: (Int, Int) = (1, 1), groups: Int = 1,
        route: ExactConvRoute, exactWeight: MLXArray? = nil
    ) -> MLXArray {
        func plain() -> MLXArray {
            var (xin, w) = (x, weight)
            if takesPadHelper2D(
                input: x, weight: weight, stride: stride, padding: padding, dilation: dilation,
                groups: groups)
            {
                var c = roundUp16(x.dim(3))
                // Padding can land a conv in the Winograd window; 16 more channels keep it out.
                if route != .winograd, dilation == (1, 1), weight.dim(1) == 3, weight.dim(2) == 3,
                    winograd2D(c: c, o: weight.dim(0), pixels: x.dim(0) * x.dim(1) * x.dim(2))
                {
                    c += 16
                }
                xin = appendingZeroChannels(x, c - x.dim(3))
                w = appendingZeroChannels(weight, c - weight.dim(3))
            }
            let y = MLX.conv2d(
                xin, w, stride: .init(stride), padding: .init(padding),
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
        var w = exactWeight ?? exactWeight2D(weight)
        if w.dim(-1) == x.dim(-1) + 1 { w = w[.ellipsis, 0 ..< x.dim(-1)] }
        var y = MLX.conv3d(
            x.expandedDimensions(axis: 1), w,
            stride: [1, 1, 1], padding: [0, padding.0, padding.1], dilation: [2, 1, 1]
        ).squeezed(axis: 1)
        if let bias { y = y + bias }
        return y
    }

    /// A conv3d (NDHWC input, ODHWC weight) kept on implicit GEMM when 0.32.x would split it into
    /// Winograd conv2d (`takesWinograd3D`); every other shape runs plain conv3d, with its channels
    /// padded here when mlx would otherwise pad them in its channel-pad helper. Cost of the exact path
    /// at a causal-VAE shape on 0.32.3 (256², 384 → 384 ch, bf16, kD = 3): 38 ms vs 6.6 ms lossy
    /// (0.1.0's one-channel pad: 39 ms). The kD = 1 / conv2d exact path: 12.0 ms (0.1.0: 12.9 ms).
    ///
    /// `exactWeight` is ignored: the padded weight is built from `weight`.
    public static func conv3d(
        _ x: MLXArray, weight: MLXArray, bias: MLXArray? = nil,
        stride: (Int, Int, Int) = (1, 1, 1), padding: (Int, Int, Int) = (0, 0, 0),
        dilation: (Int, Int, Int) = (1, 1, 1), groups: Int = 1,
        exact: Bool = true, exactWeight: MLXArray? = nil
    ) -> MLXArray {
        var (xin, w) = (x, weight)
        var depthDilation = dilation.0
        var sliceOutput: Int? = nil
        if exact
            && takesWinograd3D(
                input: x, weight: weight, stride: stride, padding: padding, dilation: dilation,
                groups: groups)
        {
            if weight.dim(1) == 1 {
                depthDilation = 2
            } else {
                xin = appendingZeroChannels(x, 16)
                w = appendingZeroChannels(weight, 16)
            }
        } else if takesPadHelper3D(input: x, weight: weight, groups: groups) {
            let (c, o) = (x.dim(4), weight.dim(0))
            var cp = roundUp16(c)
            let op = o <= 16 ? o : roundUp16(o)
            // Padding can land a conv in the Winograd window; 16 more channels keep it out.
            if exact,
                splitsPerFrame3D(
                    input: x, weight: weight, stride: stride, padding: padding, dilation: dilation,
                    groups: groups),
                winograd2D(c: cp, o: op, pixels: framePixels3D(input: x, weight: weight))
            {
                cp += 16
            }
            xin = appendingZeroChannels(x, cp - c)
            w = appendingZeroFilters(appendingZeroChannels(weight, cp - c), op - o)
            if op > o { sliceOutput = o }
        }
        var y = MLX.conv3d(
            xin, w,
            stride: [stride.0, stride.1, stride.2], padding: [padding.0, padding.1, padding.2],
            dilation: [depthDilation, dilation.1, dilation.2], groups: groups)
        if let o = sliceOutput { y = y[.ellipsis, 0 ..< o] }
        if let bias { y = y + bias }
        return y
    }
}

/// Drop-in `Conv2d` that routes Winograd-window shapes. Parameter keys are unchanged (`weight`,
/// `bias`). The exact path's weight is rebuilt per call: mlx-swift updates parameters in place, so a
/// cache keyed on the array would go stale, and the reshape is free next to the conv itself.
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
