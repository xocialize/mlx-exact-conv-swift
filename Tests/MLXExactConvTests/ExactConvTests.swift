import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import MLXExactConv

/// GPU results compared with a CPU reference. The exact bound sits far below Winograd/TF32 error
/// (~6.4e-3 fp32) and well above fp32 accumulation noise (~2e-6), so these fail if mlx's dispatch
/// moves the exact path back onto a lossy kernel. On non-NAX GPUs every route is exact; the routes
/// are still checked for correctness.
final class ExactConvTests: XCTestCase {
    let exactBound: Float = 1e-4

    func rel(_ a: MLXArray, _ b: MLXArray) -> Float {
        let d = (a.asType(.float32) - b.asType(.float32))
        return (sqrt((d * d).sum()) / sqrt((b.asType(.float32) * b.asType(.float32)).sum())).item(Float.self)
    }

    func cpu(_ body: () -> MLXArray) -> MLXArray {
        Device.withDefaultDevice(Device(.cpu)) {
            let y = body()
            eval(y)
            return y
        }
    }

    func testWinogradPredicate() {
        let x = MLXArray.zeros([1, 64, 64, 128])
        XCTAssertTrue(ExactConv.takesWinograd2D(input: x, weight: MLXArray.zeros([128, 3, 3, 128])))
        XCTAssertFalse(ExactConv.takesWinograd2D(input: x, weight: MLXArray.zeros([128, 1, 1, 128])))
        XCTAssertFalse(  // C + O < 256
            ExactConv.takesWinograd2D(
                input: MLXArray.zeros([1, 64, 64, 64]), weight: MLXArray.zeros([64, 3, 3, 64])))
        XCTAssertFalse(  // N*H*W < 4096
            ExactConv.takesWinograd2D(
                input: MLXArray.zeros([1, 32, 32, 128]), weight: MLXArray.zeros([128, 3, 3, 128])))
        XCTAssertFalse(
            ExactConv.takesWinograd2D(
                input: x, weight: MLXArray.zeros([128, 3, 3, 128]), stride: (2, 2)))
    }

    func testExactConv2dMatchesCPU() {
        MLXRandom.seed(1)
        let x = MLXRandom.normal([1, 128, 128, 256])
        let w = MLXRandom.normal([256, 3, 3, 256]) * 0.02
        let b = MLXRandom.normal([256]) * 0.1
        let ref = cpu { MLX.conv2d(x, w, padding: 1) + b }
        let exact = ExactConv.conv2d(x, weight: w, bias: b, padding: (1, 1), route: .exact)
        let wino = ExactConv.conv2d(x, weight: w, bias: b, padding: (1, 1), route: .winograd)
        eval(exact, wino)
        let (re, rw) = (rel(exact, ref), rel(wino, ref))
        print(String(format: "EXACTCONV conv2d exact %.2e  winograd %.2e", re, rw))
        XCTAssertEqual(exact.shape, ref.shape)
        XCTAssertLessThan(re, exactBound)
        XCTAssertLessThan(rw, 2e-2)
    }

    func testExactConv2dHalfAndPadding() {
        MLXRandom.seed(2)
        let x = MLXRandom.normal([1, 96, 80, 128]).asType(.float16)
        let w = (MLXRandom.normal([192, 3, 3, 128]) * 0.03).asType(.float16)
        let ref = cpu { MLX.conv2d(x.asType(.float32), w.asType(.float32), padding: 1) }
        let exact = ExactConv.conv2d(x, weight: w, padding: (1, 1), route: .exact)
        let up = ExactConv.conv2d(x, weight: w, padding: (1, 1), route: .fp32Winograd)
        eval(exact, up)
        print(String(format: "EXACTCONV conv2d fp16 exact %.2e  fp32Winograd %.2e", rel(exact, ref), rel(up, ref)))
        XCTAssertEqual(exact.dtype, .float16)
        XCTAssertEqual(up.dtype, .float16)
        XCTAssertLessThan(rel(exact, ref), 1e-3)  // fp16 storage floor ~2e-4
    }

    func testRoutedConv2dModule() throws {
        MLXRandom.seed(3)
        let conv = RoutedConv2d(inputChannels: 128, outputChannels: 128, kernelSize: 3, padding: 1)
        XCTAssertEqual(Set(conv.parameters().flattened().map(\.0)), ["weight", "bias"])
        let x = MLXRandom.normal([1, 64, 64, 128])
        let ref = cpu { MLX.conv2d(x, conv.weight, padding: 1) + conv.bias! }
        let y = conv(x)
        eval(y)
        XCTAssertLessThan(rel(y, ref), exactBound)

        // Parameter updates (in place in mlx-swift) are picked up.
        let w2 = MLXRandom.normal([128, 3, 3, 128]) * 0.05
        try conv.update(parameters: ModuleParameters.unflattened(["weight": w2]), verify: .none)
        let ref2 = cpu { MLX.conv2d(x, w2, padding: 1) + conv.bias! }
        let y2 = conv(x)
        eval(y2)
        XCTAssertLessThan(rel(y2, ref2), exactBound)
    }

    func testOutOfWindowUnchanged() {
        MLXRandom.seed(4)
        let x = MLXRandom.normal([1, 32, 32, 64])
        let w = MLXRandom.normal([64, 3, 3, 64]) * 0.05
        let a = ExactConv.conv2d(x, weight: w, padding: (1, 1), route: .exact)
        let b = MLX.conv2d(x, w, padding: 1)
        XCTAssertEqual(rel(a, b), 0)
    }

    func testExactConv3dSmallDepth() {
        MLXRandom.seed(5)
        let x = MLXRandom.normal([1, 5, 64, 64, 128])
        let w = MLXRandom.normal([128, 3, 3, 3, 128]) * 0.02
        let ref = cpu { MLX.conv3d(x, w, padding: [0, 1, 1]) }
        let exact = ExactConv.conv3d(x, weight: w, padding: (0, 1, 1))
        let raw = ExactConv.conv3d(x, weight: w, padding: (0, 1, 1), exact: false)
        eval(exact, raw)
        print(String(format: "EXACTCONV conv3d kD=3 exact %.2e  raw %.2e", rel(exact, ref), rel(raw, ref)))
        XCTAssertEqual(exact.shape, ref.shape)
        XCTAssertLessThan(rel(exact, ref), exactBound)
    }

    func testWinograd3DPredicate() {
        let w = MLXArray.zeros([128, 3, 3, 3, 128])
        XCTAssertTrue(ExactConv.takesWinograd3D(input: MLXArray.zeros([1, 3, 64, 64, 128]), weight: w))
        XCTAssertFalse(  // N = 2: mlx keeps the 3D implicit GEMM
            ExactConv.takesWinograd3D(input: MLXArray.zeros([2, 3, 64, 64, 128]), weight: w))
        XCTAssertFalse(  // depth padding
            ExactConv.takesWinograd3D(
                input: MLXArray.zeros([1, 3, 64, 64, 128]), weight: w, padding: (1, 0, 0)))
        XCTAssertFalse(  // C + O < 256: per-frame conv2d is already exact
            ExactConv.takesWinograd3D(
                input: MLXArray.zeros([1, 3, 256, 256, 96]), weight: MLXArray.zeros([96, 3, 3, 3, 96])))
    }

    func testConv3dOutsideWindowIsPlain() {
        MLXRandom.seed(6)
        let x = MLXRandom.normal([1, 3, 64, 64, 96])
        let w = MLXRandom.normal([96, 3, 3, 3, 96]) * 0.02
        let a = ExactConv.conv3d(x, weight: w, padding: (0, 1, 1))
        let b = MLX.conv3d(x, w, padding: [0, 1, 1])
        XCTAssertEqual(rel(a, b), 0)
    }

    func testExactConv3dDepthOne() {
        MLXRandom.seed(7)
        let x = MLXRandom.normal([1, 2, 64, 64, 128])
        let w = MLXRandom.normal([128, 1, 3, 3, 128]) * 0.02
        XCTAssertTrue(ExactConv.takesWinograd3D(input: x, weight: w))
        let ref = cpu { MLX.conv3d(x, w, padding: [0, 1, 1]) }
        let exact = ExactConv.conv3d(x, weight: w, padding: (0, 1, 1))
        eval(exact)
        print(String(format: "EXACTCONV conv3d kD=1 exact %.2e", rel(exact, ref)))
        XCTAssertEqual(exact.shape, ref.shape)
        XCTAssertLessThan(rel(exact, ref), exactBound)
    }

    func testLegacyExactWeightLayout() {
        MLXRandom.seed(8)
        let x = MLXRandom.normal([1, 64, 64, 128])
        let w = MLXRandom.normal([128, 3, 3, 128]) * 0.02
        let legacy = ExactConv.appendingZeroChannel(w.expandedDimensions(axis: 1))  // 0.1.0's layout
        let a = ExactConv.conv2d(x, weight: w, padding: (1, 1), route: .exact)
        let b = ExactConv.conv2d(x, weight: w, padding: (1, 1), route: .exact, exactWeight: legacy)
        let c = ExactConv.conv2d(
            x, weight: w, padding: (1, 1), route: .exact, exactWeight: ExactConv.exactWeight2D(w))
        XCTAssertEqual(rel(b, a), 0)
        XCTAssertEqual(rel(c, a), 0)
    }

    func testPadHelperPredicates() {
        let x17 = MLXArray.zeros([1, 2, 32, 32, 17])
        XCTAssertTrue(ExactConv.takesPadHelper3D(input: x17, weight: MLXArray.zeros([32, 1, 3, 3, 17])))
        XCTAssertTrue(  // O > 16, not a multiple of 16
            ExactConv.takesPadHelper3D(
                input: MLXArray.zeros([1, 2, 32, 32, 32]), weight: MLXArray.zeros([24, 1, 3, 3, 32])))
        XCTAssertFalse(  // O ≤ 16 is fine
            ExactConv.takesPadHelper3D(
                input: MLXArray.zeros([1, 2, 32, 32, 32]), weight: MLXArray.zeros([3, 1, 3, 3, 32])))
        XCTAssertFalse(
            ExactConv.takesPadHelper3D(input: x17, weight: MLXArray.zeros([32, 1, 3, 3, 17]), groups: 17))

        let x24 = MLXArray.zeros([1, 32, 32, 24])
        let w24 = MLXArray.zeros([32, 3, 3, 24])
        XCTAssertTrue(ExactConv.takesPadHelper2D(input: x24, weight: w24, padding: (1, 1)))
        XCTAssertFalse(ExactConv.takesPadHelper2D(input: x24, weight: w24, stride: (2, 2), padding: (1, 1)))
        XCTAssertFalse(  // C ≤ 4 takes the specialized kernel as is
            ExactConv.takesPadHelper2D(
                input: MLXArray.zeros([1, 32, 32, 3]), weight: MLXArray.zeros([32, 3, 3, 3]), padding: (1, 1)))
        XCTAssertFalse(  // 1x1 kernel
            ExactConv.takesPadHelper2D(input: x24, weight: MLXArray.zeros([32, 1, 1, 24])))
        XCTAssertFalse(  // N·OH·OW < 256
            ExactConv.takesPadHelper2D(
                input: MLXArray.zeros([1, 8, 8, 24]), weight: w24, padding: (1, 1)))
        XCTAssertFalse(  // unaligned output channels: mlx takes the general kernel, no pad
            ExactConv.takesPadHelper2D(input: x24, weight: MLXArray.zeros([24, 3, 3, 24]), padding: (1, 1)))
    }

    func testPadHelperShapesMatchCPU() {
        MLXRandom.seed(9)
        func check(_ name: String, _ ref: MLXArray, _ y: MLXArray) {
            eval(y)
            print(String(format: "EXACTCONV %@ %.2e", name, rel(y, ref)))
            XCTAssertEqual(y.shape, ref.shape, name)
            XCTAssertLessThan(rel(y, ref), exactBound, name)
        }
        for kd in [1, 3] {
            let x = MLXRandom.normal([1, kd + 1, 48, 48, 17])
            let w = MLXRandom.normal([32, kd, 3, 3, 17]) * 0.05
            check("conv3d C=17 kD=\(kd)", cpu { MLX.conv3d(x, w, padding: [0, 1, 1]) },
                  ExactConv.conv3d(x, weight: w, padding: (0, 1, 1)))
        }
        let x32 = MLXRandom.normal([1, 2, 48, 48, 32])
        let w24 = MLXRandom.normal([24, 1, 3, 3, 32]) * 0.05
        check("conv3d O=24", cpu { MLX.conv3d(x32, w24, padding: [0, 1, 1]) },
              ExactConv.conv3d(x32, weight: w24, padding: (0, 1, 1)))
        // Padded to 128 channels this conv3d would split into Winograd conv2d; it must stay exact.
        let x120 = MLXRandom.normal([1, 1, 64, 64, 120])
        let w120 = MLXRandom.normal([128, 1, 3, 3, 120]) * 0.02
        check("conv3d C=120 O=128", cpu { MLX.conv3d(x120, w120, padding: [0, 1, 1]) },
              ExactConv.conv3d(x120, weight: w120, padding: (0, 1, 1)))

        let x = MLXRandom.normal([1, 48, 48, 24])
        let w = MLXRandom.normal([32, 3, 3, 24]) * 0.05
        let b = MLXRandom.normal([32]) * 0.1
        check("conv2d C=24", cpu { MLX.conv2d(x, w, padding: 1) + b },
              ExactConv.conv2d(x, weight: w, bias: b, padding: (1, 1), route: .exact))
        let xb = MLXRandom.normal([1, 64, 64, 120])
        let wb = MLXRandom.normal([128, 3, 3, 120]) * 0.02
        check("conv2d C=120 O=128", cpu { MLX.conv2d(xb, wb, padding: 1) },
              ExactConv.conv2d(xb, weight: wb, padding: (1, 1), route: .exact))
    }

    /// The AB-T-0214 regression. mlx's channel-pad helpers free their scalar fill value before the
    /// command buffer runs; with the buffer cache at its limit the freed buffer goes back to the OS at
    /// once, and the GPU faults ("Invalid Resource"), which aborts this process. A cache limit of 0
    /// releases every freed buffer, and holding 20k small arrays fills the allocator's small-buffer
    /// heap, so the fill value is a standalone MTLBuffer: on 0.1.0 the first exact conv crashes.
    func testNoConvFreesABufferInFlight() {
        let previous = Memory.cacheLimit
        Memory.cacheLimit = 0
        defer { Memory.cacheLimit = previous }
        let small = (0 ..< 20_000).map { MLXArray(Float($0)) }
        eval(small)
        MLXRandom.seed(10)
        withExtendedLifetime(small) {
            for _ in 0 ..< 3 {
                var ys: [MLXArray] = []
                let x2 = MLXRandom.normal([1, 64, 64, 128])
                ys.append(ExactConv.conv2d(
                    x2, weight: MLXRandom.normal([128, 3, 3, 128]), padding: (1, 1), route: .exact))
                let x3 = MLXRandom.normal([1, 3, 64, 64, 128])
                ys.append(ExactConv.conv3d(x3, weight: MLXRandom.normal([128, 1, 3, 3, 128]), padding: (0, 1, 1)))
                ys.append(ExactConv.conv3d(x3, weight: MLXRandom.normal([128, 3, 3, 3, 128]), padding: (0, 1, 1)))
                ys.append(ExactConv.conv3d(
                    MLXRandom.normal([1, 2, 48, 48, 17]), weight: MLXRandom.normal([32, 1, 3, 3, 17]),
                    padding: (0, 1, 1)))
                ys.append(ExactConv.conv2d(
                    MLXRandom.normal([1, 48, 48, 24]), weight: MLXRandom.normal([32, 3, 3, 24]),
                    padding: (1, 1), route: .exact))
                eval(ys)
            }
        }
    }

    /// Timings for the doc comment (EXACTCONV_BENCH=1): exact routes at causal-VAE shapes.
    func testBenchExactRoutes() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["EXACTCONV_BENCH"] == "1", "EXACTCONV_BENCH=1")
        func ms(_ f: () -> MLXArray) -> Double {
            for _ in 0 ..< 3 { eval(f()) }
            let t = Date()
            for _ in 0 ..< 10 { eval(f()) }
            return -t.timeIntervalSinceNow * 100
        }
        let x = MLXRandom.normal([1, 256, 256, 384]).asType(.bfloat16)
        let w = (MLXRandom.normal([384, 3, 3, 384]) * 0.02).asType(.bfloat16)
        let x3 = MLXRandom.normal([1, 3, 256, 256, 384]).asType(.bfloat16)
        let w3 = (MLXRandom.normal([384, 3, 3, 3, 384]) * 0.02).asType(.bfloat16)
        eval(x, w, x3, w3)
        print(String(
            format: "EXACTCONV bench 256² 384→384 bf16: conv2d exact %.1f ms, winograd %.1f ms; "
                + "conv3d kD=3 exact %.1f ms, lossy %.1f ms",
            ms { ExactConv.conv2d(x, weight: w, padding: (1, 1), route: .exact) },
            ms { ExactConv.conv2d(x, weight: w, padding: (1, 1), route: .winograd) },
            ms { ExactConv.conv3d(x3, weight: w3, padding: (0, 1, 1)) },
            ms { ExactConv.conv3d(x3, weight: w3, padding: (0, 1, 1), exact: false) }))
    }

    func testEnvironmentParsing() {
        setenv("MLXEXACTCONV_TEST_ROUTE", "exact", 1)
        XCTAssertEqual(ExactConvRoute.environment("MLXEXACTCONV_TEST_ROUTE"), .exact)
        setenv("MLXEXACTCONV_TEST_ROUTE", "conv3d", 1)
        XCTAssertEqual(ExactConvRoute.environment("MLXEXACTCONV_TEST_ROUTE"), .exact)
        setenv("MLXEXACTCONV_TEST_ROUTE", "winograd", 1)
        XCTAssertEqual(ExactConvRoute.environment("MLXEXACTCONV_TEST_ROUTE"), .winograd)
        unsetenv("MLXEXACTCONV_TEST_ROUTE")
        XCTAssertNil(ExactConvRoute.environment("MLXEXACTCONV_TEST_ROUTE"))
    }
}
