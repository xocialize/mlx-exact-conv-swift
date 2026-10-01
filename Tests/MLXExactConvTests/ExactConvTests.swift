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
