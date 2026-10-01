// swift-tools-version: 5.9
import PackageDescription

// Exact-class convolutions on Apple GPUs where mlx's Winograd path is lossy (M5 / NAX: its batched
// GEMM runs TF32). Shared by every package that used to carry its own WinogradFreeConv2d copy.
let package = Package(
    name: "mlx-exact-conv-swift",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "MLXExactConv", targets: ["MLXExactConv"])
    ],
    dependencies: [
        // Works on 0.31.x and 0.32.x (see ExactConv.swift); capped at the next unvalidated minor.
        .package(url: "https://github.com/ml-explore/mlx-swift.git", "0.31.4" ..< "0.33.0")
    ],
    targets: [
        .target(
            name: "MLXExactConv",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]),
        .testTarget(
            name: "MLXExactConvTests",
            dependencies: [
                "MLXExactConv",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ]),
    ]
)
