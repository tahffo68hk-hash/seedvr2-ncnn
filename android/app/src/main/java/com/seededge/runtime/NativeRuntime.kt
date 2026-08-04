package com.seededge.runtime

object NativeRuntime {
    init { System.loadLibrary("seededge") }
    external fun nativeGetVersion(): String
    external fun nativeGetBuildInfo(): String
    external fun nativeRunSyntheticCpuTest(): String
    external fun nativeGetNcnnInfo(): String
    external fun nativeProbeVulkanSupport(): String
}
