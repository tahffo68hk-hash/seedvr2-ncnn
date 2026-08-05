package com.seededge.runtime

object NativeRuntime {
    init { System.loadLibrary("seededge") }
    external fun nativeGetVersion(): String
    external fun nativeGetBuildInfo(): String
    external fun nativeRunSyntheticCpuTest(): String
    external fun nativeGetNcnnInfo(): String
    external fun nativeProbeVulkanSupport(): String
    external fun nativeRunMinimalVideoDemo(modelRoot: String, inputFramesRoot: String, outputRoot: String, configJson: String): String
    external fun nativeGetProgressJson(): String
    external fun nativeRequestCancel(): String
    external fun nativeInspectModel(modelRoot: String): String
    external fun nativeInspectCheckpoints(outputRoot: String): String
    external fun nativeRunDecoderOnly(modelRoot: String, checkpointRoot: String, outputRoot: String, configJson: String): String
    external fun nativeRunProgressSimulation(): String
    external fun nativeRunOpenMpThreadProbe(requestedThreads: Int): String
    external fun nativeRunDecoderStageBenchmark(modelRoot: String, outputRoot: String): String
    external fun nativeTestFastConvolution3D(): String
}
