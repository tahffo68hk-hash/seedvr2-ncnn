import json
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class ScaffoldContractTests(unittest.TestCase):
    def test_gradle_project_is_arm64_only_and_pinned(self):
        app = (ROOT / "app/build.gradle.kts").read_text(encoding="utf-8")
        self.assertIn('abiFilters += listOf("arm64-v8a")', app)
        self.assertIn('cmake.version = "3.22.1"', app)
        self.assertIn('ndkVersion = "26.3.11579264"', app)
        self.assertNotRegex(app, r'armeabi|x86')

    def test_required_jni_contract_exists(self):
        kotlin = (ROOT / "app/src/main/java/com/seededge/runtime/NativeRuntime.kt").read_text(encoding="utf-8")
        cpp = (ROOT / "native/src/jni_runtime.cpp").read_text(encoding="utf-8")
        for name in ("nativeGetVersion", "nativeGetBuildInfo", "nativeRunSyntheticCpuTest", "nativeGetNcnnInfo", "nativeProbeVulkanSupport"):
            self.assertIn(f"external fun {name}(): String", kotlin)
            self.assertIn(f"Java_com_seededge_runtime_NativeRuntime_{name}", cpp)
        self.assertIn('System.loadLibrary("seededge")', kotlin)

    def test_native_json_templates_are_parseable_and_device_honest(self):
        source = (ROOT / "native/src/runtime_info.cpp").read_text(encoding="utf-8")
        self.assertIn('"device_tested":false', source.replace("\\\"", '"'))
        self.assertNotRegex(source, r'\b(?:NaN|Infinity)\b')

    def test_synthetic_cpu_backend_uses_current_ncnn_net_options(self):
        source = (ROOT / "native/src/synthetic_cpu_test.cpp").read_text(encoding="utf-8")
        self.assertIn("net.opt.use_vulkan_compute=false", source)
        self.assertIn("net.opt.num_threads=1", source)
        self.assertNotIn("ex.set_vulkan_compute", source)
        host = (ROOT / "native/tests/synthetic_cpu_host_test.cpp").read_text(encoding="utf-8")
        jni = (ROOT / "native/src/jni_runtime.cpp").read_text(encoding="utf-8")
        self.assertIn("synthetic_cpu_test_json(false)", host)
        self.assertIn("synthetic_cpu_test_json(true)", jni)

    def test_manifest_validator_rejects_dangerous_paths_and_bad_hashes(self):
        source = (ROOT / "native/src/model_manifest.cpp").read_text(encoding="utf-8")
        for token in ("path traversal", "absolute path", "sha256 must be 64 lowercase hex", "format_version must be 1", "size mismatch"):
            self.assertIn(token, source)

    def test_vulkan_contract_does_not_claim_device_execution(self):
        source = (ROOT / "native/src/vulkan_probe.cpp").read_text(encoding="utf-8")
        self.assertIn('"vulkan_compiled":true', source.replace("\\\"", '"'))
        self.assertIn('"vulkan_device_probe_executed":false', source.replace("\\\"", '"'))
        self.assertIn("No Android device connected during Phase 18A", source)

    def test_powershell_scripts_are_windows_powershell_compatible(self):
        source = (ROOT / "scripts/prepare_ncnn.ps1").read_text(encoding="utf-8")
        self.assertNotIn("utf8NoBOM", source)
        self.assertIn("fe88f421038e1bb0a25cd5c1b2dfe505db82d08f", source)
        inspector = (ROOT / "scripts/inspect_apk.ps1").read_text(encoding="utf-8")
        self.assertIn("System.IO.Compression.FileSystem", inspector)

    def test_reports_have_required_truth_fields(self):
        for report in pathlib.Path(ROOT.parent / "reports").glob("phase18a_*.json"):
            data = json.loads(report.read_text(encoding="utf-8"))
            self.assertIn("overall_pass", data, report.name)
            self.assertIn("device_required", data, report.name)
            self.assertIn("device_executed", data, report.name)

    def test_android16_log_view_uses_plain_bounded_main_thread_updates(self):
        source = (ROOT / "app/src/main/java/com/seededge/runtime/MainActivity.kt").read_text(encoding="utf-8")
        self.assertNotIn("ScrollingMovementMethod", source)
        self.assertNotIn("log.append(", source)
        self.assertIn("TextView.BufferType.NORMAL", source)
        self.assertIn("log.post", source)
        self.assertIn("isSaveEnabled = false", source)
        self.assertIn("R.id.runtime_log", source)

class Phase18dFullPhotoJniContractTests(unittest.TestCase):
    def test_full_photo_jni_entry_declared_in_kotlin_and_cpp(self):
        kotlin = (ROOT / "app/src/main/java/com/seededge/runtime/NativeRuntime.kt").read_text(encoding="utf-8")
        cpp = (ROOT / "native/src/jni_runtime.cpp").read_text(encoding="utf-8")
        self.assertIn(
            "external fun nativeRunFullPhoto(modelRoot: String, taskRoot: String, normalizedRgb: ByteArray,",
            kotlin,
        )
        self.assertIn("Java_com_seededge_runtime_NativeRuntime_nativeRunFullPhoto", cpp)
        self.assertIn("run_phase18d_full_photo_artifacts", cpp)

    def test_full_photo_jni_wrapper_keeps_stable_validation_codes(self):
        cpp = (ROOT / "native/src/jni_runtime.cpp").read_text(encoding="utf-8")
        self.assertIn("JNI_NULL_ARGUMENT", cpp)
        self.assertIn("JNI_STRING_ACQUIRE_FAILED", cpp)

    def test_full_photo_artifact_sources_are_part_of_android_library(self):
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("native/src/phase18d_image_artifacts.cpp", cmake)
        self.assertIn("native/src/phase18d_full_photo_artifacts.cpp", cmake)
        self.assertIn("native/third_party/stb/stb_impl.cpp", cmake)
        self.assertIn("native/third_party/stb", cmake)

if __name__ == "__main__":
    unittest.main()
