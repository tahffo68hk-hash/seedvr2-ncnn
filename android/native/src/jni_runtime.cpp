#include "runtime_info.h"
#include <jni.h>
static jstring js(JNIEnv* e,const std::string& s){return e->NewStringUTF(s.c_str());}
extern "C" JNIEXPORT jstring JNICALL Java_com_seededge_runtime_NativeRuntime_nativeGetVersion(JNIEnv*e,jobject){return js(e,runtime_version_json());}
extern "C" JNIEXPORT jstring JNICALL Java_com_seededge_runtime_NativeRuntime_nativeGetBuildInfo(JNIEnv*e,jobject){return js(e,runtime_build_json());}
extern "C" JNIEXPORT jstring JNICALL Java_com_seededge_runtime_NativeRuntime_nativeRunSyntheticCpuTest(JNIEnv*e,jobject){return js(e,synthetic_cpu_test_json(true));}
extern "C" JNIEXPORT jstring JNICALL Java_com_seededge_runtime_NativeRuntime_nativeGetNcnnInfo(JNIEnv*e,jobject){return js(e,ncnn_info_json());}
extern "C" JNIEXPORT jstring JNICALL Java_com_seededge_runtime_NativeRuntime_nativeProbeVulkanSupport(JNIEnv*e,jobject){return js(e,vulkan_probe_json());}
