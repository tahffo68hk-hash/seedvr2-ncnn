#include "runtime_info.h"
#include "minimal_video_pipeline.h"
#include "runtime_progress.h"
#include "model_manifest.h"
#include <jni.h>
#include <filesystem>
static jstring js(JNIEnv* e,const std::string& s){return e->NewStringUTF(s.c_str());}
extern "C" JNIEXPORT jstring JNICALL Java_com_seededge_runtime_NativeRuntime_nativeGetVersion(JNIEnv*e,jobject){return js(e,runtime_version_json());}
extern "C" JNIEXPORT jstring JNICALL Java_com_seededge_runtime_NativeRuntime_nativeGetBuildInfo(JNIEnv*e,jobject){return js(e,runtime_build_json());}
extern "C" JNIEXPORT jstring JNICALL Java_com_seededge_runtime_NativeRuntime_nativeRunSyntheticCpuTest(JNIEnv*e,jobject){return js(e,synthetic_cpu_test_json(true));}
extern "C" JNIEXPORT jstring JNICALL Java_com_seededge_runtime_NativeRuntime_nativeGetNcnnInfo(JNIEnv*e,jobject){return js(e,ncnn_info_json());}
extern "C" JNIEXPORT jstring JNICALL Java_com_seededge_runtime_NativeRuntime_nativeProbeVulkanSupport(JNIEnv*e,jobject){return js(e,vulkan_probe_json());}
extern "C" JNIEXPORT jstring JNICALL Java_com_seededge_runtime_NativeRuntime_nativeRunMinimalVideoDemo(
    JNIEnv* e,jobject,jstring model,jstring input,jstring output,jstring config){
    const char*m=e->GetStringUTFChars(model,nullptr),*i=e->GetStringUTFChars(input,nullptr),*o=e->GetStringUTFChars(output,nullptr),*c=e->GetStringUTFChars(config,nullptr);
    const std::string result=run_phase18c_minimal_video_demo(m,i,o,c);
    e->ReleaseStringUTFChars(model,m);e->ReleaseStringUTFChars(input,i);e->ReleaseStringUTFChars(output,o);e->ReleaseStringUTFChars(config,c);
    return js(e,result);
}
extern "C" JNIEXPORT jstring JNICALL Java_com_seededge_runtime_NativeRuntime_nativeGetProgressJson(JNIEnv*e,jobject){return js(e,seededge_progress::json());}
extern "C" JNIEXPORT jstring JNICALL Java_com_seededge_runtime_NativeRuntime_nativeRequestCancel(JNIEnv*e,jobject){seededge_progress::cancel();return js(e,seededge_progress::json());}
extern "C" JNIEXPORT jstring JNICALL Java_com_seededge_runtime_NativeRuntime_nativeInspectModel(JNIEnv*e,jobject,jstring root){const char*r=e->GetStringUTFChars(root,nullptr);auto s=validate_model_manifest(r);e->ReleaseStringUTFChars(root,r);return js(e,s);}
extern "C" JNIEXPORT jstring JNICALL Java_com_seededge_runtime_NativeRuntime_nativeInspectCheckpoints(JNIEnv*e,jobject,jstring root){const char*r=e->GetStringUTFChars(root,nullptr);namespace fs=std::filesystem;fs::path p=fs::path(r)/"frames";bool v=fs::exists(p/"gate_dit_block_32_vid.f32"),t=fs::exists(p/"gate_dit_block_32_txt.f32");auto sz=[&](const fs::path&x){return fs::exists(x)?static_cast<long long>(fs::file_size(x)):0LL;};std::string s="{\"schema_version\":1,\"block32_vid\":"+(v?std::string("true"):"false")+",\"block32_txt\":"+(t?std::string("true"):"false")+",\"block32_vid_bytes\":"+std::to_string(sz(p/"gate_dit_block_32_vid.f32"))+",\"block32_txt_bytes\":"+std::to_string(sz(p/"gate_dit_block_32_txt.f32"))+",\"checkpoint_verified\":"+((v&&t)?std::string("true"):"false")+"}";e->ReleaseStringUTFChars(root,r);return js(e,s);}
