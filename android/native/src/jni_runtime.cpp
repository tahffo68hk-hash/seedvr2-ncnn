#include "runtime_info.h"
#include "minimal_video_pipeline.h"
#include "phase18d_full_photo_artifacts.h"
#include "phase18d_stage_major.h"
#include "runtime_progress.h"
#include "model_manifest.h"
#include "fast_convolution3d.h"
#include <jni.h>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <thread>
#include <chrono>
#include <algorithm>
#include <atomic>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
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
extern "C" JNIEXPORT jstring JNICALL Java_com_seededge_runtime_NativeRuntime_nativeRunSinglePhotoTile(
    JNIEnv* e, jobject, jstring model, jstring task, jbyteArray rgb, jint width, jint height, jstring request) {
    if (!model || !task || !rgb || !request)
        return js(e, "{\"error_code\":\"JNI_NULL_ARGUMENT\",\"overall_pass\":false}");
    const char* m = e->GetStringUTFChars(model, nullptr);
    const char* t = e->GetStringUTFChars(task, nullptr);
    const char* r = e->GetStringUTFChars(request, nullptr);
    if (!m || !t || !r) {
        if (m) e->ReleaseStringUTFChars(model, m);
        if (t) e->ReleaseStringUTFChars(task, t);
        if (r) e->ReleaseStringUTFChars(request, r);
        return js(e, "{\"error_code\":\"JNI_STRING_ACQUIRE_FAILED\",\"overall_pass\":false}");
    }
    const jsize length = e->GetArrayLength(rgb);
    std::vector<jbyte> copied(static_cast<size_t>(length));
    if (length) e->GetByteArrayRegion(rgb, 0, length, copied.data());
    const std::string result = run_phase18d_single_photo_tile(
        m, t, reinterpret_cast<const uint8_t*>(copied.data()), copied.size(), width, height, r);
    e->ReleaseStringUTFChars(model, m);
    e->ReleaseStringUTFChars(task, t);
    e->ReleaseStringUTFChars(request, r);
    return js(e, result);
}
extern "C" JNIEXPORT jstring JNICALL Java_com_seededge_runtime_NativeRuntime_nativeRunFullPhoto(
    JNIEnv* e, jobject, jstring model, jstring task, jbyteArray rgb, jint width, jint height, jstring request) {
    if (!model || !task || !rgb || !request)
        return js(e, "{\"error_code\":\"JNI_NULL_ARGUMENT\",\"overall_pass\":false}");
    const char* m = e->GetStringUTFChars(model, nullptr);
    const char* t = e->GetStringUTFChars(task, nullptr);
    const char* r = e->GetStringUTFChars(request, nullptr);
    if (!m || !t || !r) {
        if (m) e->ReleaseStringUTFChars(model, m);
        if (t) e->ReleaseStringUTFChars(task, t);
        if (r) e->ReleaseStringUTFChars(request, r);
        return js(e, "{\"error_code\":\"JNI_STRING_ACQUIRE_FAILED\",\"overall_pass\":false}");
    }
    const jsize length = e->GetArrayLength(rgb);
    std::vector<jbyte> copied(static_cast<size_t>(length));
    if (length) e->GetByteArrayRegion(rgb, 0, length, copied.data());
    const std::string result = seededge::phase18d::run_phase18d_full_photo_artifacts(
        m, t, reinterpret_cast<const uint8_t*>(copied.data()), copied.size(), width, height, r);
    e->ReleaseStringUTFChars(model, m);
    e->ReleaseStringUTFChars(task, t);
    e->ReleaseStringUTFChars(request, r);
    return js(e, result);
}
namespace
{
struct FullPhotoJniBridge
{
    JNIEnv* env = nullptr;
    jobject bridge = nullptr;
    jmethodID on_progress_method = nullptr;
    jmethodID is_cancelled_method = nullptr;
    std::atomic<bool> cancelled{false};
    std::string callback_error;
};

void jni_emit_progress(void* context, const char* stage, int completed, int total)
{
    auto* real = static_cast<seededge::phase18d::Phase18dRealContext*>(context);
    FullPhotoJniBridge* bridge = static_cast<FullPhotoJniBridge*>(real->jni_bridge);
    JNIEnv* env = bridge->env;
    jstring stage_text = env->NewStringUTF(stage);
    env->CallBooleanMethod(bridge->bridge, bridge->on_progress_method,
                           stage_text, completed, total);
    env->DeleteLocalRef(stage_text);
    if (env->ExceptionCheck())
    {
        env->ExceptionClear();
        bridge->cancelled.store(true);
        bridge->callback_error = std::string("onProgress callback threw for stage ") + stage;
        return;
    }
}

bool jni_is_cancelled(void* context)
{
    auto* real = static_cast<seededge::phase18d::Phase18dRealContext*>(context);
    FullPhotoJniBridge* bridge = static_cast<FullPhotoJniBridge*>(real->jni_bridge);
    if (bridge->cancelled.load()) return true;
    JNIEnv* env = bridge->env;
    const jboolean cancelled = env->CallBooleanMethod(
        bridge->bridge, bridge->is_cancelled_method);
    if (env->ExceptionCheck())
    {
        env->ExceptionClear();
        bridge->cancelled.store(true);
        return true;
    }
    if (cancelled) bridge->cancelled.store(true);
    return bridge->cancelled.load();
}
}  // namespace

extern "C" JNIEXPORT jstring JNICALL Java_com_seededge_runtime_NativeRuntime_nativeRunFullPhotoWithProgress(
    JNIEnv* e, jobject, jstring model, jstring task, jbyteArray rgb, jint width, jint height,
    jstring request, jobject bridge) {
    if (!model || !task || !rgb || !request || !bridge)
        return js(e, "{\"error_code\":\"JNI_NULL_ARGUMENT\",\"overall_pass\":false}");
    const char* m = e->GetStringUTFChars(model, nullptr);
    const char* t = e->GetStringUTFChars(task, nullptr);
    const char* r = e->GetStringUTFChars(request, nullptr);
    if (!m || !t || !r) {
        if (m) e->ReleaseStringUTFChars(model, m);
        if (t) e->ReleaseStringUTFChars(task, t);
        if (r) e->ReleaseStringUTFChars(request, r);
        return js(e, "{\"error_code\":\"JNI_STRING_ACQUIRE_FAILED\",\"overall_pass\":false}");
    }
    const jsize length = e->GetArrayLength(rgb);
    std::vector<jbyte> copied(static_cast<size_t>(length));
    if (length) e->GetByteArrayRegion(rgb, 0, length, copied.data());
    jclass bridge_class = e->GetObjectClass(bridge);
    jmethodID on_progress = e->GetMethodID(bridge_class, "onProgress", "(Ljava/lang/String;II)Z");
    jmethodID is_cancelled = e->GetMethodID(bridge_class, "isCancelled", "()Z");
    if (!on_progress || !is_cancelled) {
        e->ReleaseStringUTFChars(model, m);
        e->ReleaseStringUTFChars(task, t);
        e->ReleaseStringUTFChars(request, r);
        return js(e, "{\"state\":\"failed\",\"artifacts_written\":false,\"silent_fallback\":false,\"error_code\":\"JNI_BRIDGE_METHOD_MISSING\",\"error\":\"FullPhotoProgressBridge methods not found\"}");
    }
    FullPhotoJniBridge context;
    context.env = e;
    context.bridge = bridge;
    context.on_progress_method = on_progress;
    context.is_cancelled_method = is_cancelled;
    seededge::phase18d::Phase18dRealContext real_context;
    real_context.jni_bridge = &context;
    seededge::phase18d::Phase18dFullPhotoHooks hooks =
        seededge::phase18d::make_real_full_photo_hooks(
            &real_context, &jni_is_cancelled, &jni_emit_progress);
    const std::string result = seededge::phase18d::run_phase18d_full_photo_artifacts_with_hooks(
        m, t, reinterpret_cast<const uint8_t*>(copied.data()), copied.size(), width, height, r, hooks);
    e->ReleaseStringUTFChars(model, m);
    e->ReleaseStringUTFChars(task, t);
    e->ReleaseStringUTFChars(request, r);
    if (!context.callback_error.empty())
        return js(e, "{\"state\":\"failed\",\"artifacts_written\":false,\"silent_fallback\":false,\"error_code\":\"PROGRESS_CALLBACK_EXCEPTION\",\"error\":\"" + context.callback_error + "\"}");
    return js(e, result);
}
extern "C" JNIEXPORT jstring JNICALL Java_com_seededge_runtime_NativeRuntime_nativeGetProgressJson(JNIEnv*e,jobject){return js(e,seededge_progress::json());}
extern "C" JNIEXPORT jstring JNICALL Java_com_seededge_runtime_NativeRuntime_nativeRequestCancel(JNIEnv*e,jobject){seededge_progress::cancel();return js(e,seededge_progress::json());}
extern "C" JNIEXPORT jstring JNICALL Java_com_seededge_runtime_NativeRuntime_nativeRequestPause(JNIEnv*e,jobject){seededge_progress::pause();return js(e,seededge_progress::json());}
extern "C" JNIEXPORT jstring JNICALL Java_com_seededge_runtime_NativeRuntime_nativeControlStageMajor(JNIEnv*e,jobject,jstring root,jstring state){
    const char*r=e->GetStringUTFChars(root,nullptr),*s=e->GetStringUTFChars(state,nullptr);
    seededge::phase18d::StageMajorCheckpoint checkpoint; std::string error;
    const bool ok=seededge::phase18d::load_stage_major_checkpoint(r,&checkpoint,&error)&&
        seededge::phase18d::set_stage_major_control(&checkpoint,s,&error)&&
        seededge::phase18d::persist_stage_major_checkpoint(r,checkpoint,&error);
    const std::string result=ok?"{\"ok\":true,\"state\":\""+checkpoint.state+"\"}":
        "{\"ok\":false,\"error\":\""+error+"\"}";
    e->ReleaseStringUTFChars(root,r);e->ReleaseStringUTFChars(state,s);return js(e,result);
}
extern "C" JNIEXPORT jstring JNICALL Java_com_seededge_runtime_NativeRuntime_nativeInspectModel(JNIEnv*e,jobject,jstring root){const char*r=e->GetStringUTFChars(root,nullptr);namespace fs=std::filesystem;fs::path base(r),m=base/"manifest.json";std::ifstream f(m);std::stringstream b;b<<f.rdbuf();std::string j=b.str();size_t expected=0,found=0;unsigned long long expected_bytes=0,found_bytes=0;std::smatch x;std::regex ent(R"xx("relative_path"\s*:\s*"([^"]+)"\s*,\s*"size_bytes"\s*:\s*([0-9]+))xx");for(auto it=std::sregex_iterator(j.begin(),j.end(),ent);it!=std::sregex_iterator();++it){++expected;auto rel=(*it)[1].str();auto bytes=std::stoull((*it)[2].str());expected_bytes+=bytes;fs::path p=base/rel;if(fs::is_regular_file(p)){++found;found_bytes+=fs::file_size(p);}}bool manifest=fs::exists(m)&&!j.empty();std::string s="{\"schema_version\":1,\"inspection\":\"quick\",\"model_root\":\"models/phase18c\",\"manifest_present\":"+(manifest?std::string("true"):"false")+",\"expected_files\":"+std::to_string(expected)+",\"found_files\":"+std::to_string(found)+",\"expected_bytes\":"+std::to_string(expected_bytes)+",\"found_bytes\":"+std::to_string(found_bytes)+",\"ready\":"+((manifest&&expected>0&&expected==found&&expected_bytes==found_bytes)?std::string("true"):"false")+",\"full_sha256\":false}";e->ReleaseStringUTFChars(root,r);return js(e,s);}
extern "C" JNIEXPORT jstring JNICALL Java_com_seededge_runtime_NativeRuntime_nativeInspectCheckpoints(JNIEnv*e,jobject,jstring root){const char*r=e->GetStringUTFChars(root,nullptr);namespace fs=std::filesystem;fs::path p=fs::path(r)/"frames";bool v=fs::exists(p/"gate_dit_block_32_vid.f32"),t=fs::exists(p/"gate_dit_block_32_txt.f32");auto sz=[&](const fs::path&x){return fs::exists(x)?static_cast<long long>(fs::file_size(x)):0LL;};bool bridge=fs::exists(fs::path(r)/"decoder_bridge_result.json")&&fs::exists(fs::path(r)/"frame_000.rgb");std::string s="{\"schema_version\":1,\"block32_vid\":"+(v?std::string("true"):"false")+",\"block32_txt\":"+(t?std::string("true"):"false")+",\"block32_vid_bytes\":"+std::to_string(sz(p/"gate_dit_block_32_vid.f32"))+",\"block32_txt_bytes\":"+std::to_string(sz(p/"gate_dit_block_32_txt.f32"))+",\"checkpoint_verified\":"+((v&&t)?std::string("true"):"false")+",\"decoder_bridge_verified\":"+(bridge?std::string("true"):"false")+",\"ready_to_generate_video\":false}";e->ReleaseStringUTFChars(root,r);return js(e,s);}
extern "C" JNIEXPORT jstring JNICALL Java_com_seededge_runtime_NativeRuntime_nativeRunDecoderOnly(JNIEnv*e,jobject,jstring model,jstring checkpoint,jstring output,jstring config){const char*m=e->GetStringUTFChars(model,nullptr),*k=e->GetStringUTFChars(checkpoint,nullptr),*o=e->GetStringUTFChars(output,nullptr),*c=e->GetStringUTFChars(config,nullptr);auto r=run_phase18c_decoder_only(m,k,o,c);e->ReleaseStringUTFChars(model,m);e->ReleaseStringUTFChars(checkpoint,k);e->ReleaseStringUTFChars(output,o);e->ReleaseStringUTFChars(config,c);return js(e,r);}
extern "C" JNIEXPORT jstring JNICALL Java_com_seededge_runtime_NativeRuntime_nativeRunProgressSimulation(JNIEnv*e,jobject){seededge_progress::reset("simulation", ""); const char* phases[]={"checking_model","verifying_manifest","loading_model","vae_encoder","dit_streaming","vae_decoder","writing_frames","video_encoding"}; for(int i=0;i<8 && !seededge_progress::cancelled();++i){seededge_progress::update(phases[i],"simulation_stage","synthetic progress test",0,10,i*10,100,i==4?2:0,i==5?3:0,i==5?7:0,i>=6?1:0,i>=6?1:0);std::this_thread::sleep_for(std::chrono::milliseconds(250));} const bool cancelled=seededge_progress::cancelled();seededge_progress::finish(cancelled?"cancelled":"completed",cancelled?"CANCELLED_BY_USER":"","");return js(e,cancelled?"{\"simulation\":true,\"cancel_pass\":true,\"state\":\"cancelled\"}":"{\"simulation\":true,\"state\":\"completed\"}");}
extern "C" JNIEXPORT jstring JNICALL Java_com_seededge_runtime_NativeRuntime_nativeRunOpenMpThreadProbe(JNIEnv*e,jobject,jint requested){int observed=1; long long checksum=0; const int threads=std::max(1,int(requested));
#ifdef _OPENMP
observed=0;
#pragma omp parallel num_threads(threads) reduction(+:checksum)
{ int local=0; for(int i=0;i<200000;++i)local+=(i&7); checksum+=local;
#pragma omp atomic
++observed; }
#endif
 const bool compiled =
#ifdef _OPENMP
 true;
#else
 false;
#endif
 return js(e,"{\"requested_threads\":"+std::to_string(threads)+",\"observed_worker_threads\":"+std::to_string(observed)+",\"checksum\":"+std::to_string(checksum)+",\"openmp_compiled\":"+(compiled?"true":"false")+",\"overall_pass\":"+((observed>1||threads==1)?std::string("true"):std::string("false"))+"}");}
extern "C" JNIEXPORT jstring JNICALL Java_com_seededge_runtime_NativeRuntime_nativeRunDecoderStageBenchmark(JNIEnv*e,jobject,jstring model,jstring output){const char*m=e->GetStringUTFChars(model,nullptr),*o=e->GetStringUTFChars(output,nullptr);auto r=run_phase18c_decoder_stage_benchmark(m,o);e->ReleaseStringUTFChars(model,m);e->ReleaseStringUTFChars(output,o);return js(e,r);}
extern "C" JNIEXPORT jstring JNICALL Java_com_seededge_runtime_NativeRuntime_nativeTestFastConvolution3D(JNIEnv*e,jobject){return js(e,seededge::fast_convolution3d_test_json());}

extern "C" JNIEXPORT jstring JNICALL Java_com_seededge_runtime_NativeRuntime_nativeValidateModelManifest(JNIEnv* e, jobject, jstring root) {
    const char* r = e->GetStringUTFChars(root, nullptr);
    if (!r) return js(e, "{\"valid\":false,\"error\":\"JNI_STRING_ACQUIRE_FAILED\",\"silent_fallback\":false}");
    const std::string result = validate_model_manifest(r);
    e->ReleaseStringUTFChars(root, r);
    return js(e, result);
}
