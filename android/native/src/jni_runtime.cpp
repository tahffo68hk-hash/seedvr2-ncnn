#include "runtime_info.h"
#include "minimal_video_pipeline.h"
#include "runtime_progress.h"
#include "model_manifest.h"
#include <jni.h>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <thread>
#include <chrono>
#include <algorithm>
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
extern "C" JNIEXPORT jstring JNICALL Java_com_seededge_runtime_NativeRuntime_nativeGetProgressJson(JNIEnv*e,jobject){return js(e,seededge_progress::json());}
extern "C" JNIEXPORT jstring JNICALL Java_com_seededge_runtime_NativeRuntime_nativeRequestCancel(JNIEnv*e,jobject){seededge_progress::cancel();return js(e,seededge_progress::json());}
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
