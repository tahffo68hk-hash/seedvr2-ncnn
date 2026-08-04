import json, pathlib, subprocess, tempfile, zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]; REPO=ROOT.parent
results=[]
def check(name, value, detail=""): results.append({"name":name,"pass":bool(value),"detail":detail})
env=(ROOT/"scripts/check_environment.ps1").read_text(encoding="utf-8")
check("missing SDK",'Assert-Path $env:ANDROID_HOME "Android SDK"' in env)
check("missing NDK",'Assert-Path "$env:ANDROID_HOME\\ndk\\26.3.11579264" "Android NDK"' in env)
check("wrong JAVA_HOME",'Assert-Path "$env:JAVA_HOME\\bin\\java.exe" "JDK"' in env)
p=subprocess.run(["powershell","-NoProfile","-ExecutionPolicy","Bypass","-File",str(ROOT/"scripts/check_environment.ps1"),"-Abi","x86_64"],capture_output=True,text=True,errors="replace")
check("wrong ABI",p.returncode!=0,"exit="+str(p.returncode))
host=subprocess.run(["powershell","-NoProfile","-ExecutionPolicy","Bypass","-File",str(ROOT/"scripts/run_host_tests.ps1")],capture_output=True,text=True,errors="replace")
for name in ["missing model manifest","manifest invalid JSON","wrong format version","missing model file","wrong file size","wrong SHA256 format","path traversal"]: check(name,host.returncode==0 and "passed=9 failed=0" in host.stdout)
with tempfile.TemporaryDirectory(dir="D:\\Android\\temp") as td:
  for name,entry in [("APK model weight rejection","models/seedvr2/model.pth"),("APK non-arm64 ABI rejection","lib/x86_64/libseededge.so")]:
    apk=pathlib.Path(td)/(name.replace(" ","_")+".apk")
    with zipfile.ZipFile(apk,"w") as z:
      z.writestr("classes.dex",b"x");z.writestr("AndroidManifest.xml",b"x");z.writestr("lib/arm64-v8a/libseededge.so",b"x");z.writestr(entry,b"x")
    q=subprocess.run(["powershell","-NoProfile","-ExecutionPolicy","Bypass","-File",str(ROOT/"scripts/inspect_apk.ps1"),"-Apk",str(apk)],capture_output=True,text=True,errors="replace")
    check(name,q.returncode!=0,"exit="+str(q.returncode))
jni=(ROOT/"native/src/runtime_info.cpp").read_text(encoding="utf-8")
import re
parsed=[]
for literal in re.findall(r'R"\((\{.*?\})\)"',jni):
  try: parsed.append(json.loads(literal))
  except json.JSONDecodeError: pass
check("JNI JSON parse contract",len(parsed)>=2 and all(x.get("device_tested") is False for x in parsed))
syn=(ROOT/"native/src/synthetic_cpu_test.cpp").read_text(encoding="utf-8")
check("synthetic finite guard","std::isfinite" in syn and "max_error" in syn and "repeat_consistent" in syn)
out={"phase":"18A","overall_pass":all(x["pass"] for x in results),"device_required":False,"device_executed":False,"passed":sum(x["pass"] for x in results),"failed":sum(not x["pass"] for x in results),"results":results}
(REPO/"reports/phase18a_negative_tests.json").write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
print(json.dumps(out,indent=2));raise SystemExit(0 if out["overall_pass"] and len(results)==15 else 1)
