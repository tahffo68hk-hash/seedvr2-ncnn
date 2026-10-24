# Demo screenshots

| File | Evidence | Status |
| --- | --- | --- |
| 01-build.png | Windows Terminal summary from a clean Release build; crop excludes machine-local build paths | Captured |
| 02-alignment-result.png | Windows Terminal output with desktop A/B/C alignment and tablet A/B/C measurements | Captured |
| 03-desktop-run.png | Windows Terminal output from a separate desktop Dynamic C run | Captured |
| 04-tablet-run.png | Xiaomi Pad 7 Pro device-information screen captured with `adb exec-out screencap -p`; cropped to exclude the personalized device name | Captured |

The tablet verifier runs through `adb shell`, so it does not display its stdout on the tablet. The tablet image therefore documents the device, while the terminal report and JSON provide run output. No mock or generated screenshot was used.
