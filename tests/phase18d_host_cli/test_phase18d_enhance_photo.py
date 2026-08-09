import json
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools import phase18d_enhance_photo as cli

FAKE_RUNNER = r'''
import json, pathlib, shutil, sys
model, task, rgb, width, height, request = sys.argv[1:]
task = pathlib.Path(task); task.mkdir(parents=True, exist_ok=True)
shutil.copyfile(rgb, task / "output_full.rgb")
(task / "captured_request.json").write_text(request, encoding="utf-8")
payload = {"state":"completed","runtime_mode":json.loads(request)["runtime_mode"],
 "tile_count":1,"completed_tiles":1,"output_full_sha256":"fake","finite":True,
 "nan_count":0,"inf_count":0,"silent_fallback":False}
(task / "task_report.json").write_text(json.dumps(payload), encoding="utf-8")
print(json.dumps({"state":"completed","artifacts_written":True,"silent_fallback":False}))
'''
FAKE_PROGRESS_RUNNER = r'''
import json, pathlib, shutil, sys, time
model, task, rgb, width, height, request = sys.argv[1:]
task=pathlib.Path(task); task.mkdir(parents=True,exist_ok=True)
(task/"runtime_progress.json").write_text(json.dumps({"state":"running","phase":"stage_major","stage_name":"dit","dit_block_index":7,"tile_index":2,"tile_count":4}))
time.sleep(0.3); shutil.copyfile(rgb,task/"output_full.rgb")
(task/"task_report.json").write_text(json.dumps({"state":"completed","finite":True,"silent_fallback":False}))
print('{"state":"completed","artifacts_written":true,"silent_fallback":false}')
'''

class EnhancePhotoCliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.model = self.root / "model"; self.model.mkdir()
        (self.model / "manifest.json").write_text('{"schema_version":1}', encoding="utf-8")
        self.runner = self.root / "fake_runner.py"; self.runner.write_text(FAKE_RUNNER, encoding="utf-8")

    def tearDown(self): self.tmp.cleanup()

    def run_cli(self, source, suffix):
        output = self.root / f"out_{suffix}.png"; task = self.root / f"task_{suffix}"
        args = ["--input", str(source), "--output", str(output), "--model-root", str(self.model),
                "--task-root", str(task), "--runner", sys.executable,
                "--runner-prefix", str(self.runner), "--runtime-mode", "hybrid",
                "--tile-cohort-size", "4"]
        self.assertEqual(cli.main(args), 0)
        return output, task

    def test_jpeg_png_and_webp_produce_same_size_png_and_hybrid_request(self):
        for extension in ("jpg", "png", "webp"):
            with self.subTest(extension=extension):
                source = self.root / f"input.{extension}"
                Image.new("RGB", (13, 7), (20, 40, 80)).save(source)
                output, task = self.run_cli(source, extension)
                with Image.open(output) as image:
                    self.assertEqual(image.size, (13, 7)); self.assertEqual(image.mode, "RGB")
                request = json.loads((task / "captured_request.json").read_text("utf-8"))
                identity = json.loads((task / "task_identity.json").read_text("utf-8"))
                self.assertEqual(request["runtime_mode"], "hybrid")
                self.assertEqual(request["tile_cohort_size"], 4)
                self.assertEqual(request["normalized_image_sha256"], identity["normalized_image_sha256"])

    def test_exif_orientation_is_applied_before_dimensions_and_rgb_hash(self):
        source = self.root / "rotated.jpg"; image = Image.new("RGB", (2, 3), (100, 30, 10))
        exif = Image.Exif(); exif[274] = 6; image.save(source, exif=exif)
        output, task = self.run_cli(source, "exif")
        with Image.open(output) as result: self.assertEqual(result.size, (3, 2))
        identity = json.loads((task / "task_identity.json").read_text("utf-8"))
        self.assertEqual((identity["width"], identity["height"]), (3, 2))

    def test_existing_task_rejects_different_input_instead_of_overwriting(self):
        source = self.root / "first.png"; Image.new("RGB", (4, 4), "red").save(source)
        _, task = self.run_cli(source, "collision"); Image.new("RGB", (4, 4), "blue").save(source)
        with self.assertRaisesRegex(cli.CliError, "TASK_IDENTITY_MISMATCH"):
            cli.main(["--input", str(source), "--output", str(self.root / "bad.png"),
                "--model-root", str(self.model), "--task-root", str(task), "--runner", sys.executable,
                "--runner-prefix", str(self.runner), "--runtime-mode", "hybrid",
                "--tile-cohort-size", "4", "--resume"])

    def test_default_task_ids_are_unique_and_native_safe(self):
        first = cli.new_task_id(); second = cli.new_task_id()
        self.assertNotEqual(first, second)
        self.assertRegex(first, r"^photo_[0-9]{8}_[0-9]{6}_[0-9a-f]{8}$")

    def test_progress_is_read_from_runtime_file_not_synthesized(self):
        source=self.root/"progress.png"; Image.new("RGB",(5,5),"green").save(source)
        progress_runner=self.root/"progress_runner.py"; progress_runner.write_text(FAKE_PROGRESS_RUNNER,encoding="utf-8")
        captured=io.StringIO()
        with contextlib.redirect_stdout(captured):
            self.assertEqual(cli.main(["--input",str(source),"--output",str(self.root/"progress_out.png"),
                "--model-root",str(self.model),"--task-root",str(self.root/"progress_task"),
                "--runner",sys.executable,"--runner-prefix",str(progress_runner),
                "--runtime-mode","stage_major"]),0)
        self.assertIn("Progress: stage_major/dit block=7 tile=2/4",captured.getvalue())

    def test_cancel_publishes_atomic_safe_boundary_request_without_runner(self):
        task = self.root / "cancel_task"; task.mkdir()
        self.assertEqual(cli.main(["--task-root", str(task), "--cancel"]), 0)
        control = json.loads((task / "runtime_control.json").read_text("utf-8"))
        self.assertEqual(control["action"], "cancel")
        self.assertFalse((task / "runtime_control.json.tmp").exists())

    def test_auto_prints_measured_resource_plan_and_keeps_auto_request(self):
        source = self.root / "auto.png"; Image.new("RGB", (640, 427), "navy").save(source)
        captured = io.StringIO(); task = self.root / "auto_task"
        with contextlib.redirect_stdout(captured):
            self.assertEqual(cli.main(["--input", str(source), "--output", str(self.root / "auto_out.png"),
                "--model-root", str(self.model), "--task-root", str(task),
                "--runner", sys.executable, "--runner-prefix", str(self.runner),
                "--runtime-mode", "auto"]), 0)
        request = json.loads((task / "captured_request.json").read_text("utf-8"))
        self.assertEqual(request["runtime_mode"], "auto")
        self.assertIn("Estimate [ESTIMATED from HOST_MEASURED_256_512_1024_2026-08-09]", captured.getvalue())
        self.assertIn("Auto plan: hybrid K=4", captured.getvalue())

if __name__ == "__main__": unittest.main()
