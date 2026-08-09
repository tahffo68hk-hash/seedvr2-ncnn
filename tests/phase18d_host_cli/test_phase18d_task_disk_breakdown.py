import tempfile
import unittest
from pathlib import Path

from tools import phase18d_task_disk_breakdown as disk

class DiskBreakdownTest(unittest.TestCase):
    def test_real_files_are_classified_without_double_counting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); tile=root/"tiles"/"tile_0000"; tile.mkdir(parents=True)
            files={"source_image.jpg":3,"normalized.rgb":5,"output_full.rgb":7,
                   "stage_major_checkpoint.json":11,"checkpoints/generation_0000000001.json":13,
                   "tiles/tile_0000/tile_input.rgb":17,"tiles/tile_0000/input_norm.f32":19,
                   "tiles/tile_0000/state_encoder_2.f32":23,"tiles/tile_0000/state_dit_7_vid.f32":29,
                   "tiles/tile_0000/state_scheduler.f32":31,"tiles/tile_0000/state_decoder_4.f32":37,
                   "tiles/tile_0000/output_valid.rgb":41,"tiles/tile_0000/gate_debug.f32":43}
            for name,size in files.items():
                path=root/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(bytes(size))
            report=disk.analyze_task(root)
            self.assertEqual(report["total_bytes"],sum(files.values()))
            self.assertEqual(report["categories"]["checkpoint_metadata"],24)
            self.assertEqual(report["categories"]["tile_inputs"],36)
            self.assertEqual(report["categories"]["dit_state"],29)
            self.assertEqual(report["categories"]["debug_gates"],43)
            self.assertEqual(report["tile_count"],1)
            self.assertEqual(report["bytes_per_tile"],sum(v for k,v in files.items() if k.startswith("tiles/")))

if __name__=="__main__": unittest.main()
