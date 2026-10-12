"""GPU-free checks for scripts/install_model.py on a synthetic checkpoint."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("install_model", ROOT / "scripts/install_model.py")
im = importlib.util.module_from_spec(spec)
spec.loader.exec_module(im)


def write_aux(path: Path, keys: list[str]) -> None:
    header, offset = {}, 0
    for key in keys:
        size = 4 * 2 if key.endswith("_ape") else 8 * 2
        header[key] = {"dtype": "BF16", "shape": [size // 2], "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * offset)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class InstallModelTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "ckpt"
        self.root.mkdir()
        (self.root / "model-00001-of-00001.safetensors").write_bytes(b"weights" * 100)
        (self.root / "config.json").write_text('{"model_type": "glm5next"}\n')
        index = {"metadata": {"total_size": 700}, "weight_map": {"lm_head.weight": "model-00001-of-00001.safetensors"}}
        (self.root / im.INDEX).write_text(json.dumps(index))
        write_aux(self.root / im.AUX, im.KPOOL_KEYS)
        expected = im.integrated_index((self.root / im.INDEX).read_text(), im.safetensors_header(self.root / im.AUX))
        files = {p.name: {"size": p.stat().st_size, "sha256": sha(p)} for p in self.root.iterdir()}
        self.lock = Path(self.tmp.name) / "lock.json"
        self.lock.write_text(json.dumps({
            "repo": "test/repo", "revision": "0" * 40, "files": files,
            "kpool_aux_index": {"integrated_index_sha256": hashlib.sha256(expected.encode()).hexdigest()},
        }))

    def tearDown(self):
        self.tmp.cleanup()

    def run_main(self, *extra):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = im.main([str(self.root), *extra], lock_path=self.lock)
        return code, out.getvalue() + err.getvalue()

    def test_integrates_then_idempotent(self):
        code, out = self.run_main()
        self.assertEqual(code, 0, out)
        self.assertIn("integrating", out)
        data = json.loads((self.root / im.INDEX).read_text())
        self.assertEqual(sum(v == im.AUX for v in data["weight_map"].values()), 22)
        self.assertEqual(data["metadata"]["total_size"], 700 + 11 * (8 + 16))
        self.assertTrue((self.root / im.BACKUP).is_file())
        code, out = self.run_main("--full")
        self.assertEqual(code, 0, out)
        self.assertIn("already integrated", out)

    def test_rejects_tampered_file(self):
        (self.root / "config.json").write_text('{"model_type": "evil0000"}\n')
        code, out = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("mismatch", out)
        self.assertFalse((self.root / im.BACKUP).exists())

    def test_rejects_missing_file(self):
        (self.root / "config.json").unlink()
        code, out = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("missing: config.json", out)

    def test_refuses_index_that_already_lists_kpool(self):
        with self.assertRaises(ValueError):
            im.integrated_index(json.dumps({"weight_map": {im.KPOOL_KEYS[0]: "x"}}),
                                im.safetensors_header(self.root / im.AUX))


if __name__ == "__main__":
    unittest.main()
