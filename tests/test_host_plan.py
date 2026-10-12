"""GPU-free checks for runtime/glm53_host.py on synthetic sysfs trees."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("glm53_host", ROOT / "runtime/glm53_host.py")
host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host)


def fake_sys(root: Path, cores: list[tuple[list[int], str]]) -> set[int]:
    """cores: [(hardware threads of one physical core, L3 shared_cpu_list)]; returns all CPUs."""
    cpus = set()
    for threads, l3 in cores:
        siblings = ",".join(map(str, threads))
        for cpu in threads:
            cpus.add(cpu)
            base = root / f"devices/system/cpu/cpu{cpu}"
            (base / "topology").mkdir(parents=True)
            (base / "topology/thread_siblings_list").write_text(siblings + "\n")
            for index, level in enumerate(("1", "1", "2", "3")):
                cache = base / f"cache/index{index}"
                cache.mkdir(parents=True)
                (cache / "level").write_text(level + "\n")
                (cache / "shared_cpu_list").write_text((l3 if level == "3" else siblings) + "\n")
    return cpus


def threadripper_5975wx() -> list[tuple[list[int], str]]:
    # 4 CCDs x 8 cores, SMT siblings n and n + 32
    return [([c, c + 32], f"{c // 8 * 8}-{c // 8 * 8 + 7},{c // 8 * 8 + 32}-{c // 8 * 8 + 39}") for c in range(32)]


def ryzen_7950x() -> list[tuple[list[int], str]]:
    # 2 CCDs x 8 cores, SMT siblings n and n + 16
    return [([c, c + 16], f"{c // 8 * 8}-{c // 8 * 8 + 7},{c // 8 * 8 + 16}-{c // 8 * 8 + 23}") for c in range(16)]


def intel_8c_ht() -> list[tuple[list[int], str]]:
    # one L3, siblings 2n and 2n + 1
    return [([2 * c, 2 * c + 1], "0-15") for c in range(8)]


class HostPlanTest(unittest.TestCase):
    def resolve(self, cores, allowed=None, env=None):
        with tempfile.TemporaryDirectory() as tmp:
            cpus = fake_sys(Path(tmp), cores)
            return host.resolve(Path(tmp), allowed if allowed is not None else cpus, env or {})

    def test_5975wx_matches_measured_sets(self):
        plan = self.resolve(threadripper_5975wx())
        self.assertEqual(plan["EXL3_HYBRID_CPUS_R0"], "0,1,2,8,9,10,16,17,18,24,25,26")
        self.assertEqual(plan["EXL3_HYBRID_CPUS_R1"], "4,5,6,12,13,14,20,21,22,28,29,30")
        self.assertEqual(plan["HYBRID_MAIN_CPUS"], "3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63")
        self.assertEqual(plan["EXL3_HYBRID_THREADS"], "12")

    def test_pool_siblings_stay_idle(self):
        plan = self.resolve(threadripper_5975wx())
        used = set()
        for key in ("EXL3_HYBRID_CPUS_R0", "EXL3_HYBRID_CPUS_R1", "HYBRID_MAIN_CPUS"):
            used |= set(host.parse_list(plan[key]))
        for cpu in host.parse_list(plan["EXL3_HYBRID_CPUS_R0"]) + host.parse_list(plan["EXL3_HYBRID_CPUS_R1"]):
            self.assertNotIn(cpu + 32, used)

    def test_7950x(self):
        plan = self.resolve(ryzen_7950x())
        self.assertEqual(plan["EXL3_HYBRID_CPUS_R0"], "0,1,2,8,9,10")
        self.assertEqual(plan["EXL3_HYBRID_CPUS_R1"], "4,5,6,12,13,14")
        self.assertEqual(plan["EXL3_HYBRID_THREADS"], "6")

    def test_single_l3(self):
        plan = self.resolve(intel_8c_ht())
        self.assertEqual(plan["EXL3_HYBRID_CPUS_R0"], "0,2,4")
        self.assertEqual(plan["EXL3_HYBRID_CPUS_R1"], "8,10,12")
        self.assertEqual(plan["HYBRID_MAIN_CPUS"], "6,7,14,15")

    def test_cpuset_restriction(self):
        allowed = set(range(0, 16)) | set(range(32, 48))   # first two CCDs only
        plan = self.resolve(threadripper_5975wx(), allowed=allowed)
        self.assertEqual(plan["EXL3_HYBRID_CPUS_R0"], "0,1,2,8,9,10")
        self.assertEqual(plan["HYBRID_MAIN_CPUS"], "3,7,11,15,35,39,43,47")

    def test_explicit_env_wins(self):
        plan = self.resolve(threadripper_5975wx(), env={"EXL3_HYBRID_CPUS_R0": "0-5", "EXL3_HYBRID_THREADS": "6"})
        self.assertEqual(plan["EXL3_HYBRID_CPUS_R0"], "0-5")
        self.assertEqual(plan["EXL3_HYBRID_THREADS"], "6")

    def test_too_small(self):
        with self.assertRaises(SystemExit):
            self.resolve([([0, 1], "0-3"), ([2, 3], "0-3")])


if __name__ == "__main__":
    unittest.main()
