from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from journal_cpt.services.cooldown import SharedCooldown, runtime_dir
from journal_cpt.services.mineru import _mineru_slot, mineru_providers


def _journal(tmpdir: str, journal_id: str = "j1") -> SimpleNamespace:
    return SimpleNamespace(journal_id=journal_id, output_dir=str(Path(tmpdir) / "out" / journal_id))


def _cfg(tmpdir: str, providers, **mineru):
    base = {"timeout": 3600, "parse_method": "auto", "slot_poll_seconds": 0.05, "slot_stale_seconds": 7200}
    base.update(mineru)
    if providers is not None:
        base["providers"] = providers
    return {"runtime": {"output_root": str(Path(tmpdir) / "out"), "cooldown_refresh_seconds": 0.0}, "mineru": base}


class ProviderResolutionTests(unittest.TestCase):
    def test_providers_inherit_shared_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _cfg(tmp, [{"name": "a", "url": "http://a:8000", "max_concurrency": 4}], timeout=1234)
            providers = mineru_providers(cfg)
            self.assertEqual(len(providers), 1)
            self.assertEqual(providers[0].name, "a")
            self.assertEqual(providers[0].cfg["timeout"], 1234)      # 继承顶层
            self.assertEqual(providers[0].cfg["parse_method"], "auto")
            self.assertEqual(providers[0].max_concurrency, 4)        # 自己覆盖

    def test_disabled_provider_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _cfg(tmp, [
                {"name": "a", "url": "http://a:8000"},
                {"name": "b", "url": "http://b:8000", "enabled": False},
            ])
            self.assertEqual([p.name for p in mineru_providers(cfg)], ["a"])

    def test_falls_back_to_single_instance_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _cfg(tmp, None, url="http://legacy:8000", max_concurrency=3)
            providers = mineru_providers(cfg)
            self.assertEqual(len(providers), 1)
            self.assertEqual(providers[0].url, "http://legacy:8000")
            self.assertEqual(providers[0].max_concurrency, 3)

    def test_provider_without_url_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _cfg(tmp, [{"name": "broken"}])
            with self.assertRaises(RuntimeError) as ctx:
                mineru_providers(cfg)
            self.assertIn("broken", str(ctx.exception))

    def test_all_disabled_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _cfg(tmp, [{"name": "a", "url": "http://a:8000", "enabled": False}])
            with self.assertRaises(RuntimeError):
                mineru_providers(cfg)


class SlotDispatchTests(unittest.TestCase):
    def test_second_request_goes_to_the_other_instance(self) -> None:
        """A 只有一个槽位且被占住时，第二个请求必须落到 B。"""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _cfg(tmp, [
                {"name": "a", "url": "http://a:8000", "max_concurrency": 1},
                {"name": "b", "url": "http://b:8000", "max_concurrency": 1},
            ])
            journal = _journal(tmp)
            with _mineru_slot(journal, cfg) as first:
                with _mineru_slot(journal, cfg) as second:
                    self.assertNotEqual(first.name, second.name)
                    self.assertEqual({first.name, second.name}, {"a", "b"})

    def test_slot_is_released_after_use(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _cfg(tmp, [{"name": "a", "url": "http://a:8000", "max_concurrency": 1}])
            journal = _journal(tmp)
            with _mineru_slot(journal, cfg) as provider:
                self.assertEqual(provider.name, "a")
            # 释放后还能再拿到
            with _mineru_slot(journal, cfg) as provider:
                self.assertEqual(provider.name, "a")

    def test_waits_until_a_slot_frees_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _cfg(tmp, [{"name": "a", "url": "http://a:8000", "max_concurrency": 1}])
            journal = _journal(tmp)
            held = threading.Event()
            got = {}

            def hold():
                with _mineru_slot(journal, cfg):
                    held.set()
                    time.sleep(0.3)

            def waiter():
                held.wait(2)
                began = time.monotonic()
                with _mineru_slot(journal, cfg) as provider:
                    got["name"] = provider.name
                    got["waited"] = time.monotonic() - began

            threads = [threading.Thread(target=hold), threading.Thread(target=waiter)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(5)
            self.assertEqual(got["name"], "a")
            self.assertGreater(got["waited"], 0.1, "应该等到槽位释放，而不是直接抢走")

    def test_never_exceeds_total_declared_slots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _cfg(tmp, [
                {"name": "a", "url": "http://a:8000", "max_concurrency": 2},
                {"name": "b", "url": "http://b:8000", "max_concurrency": 1},
            ])
            journal = _journal(tmp)
            active = {"now": 0, "peak": 0}
            lock = threading.Lock()

            def work():
                for _ in range(6):
                    with _mineru_slot(journal, cfg):
                        with lock:
                            active["now"] += 1
                            active["peak"] = max(active["peak"], active["now"])
                        time.sleep(0.01)
                        with lock:
                            active["now"] -= 1

            threads = [threading.Thread(target=work) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(20)
            self.assertLessEqual(active["peak"], 3)

    def test_cooling_down_provider_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _cfg(tmp, [
                {"name": "a", "url": "http://a:8000", "max_concurrency": 4},
                {"name": "b", "url": "http://b:8000", "max_concurrency": 4},
            ])
            journal = _journal(tmp)
            cooldown = SharedCooldown(runtime_dir(cfg, "mineru_cooldown"), ttl_seconds=0.0)
            cooldown.mark("a", 60)
            for _ in range(4):
                with _mineru_slot(journal, cfg) as provider:
                    self.assertEqual(provider.name, "b")

    def test_exclude_steers_retry_to_another_instance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _cfg(tmp, [
                {"name": "a", "url": "http://a:8000", "max_concurrency": 4},
                {"name": "b", "url": "http://b:8000", "max_concurrency": 4},
            ])
            journal = _journal(tmp)
            providers = mineru_providers(cfg)
            with _mineru_slot(journal, cfg, providers, exclude={"a"}) as provider:
                self.assertEqual(provider.name, "b")


class SharedCooldownTests(unittest.TestCase):
    def test_mark_is_visible_to_a_separate_instance(self) -> None:
        """模拟另一个 journal 子进程：不同的 SharedCooldown 对象，同一个目录。"""
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "cd"
            writer = SharedCooldown(directory, ttl_seconds=0.0)
            reader = SharedCooldown(directory, ttl_seconds=0.0)
            self.assertFalse(reader.cooling_down("p1"))
            writer.mark("p1", 60)
            self.assertTrue(reader.cooling_down("p1"))
            writer.clear("p1")
            self.assertFalse(reader.cooling_down("p1"))

    def test_expired_cooldown_stops_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cooldown = SharedCooldown(Path(tmp), ttl_seconds=0.0)
            cooldown.mark("p1", 0.05)
            self.assertTrue(cooldown.cooling_down("p1"))
            time.sleep(0.1)
            self.assertFalse(cooldown.cooling_down("p1"))

    def test_ttl_cache_limits_filesystem_reads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cooldown = SharedCooldown(Path(tmp), ttl_seconds=60.0)
            cooldown.cooling_down("p1")
            # TTL 内的重复查询不再读盘：外部直接写文件也不会立刻被看到
            (Path(tmp) / "p1.cooldown").write_text(str(time.time() + 999), encoding="utf-8")
            self.assertFalse(cooldown.cooling_down("p1"))

    def test_disabled_when_no_directory(self) -> None:
        cooldown = SharedCooldown(None)
        self.assertFalse(cooldown.enabled)
        cooldown.mark("p1", 60)
        self.assertFalse(cooldown.cooling_down("p1"))

    def test_corrupt_cooldown_file_is_treated_as_not_cooling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "p1.cooldown").write_text("这不是数字", encoding="utf-8")
            self.assertFalse(SharedCooldown(Path(tmp), ttl_seconds=0.0).cooling_down("p1"))


class VlmSharedCooldownTests(unittest.TestCase):
    def test_failure_in_one_pool_is_visible_to_another(self) -> None:
        """两个 VlmPool 实例 = 两个 journal 子进程，冷却状态必须互相看见。"""
        from journal_cpt.services.clients import VlmPool
        from journal_cpt.tests.test_vlm_pool_dispatch import _FakeClient
        from journal_cpt.services.clients import _PooledVlmClient

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "cd"
            shared = SharedCooldown(directory, ttl_seconds=0.0)

            def build(fail_first):
                clients = [
                    _PooledVlmClient(
                        _FakeClient("p1", fail=fail_first),
                        {"name": "p1", "model": "p1", "max_concurrency": 2, "task_types": [], "capabilities": ["text"]},
                        SharedCooldown(directory, ttl_seconds=0.0),
                    ),
                    _PooledVlmClient(
                        _FakeClient("p2"),
                        {"name": "p2", "model": "p2", "max_concurrency": 2, "task_types": [], "capabilities": ["text"]},
                        SharedCooldown(directory, ttl_seconds=0.0),
                    ),
                ]
                return VlmPool(clients, cooldown_seconds=60.0), clients

            pool_a, clients_a = build(fail_first=True)
            self.assertEqual(pool_a.chat("t", "p").provider_name, "p2")
            self.assertTrue(shared.cooling_down("p1"))

            # 另一个"进程"：p1 其实是好的，但它应当因共享冷却而被跳过
            pool_b, clients_b = build(fail_first=False)
            for _ in range(3):
                self.assertEqual(pool_b.chat("t", "p").provider_name, "p2")
            self.assertEqual(clients_b[0].client.calls, 0)


if __name__ == "__main__":
    unittest.main()
