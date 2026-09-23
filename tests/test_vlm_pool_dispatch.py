from __future__ import annotations

import threading
import time
import unittest

from journal_cpt.services.clients import VlmPool, _PooledVlmClient


class _FakeClient:
    """替身 VlmClient：可控制耗时和失败，并记录并发峰值。"""

    def __init__(self, name: str, delay: float = 0.0, fail: bool = False) -> None:
        self.name = name
        self.delay = delay
        self.fail = fail
        self.calls = 0
        self.active = 0
        self.peak = 0
        self._lock = threading.Lock()

    def chat(self, prompt, images=None):
        with self._lock:
            self.calls += 1
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            if self.fail:
                raise RuntimeError(f"{self.name} boom")
            return f"[{{\"instruction\": \"q\", \"answer\": \"{self.name}\"}}]"
        finally:
            with self._lock:
                self.active -= 1


def _pooled(name, *, delay=0.0, fail=False, max_concurrency=1, task_types=None, capabilities=None):
    cfg = {
        "name": name,
        "model": name,
        "max_concurrency": max_concurrency,
        "task_types": task_types or [],
        "capabilities": capabilities or ["text", "image"],
    }
    return _PooledVlmClient(_FakeClient(name, delay=delay, fail=fail), cfg)


def _pool(clients, **kwargs):
    kwargs.setdefault("cooldown_seconds", 0.0)
    return VlmPool(clients, **kwargs)


class DemandDrivenDispatchTests(unittest.TestCase):
    def test_busy_provider_does_not_block_when_another_is_free(self) -> None:
        """A 满载时，第二个请求应立刻落到空闲的 B，而不是排队等 A。"""
        slow = _pooled("slow", delay=0.6, max_concurrency=1)
        fast = _pooled("fast", delay=0.0, max_concurrency=1)
        pool = _pool([slow, fast])

        started = threading.Event()

        def occupy():
            started.set()
            pool.chat("t", "p")

        worker = threading.Thread(target=occupy)
        worker.start()
        started.wait(1.0)
        time.sleep(0.05)  # 确保 slow 的槽位已被占住

        began = time.monotonic()
        response = pool.chat("t", "p")
        elapsed = time.monotonic() - began
        worker.join()

        self.assertEqual(response.provider_name, "fast")
        self.assertLess(elapsed, 0.3, "不应该阻塞在满载的 provider 上")

    def test_faster_provider_takes_more_work(self) -> None:
        """谁先跑完谁先释放槽位，也就先接到下一个任务。"""
        slow = _pooled("slow", delay=0.08, max_concurrency=1)
        fast = _pooled("fast", delay=0.005, max_concurrency=1)
        pool = _pool([slow, fast])

        def run():
            for _ in range(15):
                pool.chat("t", "p")

        threads = [threading.Thread(target=run) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertGreater(
            fast.client.calls,
            slow.client.calls,
            f"fast={fast.client.calls} slow={slow.client.calls}",
        )

    def test_never_exceeds_max_concurrency(self) -> None:
        client = _pooled("only", delay=0.02, max_concurrency=3)
        pool = _pool([client])

        def run():
            for _ in range(8):
                pool.chat("t", "p")

        threads = [threading.Thread(target=run) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(client.client.calls, 80)
        self.assertLessEqual(client.client.peak, 3)

    def test_failure_falls_back_and_releases_slot(self) -> None:
        broken = _pooled("broken", fail=True, max_concurrency=1)
        healthy = _pooled("healthy", max_concurrency=1)
        pool = _pool([broken, healthy], max_attempts=2)

        response = pool.chat("t", "p")
        self.assertEqual(response.provider_name, "healthy")
        # 失败的 provider 必须把槽位还回去，否则后续请求会永远卡在等它
        self.assertEqual(broken.free_slots, broken.max_concurrency)
        self.assertEqual(healthy.free_slots, healthy.max_concurrency)

    def test_all_providers_failing_raises_and_releases_all_slots(self) -> None:
        first = _pooled("a", fail=True, max_concurrency=1)
        second = _pooled("b", fail=True, max_concurrency=1)
        pool = _pool([first, second], max_attempts=2)

        with self.assertRaises(RuntimeError):
            pool.chat("t", "p")
        self.assertEqual(first.free_slots, 1)
        self.assertEqual(second.free_slots, 1)

    def test_image_job_only_goes_to_image_capable_provider(self) -> None:
        text_only = _pooled("text_only", capabilities=["text"], max_concurrency=4)
        visual = _pooled("visual", capabilities=["text", "image"], max_concurrency=4)
        pool = _pool([text_only, visual])

        response = pool.chat("t", "p", images=[__import__("pathlib").Path("x.png")])
        self.assertEqual(response.provider_name, "visual")
        self.assertEqual(text_only.client.calls, 0)

    def test_fallback_disabled_tries_one_provider_only(self) -> None:
        broken = _pooled("broken", fail=True, max_concurrency=1)
        healthy = _pooled("healthy", max_concurrency=1)
        pool = _pool([broken, healthy], fallback_enabled=False, max_attempts=3)

        # 关闭 fallback 时只试一个 provider；排序把空闲的排前面，两个都空闲则按轮询。
        with self.assertRaises(RuntimeError):
            for _ in range(4):
                pool.chat("t", "p")
        self.assertEqual(broken.free_slots, 1)
        self.assertEqual(healthy.free_slots, 1)


class QuotaScalingTests(unittest.TestCase):
    def test_quota_is_divided_by_journal_workers(self) -> None:
        from journal_cpt.app import pipeline

        cfg = {
            "vlm_pool": {
                "providers": [
                    {"name": "a", "max_concurrency": 16, "min_interval_seconds": 0.5},
                    {"name": "b", "max_concurrency": 3},
                ]
            }
        }
        pipeline._scale_provider_quota(cfg, 4)
        providers = cfg["vlm_pool"]["providers"]
        self.assertEqual(providers[0]["max_concurrency"], 4)
        self.assertEqual(providers[0]["min_interval_seconds"], 2.0)
        self.assertEqual(providers[1]["max_concurrency"], 1)  # 向下取整但不低于 1

    def test_single_process_keeps_declared_quota(self) -> None:
        from journal_cpt.app import pipeline

        cfg = {"vlm_pool": {"providers": [{"name": "a", "max_concurrency": 16}]}}
        pipeline._scale_provider_quota(cfg, 1)
        self.assertEqual(cfg["vlm_pool"]["providers"][0]["max_concurrency"], 16)


if __name__ == "__main__":
    unittest.main()


class ProviderNameUniquenessTests(unittest.TestCase):
    """name 是 vlm_cooldown/<name>.cooldown 的文件名，重名会让实例共用冷却状态。"""

    def _cfg(self, providers):
        return {
            "runtime": {"output_root": ""},
            "vlm_pool": {"providers": providers},
            "prompts": {"system": ""},
        }

    def _provider(self, name, port):
        return {
            "name": name,
            "model": "Qwen3.8-27B",
            "url": f"http://127.0.0.1:{port}/v1/chat/completions",
            "timeout": 60,
            "max_concurrency": 2,
        }

    def test_duplicate_names_are_rejected(self) -> None:
        cfg = self._cfg([self._provider("same", 8001), self._provider("same", 8002)])
        with self.assertRaises(ValueError) as ctx:
            VlmPool.from_config(cfg, cfg["prompts"])
        self.assertIn("same", str(ctx.exception))

    def test_unique_names_are_accepted(self) -> None:
        cfg = self._cfg([self._provider("a-8001", 8001), self._provider("b-8002", 8002)])
        pool = VlmPool.from_config(cfg, cfg["prompts"])
        self.assertEqual([c.name for c in pool.clients], ["a-8001", "b-8002"])

    def _shipped_pool(self):
        from journal_cpt.core.config_loader import load_config

        cfg = load_config()
        return VlmPool.from_config(cfg, cfg["prompts"])

    def test_shipped_config_has_unique_names(self) -> None:
        pool = self._shipped_pool()
        names = [client.name for client in pool.clients]
        self.assertEqual(len(names), len(set(names)), f"重名: {names}")

    def test_shipped_pool_composition(self) -> None:
        pool = self._shipped_pool()
        cloud = [c for c in pool.clients if "zhejianglab" in c.cfg["url"]]
        local = [c for c in pool.clients if c.cfg["url"].startswith("http://10.")]
        self.assertEqual(len(cloud), 6)
        self.assertEqual(len(local), 11, "只接在线实例；暂停的 231.26:8004 不该进池")
        # 服务端并发：本地 24 用 8，云端 128 用 32
        self.assertTrue(all(c.max_concurrency == 32 for c in cloud))
        self.assertTrue(all(c.max_concurrency == 8 for c in local))
        self.assertEqual(sum(c.max_concurrency for c in pool.clients), 280)

    def test_offline_and_paused_endpoints_are_excluded(self) -> None:
        pool = self._shipped_pool()
        urls = " ".join(c.cfg["url"] for c in pool.clients)
        for gone in ("10.107.238.7", "10.200.100.103", "10.107.226.31", "10.107.231.26:8004"):
            self.assertNotIn(gone, urls, f"{gone} 已离线/暂停，不该出现在池里")

    def test_local_pool_keeps_the_real_model_name(self) -> None:
        pool = self._shipped_pool()
        local = [c for c in pool.clients if c.cfg["url"].startswith("http://10.")]
        for client in local:
            self.assertEqual(client.model, "Qwen3.8-27B")
            self.assertEqual(client.cfg["api_key"], "local-pool-key")
            self.assertEqual(client.cfg["max_tokens"], 8192)
            self.assertEqual(client.cfg["chat_template_kwargs"], {"enable_thinking": False})

    def test_context_windows_map_to_prompt_budgets(self) -> None:
        pool = self._shipped_pool()
        by_name = {c.name: c for c in pool.clients}
        self.assertEqual(by_name["Qwen3.8-Flash-Next"].max_prompt_chars, 28000)    # 32K
        self.assertEqual(by_name["Qwen3.5-122B-A10B"].max_prompt_chars, 300000)    # 256K
        self.assertEqual(by_name["Qwen3.8-27B"].max_prompt_chars, 140000)          # 128K

    def test_021sfm_is_registered_text_only(self) -> None:
        pool = self._shipped_pool()
        for name in ("021SFM-Base", "021SFM-CoT"):
            client = next(c for c in pool.clients if c.name == name)
            self.assertNotIn("image", client.capabilities, "未确认多模态前不该接图文任务")
            self.assertFalse(client.supports("figure_table_formula_to_text", has_images=True))
            self.assertTrue(client.supports("section_keypoint_summary", has_images=False))


class PromptBudgetRoutingTests(unittest.TestCase):
    """上下文窗口小的实例不该收大 payload —— 发过去只会换回一个 400。"""

    def _pool(self, budgets):
        clients = [
            _PooledVlmClient(
                _FakeClient(f"p{i}"),
                {"name": f"p{i}", "model": f"p{i}", "max_concurrency": 4,
                 "task_types": [], "capabilities": ["text"], "max_prompt_chars": b},
            )
            for i, b in enumerate(budgets)
        ]
        return VlmPool(clients, cooldown_seconds=0.0), clients

    def test_small_prompt_can_use_the_small_provider(self) -> None:
        pool, clients = self._pool([1000, 100000])
        pool.chat("t", "x" * 500)
        self.assertEqual(sum(c.client.calls for c in clients), 1)

    def test_large_prompt_skips_the_small_provider(self) -> None:
        pool, clients = self._pool([1000, 100000])
        response = pool.chat("t", "x" * 5000)
        self.assertEqual(response.provider_name, "p1")
        self.assertEqual(clients[0].client.calls, 0, "装不下的实例不该被选中")

    def test_prompt_beyond_every_provider_raises_a_clear_error(self) -> None:
        pool, _ = self._pool([1000, 2000])
        with self.assertRaises(RuntimeError) as ctx:
            pool.chat("t", "x" * 50000)
        message = str(ctx.exception)
        self.assertIn("Prompt too long", message)
        self.assertIn("prompt_chars=50000", message)
        self.assertIn("largest_provider_budget=2000", message)

    def test_zero_budget_means_unlimited(self) -> None:
        pool, clients = self._pool([0])
        pool.chat("t", "x" * 999999)
        self.assertEqual(clients[0].client.calls, 1)

    def test_capability_error_is_distinct_from_size_error(self) -> None:
        pool, _ = self._pool([100000])
        with self.assertRaises(RuntimeError) as ctx:
            pool.chat("t", "x" * 10, images=[__import__("pathlib").Path("a.png")])
        self.assertIn("image-capable", str(ctx.exception))
