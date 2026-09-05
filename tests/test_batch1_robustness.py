from __future__ import annotations

import json
import os
import random
import tempfile
import unittest
from pathlib import Path

from journal_cpt.core.io_utils import atomic_write_text, read_json, write_json, write_jsonl
from journal_cpt.tasks.dedup import _NearDuplicateIndex, _near_duplicate, deduplicate


class AtomicWriteTests(unittest.TestCase):
    def test_write_json_leaves_no_temp_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "state.json"
            write_json(path, {"a": 1})
            self.assertEqual(read_json(path), {"a": 1})
            self.assertEqual([p.name for p in path.parent.iterdir()], ["state.json"])

    def test_failed_write_keeps_previous_content_and_cleans_temp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            write_json(path, {"good": True})

            def boom(handle):
                handle.write('{"half":')
                raise RuntimeError("killed mid-write")

            with self.assertRaises(RuntimeError):
                atomic_write_text(path, boom)

            # 旧内容必须完好，且不留 .tmp 残骸
            self.assertEqual(read_json(path), {"good": True})
            self.assertEqual([p.name for p in path.parent.iterdir()], ["state.json"])

    def test_write_jsonl_is_atomic_too(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            write_jsonl(path, [{"i": 1}, {"i": 2}])
            self.assertEqual(path.read_text(encoding="utf-8").count("\n"), 2)

            def boom(handle):
                handle.write('{"i": 1}\n')
                raise RuntimeError("boom")

            with self.assertRaises(RuntimeError):
                atomic_write_text(path, boom)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows, [{"i": 1}, {"i": 2}])
            self.assertEqual([p.name for p in path.parent.iterdir()], ["rows.jsonl"])


class NearDuplicateIndexTests(unittest.TestCase):
    """倒排索引必须与逐条比较逐一等价，不是近似。"""

    def _reference(self, texts, threshold):
        seen: list[str] = []
        for text in texts:
            if not _near_duplicate(text, seen, threshold):
                seen.append(text)
        return seen

    def _indexed(self, texts, threshold):
        index = _NearDuplicateIndex(threshold)
        return [text for text in texts if index.add_if_new(text)]

    def test_matches_pairwise_on_random_corpus(self) -> None:
        random.seed(7)
        vocab = "阻尼梁高频能量流响应解析模型振动传递损耗实验仿真参数边界条件"
        texts = []
        for _ in range(60):
            texts.append("".join(random.choice(vocab) for _ in range(random.randint(40, 200))))
        # 掺入完全重复、近似重复和包含关系
        texts += [texts[3], texts[10][:80], texts[20] + "补充说明若干。", texts[5]]
        random.shuffle(texts)

        for threshold in (0.6, 0.8, 0.92, 0.99):
            with self.subTest(threshold=threshold):
                self.assertEqual(self._indexed(texts, threshold), self._reference(texts, threshold))

    def test_exact_duplicate_is_dropped(self) -> None:
        index = _NearDuplicateIndex(0.92)
        self.assertTrue(index.add_if_new("自由阻尼梁的高频能量流响应解析模型研究"))
        self.assertFalse(index.add_if_new("自由阻尼梁的高频能量流响应解析模型研究"))

    def test_distinct_text_is_kept(self) -> None:
        index = _NearDuplicateIndex(0.92)
        self.assertTrue(index.add_if_new("等离子体合成射流激励器的流动控制技术进展"))
        self.assertTrue(index.add_if_new("涡轮叶片气膜冷却效率的数值仿真与实验验证"))

    def test_empty_and_whitespace_text_is_kept(self) -> None:
        index = _NearDuplicateIndex(0.92)
        self.assertTrue(index.add_if_new(""))
        self.assertTrue(index.add_if_new("   "))
        self.assertTrue(index.add_if_new(""))


class DeduplicateIntegrationTests(unittest.TestCase):
    def _cfg(self):
        return {"validation": {"max_samples_per_page_task": 8, "dedup_similarity_threshold": 0.92}}

    def _pt_sample(self, sample_id, text, page=1):
        return {
            "id": sample_id,
            "task_type": "domain_knowledge_corpus",
            "output_payload": {"text": text},
            "metadata": {"journal_id": "j", "article_id": "a", "page_index": page, "task_type": "domain_knowledge_corpus"},
        }

    def test_near_duplicate_pt_samples_are_removed(self) -> None:
        base = "自由阻尼梁在高频段的能量流响应可以用解析模型描述。" * 4
        samples = [
            self._pt_sample("s1", base),
            self._pt_sample("s2", base + "另有少量补充。"),
            self._pt_sample("s3", "等离子体合成射流激励器用于附面层分离控制。" * 4, page=2),
        ]
        kept = deduplicate(samples, self._cfg())
        self.assertEqual([sample["id"] for sample in kept], ["s1", "s3"])

    def test_non_pt_tasks_are_not_near_deduped(self) -> None:
        rows = [
            {"id": "a", "task_type": "section_keypoint_summary", "question": "q", "answer": "结论基本一致",
             "output_payload": {"summary": "结论基本一致"},
             "metadata": {"journal_id": "j", "article_id": "a", "page_index": 1, "task_type": "section_keypoint_summary"}},
            {"id": "b", "task_type": "section_keypoint_summary", "question": "q2", "answer": "结论基本一致的另一句",
             "output_payload": {"summary": "结论基本一致的另一句"},
             "metadata": {"journal_id": "j", "article_id": "a", "page_index": 1, "task_type": "section_keypoint_summary"}},
        ]
        self.assertEqual(len(deduplicate(rows, self._cfg())), 2)


class ImageQualityCacheTests(unittest.TestCase):
    def test_repeated_calls_hit_cache(self) -> None:
        from journal_cpt.processing import image_quality

        image_quality._meaningful_image_file_cached.cache_clear()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "page.png"
            path.write_bytes(b"not really a png")
            cfg = {"crop_filter": {"enabled": True}}
            for _ in range(5):
                image_quality.is_meaningful_image_file(path, cfg)
            info = image_quality._meaningful_image_file_cached.cache_info()
            self.assertEqual(info.misses, 1)
            self.assertEqual(info.hits, 4)

    def test_rewritten_file_is_not_served_from_cache(self) -> None:
        from journal_cpt.processing import image_quality

        image_quality._meaningful_image_file_cached.cache_clear()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "page.png"
            path.write_bytes(b"a")
            cfg = {"crop_filter": {"enabled": True}}
            image_quality.is_meaningful_image_file(path, cfg)
            path.write_bytes(b"bb")
            os.utime(path, ns=(0, 0))  # 强制 mtime 变化
            image_quality.is_meaningful_image_file(path, cfg)
            self.assertEqual(image_quality._meaningful_image_file_cached.cache_info().misses, 2)

    def test_missing_file_is_not_meaningful(self) -> None:
        from journal_cpt.processing import image_quality

        cfg = {"crop_filter": {"enabled": True}}
        self.assertFalse(image_quality.is_meaningful_image_file(Path("/nonexistent/x.png"), cfg))


if __name__ == "__main__":
    unittest.main()


class GenerationCheckpointFlushTests(unittest.TestCase):
    """checkpoint 攒批：少量 job 只在收尾写一次，但断点信息仍完整。"""

    def _cfg(self, tmpdir, **runtime):
        base = {
            "max_workers": 1,
            "vlm_max_pending": 1,
            "reuse_samples": True,
            "generation_state_flush_every": 20,
            "generation_state_flush_seconds": 10.0,
        }
        base.update(runtime)
        return {
            "pipeline_version": "test-pipeline",
            "task_types": ["section_keypoint_summary"],
            "paths": {
                "sample_raw": "samples/raw/{task_type}.jsonl",
                "sample_generation_state": "samples/generation_state.json",
            },
            "runtime": base,
        }

    def _run(self, cfg, output_dir, job_count):
        from types import SimpleNamespace
        from unittest.mock import patch

        from journal_cpt.app import pipeline

        journal = SimpleNamespace(journal_id="j1")
        jobs = [
            SimpleNamespace(
                task_type="section_keypoint_summary",
                journal=journal,
                article=SimpleNamespace(article_id="a1"),
                page=SimpleNamespace(page_index=index),
                block=None,
            )
            for index in range(job_count)
        ]

        def fake_generate(job, *_args):
            return [{"id": f"s{job.page.page_index}", "task_type": "section_keypoint_summary"}]

        writes = {"count": 0}
        real_write = pipeline._write_generation_state

        def counting_write(*args, **kwargs):
            writes["count"] += 1
            return real_write(*args, **kwargs)

        with patch.object(pipeline, "generate_for_job", side_effect=fake_generate), patch.object(
            pipeline, "_write_generation_state", side_effect=counting_write
        ):
            pipeline._generate_samples_for_jobs(
                jobs=jobs,
                output_dir=output_dir,
                cfg=cfg,
                vlm=None,
                state=SimpleNamespace(error=lambda payload: None),
                mineru_content_hash="hash-1",
            )
        return writes["count"]

    def test_small_batch_writes_checkpoint_once_at_the_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            cfg = self._cfg(tmpdir)
            writes = self._run(cfg, output_dir, job_count=8)
            # 一次初始化写 + 一次收尾攒批写，而不是每个 job 一次
            self.assertLessEqual(writes, 2)

            payload = read_json(output_dir / "samples/generation_state.json")
            self.assertEqual(payload["mineru_content_hash"], "hash-1")
            self.assertEqual(len(payload["completed_jobs"]), 8)

    def test_flush_every_triggers_intermediate_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            cfg = self._cfg(tmpdir, generation_state_flush_every=2)
            writes = self._run(cfg, output_dir, job_count=8)
            self.assertGreaterEqual(writes, 4)
            payload = read_json(output_dir / "samples/generation_state.json")
            self.assertEqual(len(payload["completed_jobs"]), 8)

    def test_corrupt_checkpoint_is_reported_not_silently_ignored(self) -> None:
        from journal_cpt.app import pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            cfg = self._cfg(tmpdir)
            path = output_dir / "samples/generation_state.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"completed_jobs": {"a":', encoding="utf-8")  # 半截文件

            with self.assertLogs(pipeline._state_logger(cfg), level="WARNING") as captured:
                self.assertEqual(pipeline._read_generation_state(output_dir, cfg), {})
            self.assertIn("checkpoint unreadable", "\n".join(captured.output))


class GenerationProgressLogTests(unittest.TestCase):
    """生成阶段原本完全静默，几十分钟看不到动静。"""

    def _run(self, job_count, **runtime):
        import logging
        from types import SimpleNamespace
        from unittest.mock import patch

        from journal_cpt.app import pipeline

        base = {
            "max_workers": 1,
            "vlm_max_pending": 1,
            "generation_progress_every": 2,
            "generation_progress_seconds": 9999.0,
        }
        base.update(runtime)
        cfg = {
            "task_types": ["section_keypoint_summary"],
            "paths": {"sample_raw": "samples/raw/{task_type}.jsonl"},
            "runtime": base,
        }
        journal = SimpleNamespace(journal_id="j1")
        jobs = [
            SimpleNamespace(task_type="section_keypoint_summary", journal=journal,
                            page=SimpleNamespace(page_index=i), article=None, block=None)
            for i in range(job_count)
        ]
        logger = logging.getLogger("journal_cpt_progress_test")
        state = SimpleNamespace(error=lambda payload: None, logger=logger)

        def fake_generate(job, *_args):
            return [{"id": f"s{job.page.page_index}", "task_type": "section_keypoint_summary"}]

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(pipeline, "generate_for_job", side_effect=fake_generate):
                with self.assertLogs(logger, level="INFO") as captured:
                    pipeline._generate_samples_for_jobs(
                        jobs=jobs, output_dir=Path(tmp), cfg=cfg, vlm=None, state=state,
                    )
        return [line for line in captured.output if "generate progress" in line]

    def test_progress_is_logged_during_generation(self) -> None:
        lines = self._run(job_count=6)
        self.assertTrue(lines, "生成阶段必须打进度")
        self.assertIn("jobs=6/6", lines[-1])
        self.assertIn("samples=6", lines[-1])

    def test_progress_reports_eta_and_failures(self) -> None:
        lines = self._run(job_count=4)
        self.assertIn("预计剩余", lines[-1])
        self.assertIn("failed=0", lines[-1])

    def test_final_progress_is_always_emitted(self) -> None:
        # job 数少于 progress_every 时，中途不打，但收尾必须打一条
        lines = self._run(job_count=1, generation_progress_every=100)
        self.assertEqual(len(lines), 1)
        self.assertIn("jobs=1/1", lines[0])

    def test_missing_logger_on_state_does_not_crash(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import patch

        from journal_cpt.app import pipeline

        cfg = {
            "task_types": ["section_keypoint_summary"],
            "paths": {"sample_raw": "samples/raw/{task_type}.jsonl"},
            "runtime": {"max_workers": 1, "vlm_max_pending": 1},
            "logger_name": "journal_cpt_test",
        }
        journal = SimpleNamespace(journal_id="j1")
        jobs = [SimpleNamespace(task_type="section_keypoint_summary", journal=journal,
                                page=SimpleNamespace(page_index=0), article=None, block=None)]
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(pipeline, "generate_for_job", return_value=[{"id": "s", "task_type": "section_keypoint_summary"}]):
                samples = pipeline._generate_samples_for_jobs(
                    jobs=jobs, output_dir=Path(tmp), cfg=cfg, vlm=None,
                    state=SimpleNamespace(error=lambda payload: None),   # 没有 .logger
                )
        self.assertEqual(len(samples), 1)
