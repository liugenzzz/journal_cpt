from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from journal_cpt.app import pipeline


class IncrementalRawSampleWriteTests(unittest.TestCase):
    def test_successful_job_samples_are_written_before_entire_batch_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            cfg = {
                "task_types": ["section_keypoint_summary"],
                "paths": {"sample_raw": "samples/raw/{task_type}.jsonl"},
                "runtime": {"max_workers": 2, "vlm_max_pending": 2},
            }
            state = SimpleNamespace(error=lambda payload: None)
            journal = SimpleNamespace(journal_id="journal-1")
            first_job = SimpleNamespace(task_type="section_keypoint_summary", journal=journal)
            second_job = SimpleNamespace(task_type="section_keypoint_summary", journal=journal)
            second_job_started = threading.Event()
            release_second_job = threading.Event()

            def fake_generate_for_job(job: object, *_args: object) -> list[dict[str, object]]:
                if job is second_job:
                    second_job_started.set()
                    release_second_job.wait(timeout=5)
                    return [{"id": "second", "task_type": "section_keypoint_summary"}]
                return [{"id": "first", "task_type": "section_keypoint_summary"}]

            result_holder: dict[str, object] = {}

            def run_generation() -> None:
                try:
                    result_holder["samples"] = pipeline._generate_samples_for_jobs(
                        jobs=[first_job, second_job],
                        output_dir=output_dir,
                        cfg=cfg,
                        vlm=None,
                        state=state,
                    )
                except Exception as exc:  # pragma: no cover - surfaced after join
                    result_holder["error"] = exc

            with patch.object(pipeline, "generate_for_job", side_effect=fake_generate_for_job):
                worker = threading.Thread(target=run_generation)
                worker.start()
                try:
                    self.assertTrue(second_job_started.wait(timeout=2))
                    raw_path = output_dir / "samples/raw/section_keypoint_summary.jsonl"
                    written_rows = self._wait_for_rows(raw_path, expected_count=1, timeout=1.5)
                    self.assertEqual([row["id"] for row in written_rows], ["first"])
                finally:
                    release_second_job.set()
                    worker.join(timeout=5)

            self.assertFalse(worker.is_alive())
            if "error" in result_holder:
                raise result_holder["error"]  # type: ignore[misc]

    def test_generation_resumes_from_completed_raw_sample_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            cfg = {
                "pipeline_version": "test-pipeline",
                "task_types": ["section_keypoint_summary"],
                "paths": {
                    "sample_raw": "samples/raw/{task_type}.jsonl",
                    "sample_generation_state": "samples/generation_state.json",
                },
                "runtime": {"max_workers": 1, "vlm_max_pending": 1, "reuse_samples": True},
                "default_block_id": "page",
            }
            state = SimpleNamespace(error=lambda payload: None)
            journal = SimpleNamespace(journal_id="journal-1")
            first_page = SimpleNamespace(
                article_id="article-1",
                page_index=1,
                page_label="1",
            )
            second_page = SimpleNamespace(
                article_id="article-1",
                page_index=2,
                page_label="2",
            )
            first_job = SimpleNamespace(task_type="section_keypoint_summary", journal=journal, page=first_page, article=None, block=None)
            second_job = SimpleNamespace(task_type="section_keypoint_summary", journal=journal, page=second_page, article=None, block=None)
            raw_path = output_dir / "samples/raw/section_keypoint_summary.jsonl"
            raw_path.parent.mkdir(parents=True)
            first_sample = {
                "id": "first",
                "task_type": "section_keypoint_summary",
                "metadata": {
                    "task_type": "section_keypoint_summary",
                    "journal_id": "journal-1",
                    "article_id": "article-1",
                    "page_index": 1,
                    "block_id": "page",
                },
            }
            raw_path.write_text(json.dumps(first_sample, ensure_ascii=False) + "\n", encoding="utf-8")
            generation_state_path = output_dir / "samples/generation_state.json"
            generation_state_path.write_text(
                json.dumps(
                    {
                        "mineru_content_hash": "hash-1",
                        "pipeline_version": "test-pipeline",
                        "completed_jobs": {
                            "section_keypoint_summary|journal-1|article-1|1|page": {
                                "task_type": "section_keypoint_summary",
                                "sample_count": 1,
                            }
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            generated_job_ids: list[int] = []

            def fake_generate_for_job(job: object, *_args: object) -> list[dict[str, object]]:
                generated_job_ids.append(job.page.page_index)  # type: ignore[attr-defined]
                return [
                    {
                        "id": "second",
                        "task_type": "section_keypoint_summary",
                        "metadata": {
                            "task_type": "section_keypoint_summary",
                            "journal_id": "journal-1",
                            "article_id": "article-1",
                            "page_index": 2,
                            "block_id": "page",
                        },
                    }
                ]

            with patch.object(pipeline, "generate_for_job", side_effect=fake_generate_for_job):
                samples = pipeline._generate_samples_for_jobs(
                    jobs=[first_job, second_job],
                    output_dir=output_dir,
                    cfg=cfg,
                    vlm=None,
                    state=state,
                    mineru_content_hash="hash-1",
                )

            self.assertEqual(generated_job_ids, [2])
            self.assertEqual([sample["id"] for sample in samples], ["first", "second"])
            raw_rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["id"] for row in raw_rows], ["first", "second"])

    def _wait_for_rows(self, path: Path, expected_count: int, timeout: float) -> list[dict[str, object]]:
        deadline = time.monotonic() + timeout
        rows: list[dict[str, object]] = []
        while time.monotonic() < deadline:
            if path.exists():
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                if len(rows) >= expected_count:
                    return rows
            time.sleep(0.05)
        return rows


if __name__ == "__main__":
    unittest.main()
