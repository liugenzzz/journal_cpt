from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from journal_cpt.core.models import JournalRecord
from journal_cpt.processing.watermark import UnreadablePdfError, clean_pdf_watermarks


def _journal(tmpdir: str, pdf: Path) -> JournalRecord:
    return JournalRecord(
        journal_id="broken",
        journal_name="broken",
        source_pdf=str(pdf),
        output_dir=str(Path(tmpdir) / "out" / "broken"),
        page_count=0,
        file_size=pdf.stat().st_size,
        file_hash="h",
        status="pending",
        created_at="t",
        pipeline_version="test",
    )


def _cfg(tmpdir: str) -> dict:
    return {
        "runtime": {"output_root": str(Path(tmpdir) / "out"), "skip_vlm": True},
        "watermark": {"enabled": True},
        "hash_chunk_size": 65536,
        "paths": {"skipped_journals": "skipped_journals.jsonl", "manifest": "manifest.jsonl"},
        "logger_name": "journal_cpt_test",
        "statuses": {"done": "done"},
    }


class UnreadablePdfDetectionTests(unittest.TestCase):
    def test_truncated_pdf_raises_typed_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "truncated.pdf"
            # 有 PDF 头但没有 %%EOF —— 就是用户遇到的那种截断文件
            pdf.write_bytes(b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog >>\nendobj\n")
            with self.assertRaises(UnreadablePdfError) as ctx:
                clean_pdf_watermarks(_journal(tmp, pdf), _cfg(tmp))
            self.assertIn("PdfStreamError", str(ctx.exception))

    def test_empty_file_raises_typed_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "empty.pdf"
            pdf.write_bytes(b"")
            with self.assertRaises(UnreadablePdfError):
                clean_pdf_watermarks(_journal(tmp, pdf), _cfg(tmp))

    def test_disabled_watermark_cleaning_does_not_touch_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "truncated.pdf"
            pdf.write_bytes(b"%PDF-1.7\n")
            cfg = _cfg(tmp)
            cfg["watermark"]["enabled"] = False
            result = clean_pdf_watermarks(_journal(tmp, pdf), cfg)
            self.assertFalse(result.cleaned)


class PipelineSkipTests(unittest.TestCase):
    def test_unreadable_pdf_is_skipped_not_failed(self) -> None:
        from journal_cpt.app import pipeline

        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "truncated.pdf"
            pdf.write_bytes(b"%PDF-1.7\n")
            journal = _journal(tmp, pdf)
            cfg = _cfg(tmp)

            result = pipeline._process_journal(journal, cfg)

            self.assertEqual(result["skipped"], "unreadable_pdf")
            self.assertNotIn("error", result)
            self.assertEqual(result["sample_count"], 0)
            self.assertIn("PdfStreamError", result["skip_reason"])

    def test_skipped_journals_are_recorded_and_counted_separately(self) -> None:
        from journal_cpt.app import pipeline

        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "out"
            output_root.mkdir(parents=True)
            journals = [
                SimpleNamespace(journal_id="good", output_dir=str(output_root / "good"), source_pdf="a.pdf"),
                SimpleNamespace(journal_id="bad", output_dir=str(output_root / "bad"), source_pdf="b.pdf"),
            ]

            def fake_process(journal, cfg, vlm=None):
                if journal.journal_id == "bad":
                    return {
                        "journal_id": "bad",
                        "output_dir": journal.output_dir,
                        "sample_count": 0,
                        "skipped": "unreadable_pdf",
                        "skip_reason": "PdfStreamError: truncated",
                        "source_pdf": journal.source_pdf,
                    }
                return {"journal_id": "good", "output_dir": journal.output_dir, "sample_count": 3}

            cfg = {
                "runtime": {
                    "input_dir": str(tmp), "output_root": str(output_root), "journal_workers": 1,
                    "skip_vlm": True, "progress": False, "recursive": True,
                },
                "paths": {"skipped_journals": "skipped_journals.jsonl"},
            }

            with patch.object(pipeline, "_apply_options", return_value=cfg), patch.object(
                pipeline, "_resolve_config_runtime_paths", return_value=cfg
            ), patch.object(pipeline, "scan_journals", return_value=journals), patch.object(
                pipeline, "_process_journal", side_effect=fake_process
            ), patch.object(pipeline, "load_config", return_value=cfg):
                results = pipeline.run_pipeline()

            self.assertEqual(len(results), 2)
            skipped = [item for item in results if item.get("skipped")]
            failed = [item for item in results if item.get("error")]
            self.assertEqual(len(skipped), 1)
            self.assertEqual(len(failed), 0, "损坏文件不该记进失败统计")

            log = output_root / "skipped_journals.jsonl"
            rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["journal_id"], "bad")
            self.assertEqual(rows[0]["source_pdf"], "b.pdf")
            self.assertIn("truncated", rows[0]["reason"])


if __name__ == "__main__":
    unittest.main()
