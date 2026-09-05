from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from journal_cpt.core.models import JournalBlockRecord, JournalPageRecord, JournalRecord
from journal_cpt.processing.normalize import _link_visual_context
from journal_cpt.tasks.routing import build_sample_jobs


class SubfigureGroupingTests(unittest.TestCase):
    def test_groups_two_subfigures_and_keeps_following_english_caption(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            page = JournalPageRecord(
                journal_id="j1",
                journal_name="journal",
                source_pdf="source.pdf",
                page_index=1,
                page_label="1",
                page_type="article_body",
                article_id="a1",
                article_title="article",
                article_role="body",
                page_image="images/pages/p001.png",
                width=600,
                height=760,
                column_mode="single_column",
                blocks=[
                    JournalBlockRecord("p001_b001", "figure", bbox=[100, 80, 500, 180], text="(a) Ref configuration", reading_order=1, column="left"),
                    JournalBlockRecord("p001_b002", "figure", bbox=[100, 200, 500, 300], text="(b) HCT configuration 图 8 两种构型云图 Fig. 8 Contours for both configurations", reading_order=2, column="left"),
                    JournalBlockRecord("p001_b003", "figure", semantic_role="figure_caption", bbox=[100, 330, 500, 430], text="Fig. 8 English caption for both configurations (a) Ref configuration", reading_order=3, column="left"),
                    JournalBlockRecord("p001_b004", "figure", bbox=[100, 450, 500, 550], text="(b) HCT configuration 图 9 压力系数云图 Fig. 9 Pressure coefficient contours", reading_order=4, column="left"),
                ],
            )
            cfg = {
                "routing": {"visual_block_types": ["figure", "table", "formula"]},
                "generation": {"max_related_context_blocks": 8, "max_subfigures_per_crop": 2},
                "crop_filter": {"padding": 4},
                "paths": {"block_images": "images/blocks/p{page_no:03d}/{block_id}_{block_type}.png"},
            }

            _link_visual_context(page, cfg, output_dir)

            link_by_target = {link["target_block_id"]: link for link in page.figure_links}
            fig8_link = link_by_target["p001_b001"]
            fig9_link = link_by_target["p001_b003"]

            self.assertEqual(fig8_link["group_member_block_ids"], ["p001_b001", "p001_b002"])
            self.assertIn("p001_b003", fig8_link["caption_block_ids"])
            self.assertIn("(a)", fig8_link["caption_note"])
            self.assertIn("(b)", fig8_link["caption_note"])
            self.assertIn("target_image", fig8_link)

            self.assertEqual(fig9_link["group_member_block_ids"], ["p001_b003", "p001_b004"])
            self.assertIn("图 9", fig9_link["object_label"])
            self.assertIn("target_image", fig9_link)

    def test_chunks_three_subfigures_and_notes_remaining_parts(self) -> None:
        page = JournalPageRecord(
            journal_id="j1",
            journal_name="journal",
            source_pdf="source.pdf",
            page_index=1,
            page_label="1",
            page_type="article_body",
            article_id="a1",
            article_title="article",
            article_role="body",
            page_image="images/pages/p001.png",
            width=600,
            height=760,
            column_mode="single_column",
            blocks=[
                JournalBlockRecord("p001_b001", "figure", bbox=[100, 80, 500, 180], text="(a) first part", reading_order=1, column="left"),
                JournalBlockRecord("p001_b002", "figure", bbox=[100, 200, 500, 300], text="(b) second part", reading_order=2, column="left"),
                JournalBlockRecord("p001_b003", "figure", bbox=[100, 320, 500, 420], text="(c) third part 图 11 三子图结果 Fig. 11 Three-part result", reading_order=3, column="left"),
            ],
        )
        cfg = {
            "routing": {"visual_block_types": ["figure", "table", "formula"]},
            "generation": {"max_related_context_blocks": 8, "max_subfigures_per_crop": 2},
            "crop_filter": {"padding": 4},
            "paths": {"block_images": "images/blocks/p{page_no:03d}/{block_id}_{block_type}.png"},
        }

        _link_visual_context(page, cfg, None)

        link_by_target = {link["target_block_id"]: link for link in page.figure_links}
        first_chunk = link_by_target["p001_b001"]
        second_chunk = link_by_target["p001_b003"]

        self.assertEqual(first_chunk["group_member_block_ids"], ["p001_b001", "p001_b002"])
        self.assertEqual(first_chunk["chunk_count"], 2)
        self.assertIn("(c)", first_chunk["caption_note"])
        self.assertEqual(second_chunk["group_member_block_ids"], ["p001_b003"])
        self.assertEqual(second_chunk["chunk_index"], 2)
        self.assertIn("已分片抽取", second_chunk["caption_note"])

    def test_routing_skips_non_target_members_of_subfigure_group(self) -> None:
        page = JournalPageRecord(
            journal_id="j1",
            journal_name="journal",
            source_pdf="source.pdf",
            page_index=1,
            page_label="1",
            page_type="article_body",
            article_id="a1",
            article_title="article",
            article_role="body",
            page_image="images/pages/p001.png",
            width=600,
            height=760,
            column_mode="single_column",
            full_text="正文内容足够用于训练。图 8 展示两张子图。",
            blocks=[
                JournalBlockRecord("p001_b001", "figure", bbox=[100, 80, 500, 180], text="(a) first part", reading_order=1, column="left"),
                JournalBlockRecord("p001_b002", "figure", bbox=[100, 200, 500, 300], text="(b) second part 图 8 两子图结果 Fig. 8 Two-part result", reading_order=2, column="left"),
            ],
        )
        cfg = {
            "routing": {
                "enabled_tasks": {
                    "page_to_journal_layout_description": False,
                    "article_metadata_extraction": False,
                    "two_column_reading_order_reconstruction": False,
                    "section_heading_scope_alignment": False,
                    "section_keypoint_summary": False,
                    "figure_table_formula_to_text": True,
                    "method_experiment_condition_extraction": False,
                    "evidence_to_claim_chain": False,
                    "cross_page_article_context": False,
                    "article_contribution_conclusion": False,
                    "domain_knowledge_corpus": False,
                },
                "low_value_page_types": [],
                "text_block_types": ["text", "paragraph", "list"],
                "visual_block_types": ["figure", "table", "formula"],
                "min_page_text_chars": 1,
                "min_block_text_chars": 1,
                "min_section_text_chars": 1,
                "cross_page_min_images": 2,
                "cross_page_max_images": 3,
                "cross_page_window": 3,
                "article_window": 8,
                "method_keywords": [],
                "claim_keywords": [],
                "conclusion_keywords": [],
            },
            "generation": {"max_related_context_blocks": 8, "max_subfigures_per_crop": 2},
            "crop_filter": {"padding": 4},
            "paths": {"block_images": "images/blocks/p{page_no:03d}/{block_id}_{block_type}.png"},
        }
        _link_visual_context(page, cfg, None)
        journal = JournalRecord("j1", "journal", "source.pdf", ".", 1, 1, "hash", "ready", "now", "test")

        jobs = build_sample_jobs(journal, [page], [], cfg)

        self.assertEqual([job.block.block_id for job in jobs], ["p001_b001"])
        self.assertTrue(jobs[0].source["figure_link"]["is_subfigure_group"])


if __name__ == "__main__":
    unittest.main()
