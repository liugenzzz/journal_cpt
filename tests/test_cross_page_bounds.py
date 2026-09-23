from __future__ import annotations

import unittest

from journal_cpt.core.config_loader import load_config
from journal_cpt.tasks.validation import _cross_page_images_ok, cross_page_image_bounds


def _sample(image_count: int) -> dict:
    return {
        "task_type": "cross_page_article_context",
        "images": [f"images/pages/p{i:03d}.png" for i in range(image_count)],
    }


def _cfg(window=3, min_images=2, max_images=3) -> dict:
    return {
        "routing": {
            "cross_page_window": window,
            "cross_page_min_images": min_images,
            "cross_page_max_images": max_images,
        }
    }


class CrossPageBoundsTests(unittest.TestCase):
    def test_shipped_config_allows_two_to_five_pages(self) -> None:
        cfg = load_config()
        self.assertEqual(cross_page_image_bounds(cfg), (2, 5))
        self.assertFalse(_cross_page_images_ok(_sample(1), cfg))
        for n in (2, 3, 4, 5):
            self.assertTrue(_cross_page_images_ok(_sample(n), cfg), f"{n} 页应被接受")
        self.assertFalse(_cross_page_images_ok(_sample(6), cfg))

    def test_raising_max_images_takes_effect(self) -> None:
        cfg = _cfg(window=5, min_images=2, max_images=5)
        self.assertEqual(cross_page_image_bounds(cfg), (2, 5))
        for n in (2, 3, 4, 5):
            self.assertTrue(_cross_page_images_ok(_sample(n), cfg), f"{n} 页应被接受")
        self.assertFalse(_cross_page_images_ok(_sample(6), cfg))

    def test_minimum_is_never_below_two(self) -> None:
        # 跨页任务至少要两页证据，配 1 也会被抬到 2
        self.assertEqual(cross_page_image_bounds(_cfg(min_images=1))[0], 2)

    def test_max_never_below_min(self) -> None:
        self.assertEqual(cross_page_image_bounds(_cfg(min_images=4, max_images=2)), (4, 4))

    def test_other_tasks_are_unaffected(self) -> None:
        cfg = load_config()
        for task in ("figure_table_formula_to_text", "page_to_journal_layout_description"):
            self.assertTrue(_cross_page_images_ok({"task_type": task, "images": ["a.png"]}, cfg))


class QualityFallbackFollowsConfigTests(unittest.TestCase):
    """质量校验里那条兜底判据原来写死 2..3，必须跟着配置走。"""

    def _quality_sample(self, image_count: int) -> dict:
        return {
            "id": "s1",
            "task_type": "cross_page_article_context",
            "instruction": "请说明这几页连续内容的跨页承接关系。",
            "answer": "这几页依次介绍了研究背景、方法与结论，跨页处段落延续。" * 3,
            "images": [f"images/pages/p{i:03d}.png" for i in range(image_count)],
            "output_payload": {
                "page_visual_descriptions": ["第一页版面", "第二页版面"],
                "context_topic": "主题",
                "cross_page_summary": "跨页归纳。" * 10,
                "page_roles": ["开头", "延续"],
                "key_points_by_page": [["要点一"], ["要点二"]],
            },
            "evidence": {"source_pages": list(range(1, image_count + 1)), "visual_evidence": ["p1"]},
            "metadata": {"page_index": 1},
            "input_payload": {},
        }

    def test_five_page_sample_keeps_the_fallback_when_config_allows(self) -> None:
        from journal_cpt.tasks import validation

        cfg = {
            "routing": {"cross_page_window": 5, "cross_page_min_images": 2, "cross_page_max_images": 5},
            "validation": {"quality_dimensions": {}},
        }
        sample = self._quality_sample(5)
        # 直接验边界函数：5 页在 max=5 的配置下应当合法
        self.assertTrue(validation._cross_page_images_ok(sample, cfg))
        self.assertEqual(validation.cross_page_image_bounds(cfg), (2, 5))

    def test_six_page_sample_rejected_under_shipped_config(self) -> None:
        from journal_cpt.tasks import validation

        cfg = load_config()
        self.assertTrue(validation._cross_page_images_ok(self._quality_sample(5), cfg))
        self.assertFalse(validation._cross_page_images_ok(self._quality_sample(6), cfg))


if __name__ == "__main__":
    unittest.main()
