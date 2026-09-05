from __future__ import annotations

import json
import unittest

from journal_cpt.services.clients import parse_json_array, parse_jsonl_objects
from journal_cpt.tasks.generation import _dedupe_payload, _enforce_prompt_budget, _payload_chars


class ControlCharacterJsonTests(unittest.TestCase):
    """模型在字符串里直接敲回车是常态，不该整条响应作废。"""

    def test_raw_newline_inside_string_is_accepted(self) -> None:
        raw = '[{"instruction": "解读这张图。", "answer": "第一行。\n第二行。"}]'
        rows = parse_json_array(raw)
        self.assertEqual(rows[0]["answer"], "第一行。\n第二行。")

    def test_raw_tab_inside_string_is_accepted(self) -> None:
        raw = '[{"instruction": "q", "answer": "列一\t列二"}]'
        self.assertEqual(parse_json_array(raw)[0]["answer"], "列一\t列二")

    def test_control_chars_survive_with_other_repairs(self) -> None:
        # 同时有：代码围栏、真实换行、尾逗号
        raw = '```json\n[{"instruction": "q", "answer": "甲\n乙",},]\n```'
        rows = parse_json_array(raw)
        self.assertEqual(rows[0]["answer"], "甲\n乙")

    def test_jsonl_accepts_control_chars(self) -> None:
        raw = '{"text": "第一段。\tTab 也可以。"}'
        self.assertEqual(parse_jsonl_objects(raw), [{"text": "第一段。\tTab 也可以。"}])

    def test_genuinely_broken_json_still_raises_with_preview(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            parse_json_array("模型直接说了句白话，没有任何 JSON")
        self.assertIn("响应开头", str(ctx.exception))


class PayloadDedupeTests(unittest.TestCase):
    def test_identical_subtree_is_replaced_by_a_pointer(self) -> None:
        blocks = [{"block_id": f"b{i}", "text": "正文" * 60} for i in range(3)]
        payload = {"context": {"template_input": {"layout_blocks": blocks}, "layout_blocks": blocks}}
        result = _dedupe_payload(payload, {}, "")

        self.assertEqual(result["context"]["template_input"]["layout_blocks"], blocks)
        self.assertIsInstance(result["context"]["layout_blocks"], str)
        self.assertIn("template_input.layout_blocks", result["context"]["layout_blocks"])

    def test_first_occurrence_is_kept_verbatim(self) -> None:
        text = "整页正文。" * 100
        payload = {"a": text, "b": text, "c": text}
        result = _dedupe_payload(payload, {}, "")
        self.assertEqual(result["a"], text)
        self.assertIn("<同 a", result["b"])
        self.assertIn("<同 a", result["c"])

    def test_short_repeats_are_left_alone(self) -> None:
        # 小值（页码、column_mode 之类）到处重复是正常的，替换成引用只会更长更乱
        payload = {"x": "two_column", "y": "two_column", "z": [1, 2, 3]}
        self.assertEqual(_dedupe_payload(payload, {}, ""), payload)

    def test_distinct_content_is_never_dropped(self) -> None:
        payload = {
            "one": "内容甲。" * 60,
            "two": "内容乙。" * 60,
            "three": "内容甲。" * 60,
        }
        result = _dedupe_payload(payload, {}, "")
        self.assertEqual(result["one"], payload["one"])
        self.assertEqual(result["two"], payload["two"])   # 不同内容原样保留
        self.assertIn("<同 one", result["three"])

    def test_dedupe_is_idempotent(self) -> None:
        payload = {"a": "文本" * 200, "b": "文本" * 200}
        once = _dedupe_payload(payload, {}, "")
        twice = _dedupe_payload(json.loads(json.dumps(once)), {}, "")
        self.assertEqual(once, twice)


class PromptBudgetTests(unittest.TestCase):
    def _payload(self):
        return {
            "task_type": "t",
            "context": {
                "template_input": {"source_text": "主输入。" * 200},
                "source": {"page": {"blocks": [{"text": "块" * 400}]}},
                "page_window": [{"full_text": "页" * 2000}],
            },
        }

    def test_no_change_when_within_budget(self) -> None:
        payload = self._payload()
        before = json.loads(json.dumps(payload))
        self.assertEqual(_enforce_prompt_budget(payload, 10_000_000), before)

    def test_largest_auxiliary_field_is_elided_first(self) -> None:
        payload = self._payload()
        self.assertGreater(_payload_chars(payload), 1500, "前提：构造的 payload 必须先超预算")
        result = _enforce_prompt_budget(payload, 1500)
        self.assertLessEqual(_payload_chars(result), 1500)
        self.assertIsInstance(result["context"]["page_window"], str)
        self.assertIn("超出 prompt 预算", result["context"]["page_window"])

    def test_template_input_is_never_elided(self) -> None:
        payload = self._payload()
        result = _enforce_prompt_budget(payload, 100)
        # 预算再紧也不动任务主输入，宁可超一点也要保住它
        self.assertIsInstance(result["context"]["template_input"], dict)
        self.assertIn("source_text", result["context"]["template_input"])

    def test_zero_budget_disables_the_check(self) -> None:
        payload = self._payload()
        self.assertEqual(_enforce_prompt_budget(payload, 0), payload)


if __name__ == "__main__":
    unittest.main()
