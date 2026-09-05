from __future__ import annotations

import unittest

from journal_cpt.services.clients import (
    DEFAULT_CHAT_TEMPLATE_KWARGS,
    VlmClient,
    parse_json_array,
    parse_jsonl_objects,
    strip_reasoning,
)


class _FakeVlmClient(VlmClient):
    def __init__(self, cfg: dict) -> None:  # 跳过 requests 依赖检查
        self.cfg = cfg
        self.prompts = {"system": ""}
        self.name = str(cfg.get("name") or "vlm")


class StripReasoningTests(unittest.TestCase):
    def test_removes_paired_think_block(self) -> None:
        text = "<think>先看图 1，再列出 [步骤一, 步骤二]</think>\n[{\"a\": 1}]"
        self.assertEqual(strip_reasoning(text), '[{"a": 1}]')

    def test_removes_dangling_close_tag(self) -> None:
        text = "我先梳理一下要点。\n</think>\n[{\"a\": 1}]"
        self.assertEqual(strip_reasoning(text), '[{"a": 1}]')

    def test_removes_every_paired_block_and_keeps_text_between(self) -> None:
        text = "<think>甲</think>中间<think>乙</think>[{\"a\": 1}]"
        self.assertEqual(strip_reasoning(text), '中间[{"a": 1}]')

    def test_handles_paired_block_followed_by_dangling_close_tag(self) -> None:
        text = "<think>甲</think>还在想\n</think>\n[{\"a\": 1}]"
        self.assertEqual(strip_reasoning(text), '[{"a": 1}]')

    def test_drops_truncated_open_block(self) -> None:
        self.assertEqual(strip_reasoning("<think>还没想完就被截断了"), "")

    def test_passthrough_without_tags(self) -> None:
        self.assertEqual(strip_reasoning('  [{"a": 1}]  '), '[{"a": 1}]')


class ParseWithReasoningTests(unittest.TestCase):
    def test_json_array_after_think_containing_brackets(self) -> None:
        response = (
            "<think>题目要求返回数组，我先草拟 [{\"instruction\": \"草稿\"}]，再核对图号。</think>\n"
            "```json\n"
            '[{"instruction": "请解读图 1。", "answer": "图 1 给出频响曲线。"}]\n'
            "```"
        )
        rows = parse_json_array(response)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["instruction"], "请解读图 1。")

    def test_jsonl_objects_after_think_containing_braces(self) -> None:
        response = "<think>大概是 {text: ...} 这种形状。</think>\n" '{"text": "阻尼梁的高频能量流响应。"}'
        rows = parse_jsonl_objects(response)
        self.assertEqual(rows, [{"text": "阻尼梁的高频能量流响应。"}])

    def test_jsonl_ignores_unparseable_reasoning_lines(self) -> None:
        response = '思考：可能是 {不完整的 花括号\n{"text": "正文片段。"}'
        self.assertEqual(parse_jsonl_objects(response), [{"text": "正文片段。"}])

    def test_invalid_payload_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            parse_jsonl_objects("完全不是 JSON 的一段话")


class ExtractMessageTests(unittest.TestCase):
    def test_prefers_content_and_strips_inline_think(self) -> None:
        client = _FakeVlmClient({"name": "p1"})
        payload = {"choices": [{"message": {"content": '<think>推理</think>[{"a": 1}]'}}]}
        self.assertEqual(client._extract_message(payload), '[{"a": 1}]')

    def test_ignores_reasoning_content_when_content_present(self) -> None:
        client = _FakeVlmClient({"name": "p1"})
        payload = {"choices": [{"message": {"content": '[{"a": 1}]', "reasoning_content": "一堆推理"}}]}
        self.assertEqual(client._extract_message(payload), '[{"a": 1}]')

    def test_raises_when_only_reasoning_content(self) -> None:
        client = _FakeVlmClient({"name": "p1"})
        payload = {"choices": [{"message": {"content": "", "reasoning_content": "只想没答"}}]}
        with self.assertRaises(RuntimeError):
            client._extract_message(payload)

    def test_skips_thinking_blocks_in_list_content(self) -> None:
        client = _FakeVlmClient({"name": "p1"})
        payload = {
            "choices": [
                {"message": {"content": [{"type": "thinking", "text": "推理"}, {"type": "text", "text": '[{"a": 1}]'}]}}
            ]
        }
        self.assertEqual(client._extract_message(payload), '[{"a": 1}]')


class ChatTemplateKwargsTests(unittest.TestCase):
    def test_empty_dict_still_disables_thinking(self) -> None:
        client = _FakeVlmClient({"name": "p1", "chat_template_kwargs": {}})
        self.assertEqual(client._chat_template_kwargs(), DEFAULT_CHAT_TEMPLATE_KWARGS)

    def test_missing_key_still_disables_thinking(self) -> None:
        client = _FakeVlmClient({"name": "p1"})
        self.assertEqual(client._chat_template_kwargs(), DEFAULT_CHAT_TEMPLATE_KWARGS)

    def test_provider_override_wins(self) -> None:
        client = _FakeVlmClient({"name": "p1", "chat_template_kwargs": {"enable_thinking": True, "top_k": 20}})
        self.assertEqual(client._chat_template_kwargs(), {"enable_thinking": True, "top_k": 20})

    def test_disable_thinking_false_keeps_raw_config(self) -> None:
        client = _FakeVlmClient({"name": "p1", "disable_thinking": False, "chat_template_kwargs": {}})
        self.assertEqual(client._chat_template_kwargs(), {})


if __name__ == "__main__":
    unittest.main()
