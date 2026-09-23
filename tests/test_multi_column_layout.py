from __future__ import annotations

import unittest
from collections import Counter

from journal_cpt.core.config_loader import load_config
from journal_cpt.core.models import JournalBlockRecord
from journal_cpt.processing import normalize as N

W = 1174.0


def _blocks(bands, rows=4, block_type="text", start_y=120.0):
    """按栏区间（归一化 x）生成正文块，每栏 rows 段。"""
    out = []
    for ci, (a, b) in enumerate(bands):
        for r in range(rows):
            out.append(
                JournalBlockRecord(
                    block_id=f"c{ci}r{r}",
                    block_type=block_type,
                    bbox=[a * W, start_y + r * 90, b * W, start_y + 70 + r * 90],
                    text="正文内容" * 20,
                )
            )
    return out


def _span(a, b, y0, y1, block_id="span", block_type="figure"):
    return JournalBlockRecord(
        block_id=block_id, block_type=block_type,
        bbox=[a * W, y0, b * W, y1], text="跨栏图表说明文字",
    )


class ColumnDetectionTests(unittest.TestCase):
    """栏位区间取自真实《航空知识》扫描页的墨迹投影测量值。"""

    def setUp(self):
        self.cfg = load_config()

    def _mode(self, blocks):
        return N._assign_columns(blocks, int(W), self.cfg)

    def test_three_column_page(self):
        # 页 80 实测：0.065-0.342 / 0.361-0.637 / 0.657-1.0
        mode = self._mode(_blocks([(0.065, 0.342), (0.361, 0.637), (0.657, 1.0)]))
        self.assertEqual(mode, "three_column")
        self.assertEqual(N.column_count(mode), 3)

    def test_three_column_blocks_get_col_labels_and_indexes(self):
        blocks = _blocks([(0.065, 0.342), (0.361, 0.637), (0.657, 1.0)])
        self._mode(blocks)
        self.assertEqual(dict(Counter(b.column for b in blocks)), {"col1": 4, "col2": 4, "col3": 4})
        self.assertEqual(sorted({b.column_index for b in blocks}), [0, 1, 2])

    def test_two_column_still_uses_left_right(self):
        # 两栏沿用 left/right，下游既有逻辑和导出标签不受影响
        blocks = _blocks([(0.064, 0.498), (0.536, 0.936)])
        self.assertEqual(self._mode(blocks), "two_column")
        self.assertEqual(dict(Counter(b.column for b in blocks)), {"left": 4, "right": 4})
        self.assertEqual({b.column_index for b in blocks}, {0, 1})

    def test_asymmetric_two_column(self):
        # 页 30/42 实测：宽栏 0.572 + 窄栏 0.274。
        # 固定阈值 0.55 会把宽栏当跨栏排除掉，多阈值扫描才能识别。
        self.assertEqual(self._mode(_blocks([(0.065, 0.637), (0.658, 0.933)])), "two_column")

    def test_single_column_page(self):
        self.assertEqual(self._mode(_blocks([(0.08, 0.92)])), "single_column")

    def test_full_page_image_falls_back_to_single(self):
        # 整版图片：所有块都超过任何 narrow 阈值，投影里找不到栏
        blocks = [_span(0.02, 0.98, 50, 950, "full")]
        mode = self._mode(blocks)
        self.assertIn(mode, {"single_column", "full_width"})
        self.assertNotEqual(mode, "unknown")

    def test_cross_column_figure_is_not_a_column(self):
        blocks = _blocks([(0.065, 0.342), (0.361, 0.637), (0.657, 1.0)])
        blocks.append(_span(0.361, 1.0, 40, 110, "fig23"))      # 跨第 2-3 栏
        mode = self._mode(blocks)
        self.assertEqual(N.column_count(mode), 3)
        fig = next(b for b in blocks if b.block_id == "fig23")
        self.assertEqual(fig.span_kind, "cross_column")
        self.assertEqual(fig.column, "full_width")
        self.assertEqual(fig.column_index, -1)

    def test_full_width_banner_spans_every_column(self):
        blocks = _blocks([(0.065, 0.342), (0.361, 0.637), (0.657, 1.0)])
        blocks.append(_span(0.065, 1.0, 40, 110, "banner"))
        self._mode(blocks)
        banner = next(b for b in blocks if b.block_id == "banner")
        self.assertEqual(banner.span_kind, "full_width")

    def test_middle_column_is_not_mistaken_for_cross_column(self):
        """三栏的中间栏正好横跨页宽中线 —— 老实现按中线劈开会把它误判成跨栏图表。"""
        blocks = _blocks([(0.065, 0.342), (0.361, 0.637), (0.657, 1.0)])
        self._mode(blocks)
        middle = [b for b in blocks if b.column == "col2"]
        self.assertEqual(len(middle), 4)
        self.assertTrue(all(b.span_kind == "single_column" for b in middle))

    def test_column_count_capped(self):
        self.assertLessEqual(N.column_count("three_column"), self.cfg["layout"]["max_columns"])


class ReadingOrderTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()

    def test_three_column_reads_left_to_right_top_to_bottom(self):
        blocks = _blocks([(0.065, 0.342), (0.361, 0.637), (0.657, 1.0)], rows=3)
        mode = N._assign_columns(blocks, int(W), self.cfg)
        ordered = N._recover_reading_order(blocks, mode)
        seq = [(N._column_index_of(b), b.bbox[1]) for b in ordered]
        self.assertEqual([c for c, _ in seq], [0, 0, 0, 1, 1, 1, 2, 2, 2])
        for ci in (0, 1, 2):
            tops = [y for c, y in seq if c == ci]
            self.assertEqual(tops, sorted(tops), "同一栏内必须从上到下")

    def test_multi_column_mode_recognises_three_columns(self):
        self.assertTrue(N.is_multi_column_mode("three_column"))
        self.assertTrue(N.is_multi_column_mode("mixed_full_width_and_three_column"))
        self.assertTrue(N.is_multi_column_mode("two_column"))
        self.assertFalse(N.is_multi_column_mode("single_column"))
        self.assertFalse(N.is_multi_column_mode("unknown"))

    def test_column_order_check_accepts_valid_three_column_order(self):
        blocks = _blocks([(0.065, 0.342), (0.361, 0.637), (0.657, 1.0)], rows=2)
        mode = N._assign_columns(blocks, int(W), self.cfg)
        N._recover_reading_order(blocks, mode)
        self.assertTrue(N._two_column_left_then_right_ok(blocks, mode))

    def test_column_order_check_rejects_backwards_jump(self):
        blocks = _blocks([(0.065, 0.342), (0.361, 0.637), (0.657, 1.0)], rows=2)
        mode = N._assign_columns(blocks, int(W), self.cfg)
        N._recover_reading_order(blocks, mode)
        # 人为把第三栏的块排到最前面（注意 reading_order 从 1 起，0 会被当成"缺失"）
        third = next(b for b in blocks if N._column_index_of(b) == 2)
        first = next(b for b in blocks if N._column_index_of(b) == 0)
        third.reading_order, first.reading_order = 1, 99
        self.assertFalse(N._two_column_left_then_right_ok(blocks, mode))


class SellerWatermarkTests(unittest.TestCase):
    """这批扫描件每页页脚都有卖家推广，OCR 后会混进正文。"""

    def setUp(self):
        self.cfg = load_config()

    def test_footer_watermark_becomes_watermark_role(self):
        text = "PDF过刊杂志收藏购买微信: bfwz888888"
        self.assertTrue(N.looks_like_seller_watermark(text, self.cfg))
        self.assertEqual(N._semantic_role("text", text, None, 0, 5, self.cfg), "watermark")

    def test_watermark_block_is_excluded_from_train_text(self):
        self.assertTrue(N._skip_from_train_text(
            JournalBlockRecord(block_id="w", block_type="text", semantic_role="watermark", text="PDF过刊…"),
            "article_body",
        ))

    def test_watermark_block_counts_as_noise_for_reading_order(self):
        block = JournalBlockRecord(block_id="w", block_type="text", semantic_role="watermark", text="PDF过刊杂志收藏购买微信")
        self.assertTrue(N._noise(block))

    def test_real_article_text_is_not_a_watermark(self):
        text = "第六届珠海航展胜利闭幕，中国航空工业创建55周年在航展上得到集中体现。"
        self.assertFalse(N.looks_like_seller_watermark(text, self.cfg))
        self.assertEqual(N._semantic_role("text", text, None, 0, 5, self.cfg), "body")


class NoisePageTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.blocks = [JournalBlockRecord(block_id="b", block_type="text", semantic_role="body", text="x")]

    def _type(self, text):
        return N._page_type(5, text, self.blocks, "three_column", self.cfg)

    def test_qr_promo_page(self):
        # 纯二维码页正文只有二三十字，凑不出多个特征，靠强特征命中
        self.assertEqual(self._type("PDF过刊杂志收藏购买微信：bfwz888888 PDF杂志购买微信"), "advertisement_or_notice")

    def test_subscription_notice_page(self):
        text = ("《航空知识》杂志社声明 尊敬的读者，2007年《航空知识》仍为每期8元，全年订价96元，"
                "集体订户（10套以上）可享受9折优惠。发行部电话：010-82317056 我社网址：www.aeroknow.com.cn")
        self.assertEqual(self._type(text), "advertisement_or_notice")

    def test_real_article_page_is_not_flagged(self):
        text = ("第六届珠海航展胜利闭幕。11月5日，第六届中国国际航空航天博览会落下帷幕，"
                "中国航空工业集团公司与波音公司签订了30架新舟60飞机的购销合同。") * 4
        self.assertEqual(self._type(text), "article_body")

    def test_article_mentioning_wechat_once_is_not_flagged(self):
        text = "本文介绍某型运输机的气动布局设计与风洞试验结果。更多内容请关注我刊微信公众号。" * 6
        self.assertEqual(self._type(text), "article_body")

    def test_advertisement_type_is_filtered_downstream(self):
        self.assertIn("advertisement_or_notice", self.cfg["routing"]["low_value_page_types"])
        self.assertIn("advertisement_or_notice", self.cfg["routing"]["skip_layout_page_types"])
        self.assertTrue(N._skip_from_train_text(self.blocks[0], "advertisement_or_notice"))


if __name__ == "__main__":
    unittest.main()
