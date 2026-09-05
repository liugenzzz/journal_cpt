from __future__ import annotations

import unittest

from journal_cpt.services.clients import parse_json_array


class JsonArrayParsingTests(unittest.TestCase):
    def test_repairs_invalid_backslashes_inside_json_strings(self) -> None:
        response = r"""[
          {
            "instruction": "请解读这张图。",
            "answer": "启动条件包含 $55.3\%$ 门槛，\alpha 增大。\n第二行仍应保留为换行。"
          }
        ]"""

        rows = parse_json_array(response)

        self.assertEqual(rows[0]["answer"], "启动条件包含 $55.3\\%$ 门槛，\\alpha 增大。\n第二行仍应保留为换行。")


if __name__ == "__main__":
    unittest.main()
