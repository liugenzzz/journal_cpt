from __future__ import annotations

import io
import multiprocessing
import os
import sys
import unittest
from unittest.mock import patch

from journal_cpt.core.logging_utils import _ProgressAwareHandler
from journal_cpt.core.progress import ProgressBar


class _FakeTty(io.StringIO):
    def isatty(self) -> bool:
        return True


def _child_render(queue) -> None:
    """在 fork 出来的子进程里模拟"打一条日志触发 redraw"。"""
    stream = _FakeTty()
    with patch.object(sys, "stderr", stream):
        bar = ProgressBar.active
        queue.put(
            {
                "inherited_bar": bar is not None,
                "inherited_done": getattr(bar, "done", None),
                "wrote": (bar.redraw() or stream.getvalue()) if bar is not None else "",
            }
        )


class ForkedProgressBarTests(unittest.TestCase):
    def test_child_process_does_not_repaint_the_bar(self) -> None:
        """子进程继承的是 done=0 的副本，重绘会把父进程真实计数覆盖成 0/N。"""
        if multiprocessing.get_start_method(allow_none=True) not in (None, "fork"):
            self.skipTest("需要 fork 启动方式")
        try:
            ctx = multiprocessing.get_context("fork")
        except ValueError:
            self.skipTest("本平台不支持 fork")

        stream = _FakeTty()
        with patch.object(sys, "stderr", stream):
            with ProgressBar(100, desc="期刊", enabled=True) as bar:
                for _ in range(7):
                    bar.advance(ok=True)
                self.assertEqual(bar.done, 7)

                queue = ctx.Queue()
                child = ctx.Process(target=_child_render, args=(queue,))
                child.start()
                payload = queue.get(timeout=10)
                child.join(10)

        self.assertTrue(payload["inherited_bar"], "前提：子进程确实继承了 ProgressBar.active")
        self.assertEqual(payload["inherited_done"], 7, "fork 时的快照")
        self.assertEqual(payload["wrote"], "", "子进程不得往 stderr 写任何进度条内容")

    def test_owner_process_still_renders(self) -> None:
        stream = _FakeTty()
        with patch.object(sys, "stderr", stream):
            with ProgressBar(10, desc="期刊", enabled=True) as bar:
                bar.advance(ok=True)
        output = stream.getvalue()
        self.assertIn("1/10", output)
        self.assertIn("期刊", output)

    def test_pid_change_disables_rendering(self) -> None:
        stream = _FakeTty()
        with patch.object(sys, "stderr", stream):
            bar = ProgressBar(10, desc="期刊", enabled=True)
            bar.__enter__()
            stream.truncate(0), stream.seek(0)
            with patch.object(os, "getpid", return_value=bar._owner_pid + 1):
                bar.advance(ok=True)
                bar.redraw()
                bar.clear_line()
                bar.set_current("x")
            self.assertEqual(stream.getvalue(), "")
        ProgressBar.active = None

    def test_log_handler_in_child_writes_no_bar(self) -> None:
        """日志 handler 是子进程唯一会碰进度条的地方。"""
        import logging

        stream = _FakeTty()
        with patch.object(sys, "stderr", stream):
            with ProgressBar(10, desc="期刊", enabled=True) as bar:
                bar.advance(ok=True)
                stream.truncate(0), stream.seek(0)
                handler = _ProgressAwareHandler(stream)
                record = logging.LogRecord("t", logging.INFO, __file__, 1, "子进程日志", None, None)
                with patch.object(os, "getpid", return_value=bar._owner_pid + 1):
                    handler.emit(record)
                # 必须在 with 内取值：__exit__ 会在父进程身份下渲染最终状态
                written = stream.getvalue()
        self.assertIn("子进程日志", written)
        self.assertNotIn("/10", written, "不该带出进度条")


if __name__ == "__main__":
    unittest.main()
