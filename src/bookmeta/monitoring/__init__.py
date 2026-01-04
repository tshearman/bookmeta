import curses
import logging
import shutil
import time
from collections import deque
from math import sqrt
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Generic, TypeVar

T = TypeVar("T")


LOGGER = logging.getLogger(__name__)


class _RunningStats:
    """Online mean/stddev to avoid storing all durations."""

    def __init__(self) -> None:
        self.count: int = 0
        self.mean: float = 0.0
        self._m2: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self._m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self._m2 / (self.count - 1)

    @property
    def stddev(self) -> float:
        return sqrt(self.variance)


class _MonitorLogHandler(logging.Handler):
    """Capture debug messages for display in the monitor window."""

    def __init__(self, monitor: "QueueMonitor") -> None:
        super().__init__(level=logging.DEBUG)
        self.monitor = monitor

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:
            try:
                message = record.getMessage()
            except Exception:
                return
        self.monitor._debug_lines.append(message)
        self.monitor.touch("debug")


class QueueMonitor:
    """Lightweight, non-blocking queue monitor that renders a live table."""

    def __init__(
        self,
        *,
        title: str = "Queue Monitor",
        refresh_interval: float = 0.25,
        enabled: bool = True,
    ) -> None:
        self.title = title
        self.refresh_interval = refresh_interval
        self.enabled = enabled
        self._queues: dict[str, "MonitoredQueue"] = {}
        self._done: dict[str, bool] = {}
        self._updates: Queue[str | None] = Queue()
        self._stop = Event()
        self._thread: Thread | None = None
        self._stats: dict[str, _RunningStats] = {}
        self._wait_stats: dict[str, _RunningStats] = {}
        self._debug_lines: deque[str] = deque(maxlen=50)
        self._active: dict[str, int] = {}
        self._log_handler = _MonitorLogHandler(self)
        self._lock = Lock()

    def register(self, name: str, queue: "MonitoredQueue") -> None:
        if not self.enabled:
            return
        self._queues[name] = queue
        self._done.setdefault(name, False)
        self._stats.setdefault(name, _RunningStats())
        self._wait_stats.setdefault(name, _RunningStats())
        with self._lock:
            self._active.setdefault(name, 0)
        self.touch(name)

    def start(self) -> None:
        if not self.enabled or self._thread:
            return
        LOGGER.addHandler(self._log_handler)
        self._thread = Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.touch(None)

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop.set()
        self.touch(None)
        if self._thread:
            self._thread.join(timeout=1.0)
        LOGGER.removeHandler(self._log_handler)
        self._print_final_table()

    def touch(self, name: str | None) -> None:
        if self.enabled:
            self._updates.put(name)

    def mark_done(self, name: str) -> None:
        if not self.enabled:
            return
        self._done[name] = True
        self.touch(name)

    def record_duration(self, name: str, duration: float) -> None:
        if not self.enabled:
            return
        stats = self._stats.setdefault(name, _RunningStats())
        stats.update(duration)
        self.touch(name)

    def record_wait(self, name: str, duration: float) -> None:
        if not self.enabled:
            return
        stats = self._wait_stats.setdefault(name, _RunningStats())
        stats.update(duration)
        self.touch(name)

    def work_started(self, name: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._active[name] = self._active.get(name, 0) + 1
        self.touch(name)

    def work_finished(self, name: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._active[name] = max(0, self._active.get(name, 0) - 1)
        self.touch(name)

    def _loop(self) -> None:
        stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        curses.curs_set(0)
        stdscr.nodelay(True)
        try:
            while not self._stop.is_set():
                try:
                    self._updates.get(timeout=self.refresh_interval)
                except Empty:
                    pass
                else:
                    while True:
                        try:
                            self._updates.get_nowait()
                        except Empty:
                            break
                self._render(stdscr)
                try:
                    ch = stdscr.getch()
                    if ch in (ord("q"), ord("Q")):
                        self._stop.set()
                        break
                except Exception:
                    pass
        finally:
            try:
                curses.curs_set(1)
            except Exception:
                pass
            curses.echo()
            curses.nocbreak()
            curses.endwin()

    def _render(self, stdscr: curses.window) -> None:
        height, width = stdscr.getmaxyx()
        output_lines = self._render_table_lines(width, include_debug=True)
        stdscr.erase()
        for idx, line in enumerate(output_lines):
            if idx >= height - 1:
                break
            stdscr.addnstr(idx, 0, line, max(0, width - 1))
        stdscr.refresh()

    def _render_table_lines(self, width: int, *, include_debug: bool) -> list[str]:
        now = time.strftime("%H:%M:%S")
        queue_rows = self._build_queue_rows()
        worker_rows = self._build_worker_rows()

        sections: list[str] = []
        if queue_rows:
            queue_columns = [
                ("Queue", "<"),
                ("Size", ">"),
                ("Capacity", ">"),
                ("Put/Get", "<"),
                ("Done", "^"),
            ]
            sections.append(
                self._table(
                    f"{self.title} — Queues ({now})", queue_columns, queue_rows, width
                )
            )
        if worker_rows:
            worker_columns = [
                ("Stage", "<"),
                ("Active", ">"),
                ("Proc avg±sd (s)", ">"),
                ("Wait avg±sd (s)", ">"),
            ]
            sections.append(
                self._table("Worker Timing", worker_columns, worker_rows, width)
            )

        output_lines = "\n\n".join(sections).splitlines()
        if include_debug:
            debug_lines = list(self._debug_lines)
            if debug_lines:
                output_lines.append("")
                output_lines.append("Debug (latest):")
                output_lines.extend(debug_lines)
        output_lines.extend(["\n"])
        return output_lines

    def _build_queue_rows(self) -> list[tuple[str, str, str, str, str]]:
        rows: list[tuple[str, str, str, str, str]] = []
        for name, queue in self._queues.items():
            size = str(queue.qsize())
            cap_display = "∞" if queue.maxsize <= 0 else str(queue.maxsize)
            done = "✔" if self._done.get(name) else ""
            flow = f"{queue.put_count}/{queue.get_count}"
            rows.append((name, size, cap_display, flow, done))
        return rows

    def _build_worker_rows(self) -> list[tuple[str, str, str, str]]:
        rows: list[tuple[str, str, str, str]] = []
        for name in self._queues.keys():
            stats = self._stats.get(name)
            proc_mean = stats.mean if stats and stats.count else None
            proc_stddev = stats.stddev if stats and stats.count > 1 else 0.0
            proc_display = (
                f"{proc_mean:.2f}±{proc_stddev:.2f}" if proc_mean is not None else "-"
            )
            wait_stats = self._wait_stats.get(name)
            wait_mean = wait_stats.mean if wait_stats and wait_stats.count else None
            wait_stddev = (
                wait_stats.stddev if wait_stats and wait_stats.count > 1 else 0.0
            )
            wait_display = (
                f"{wait_mean:.2f}±{wait_stddev:.2f}" if wait_mean is not None else "-"
            )
            with self._lock:
                active = self._active.get(name, 0)
            rows.append((name, str(active), proc_display, wait_display))
        return rows

    def _print_final_table(self) -> None:
        if not self.enabled or not self._queues:
            return
        width = shutil.get_terminal_size((80, 20)).columns
        lines = self._render_table_lines(width, include_debug=False)
        print("\n".join(lines))

    def _table(
        self,
        header: str,
        columns: list[tuple[str, str]],
        rows: list[tuple[str, ...]],
        width: int,
    ) -> str:
        if not rows:
            return header

        col_widths: list[int] = []
        for idx, (heading, _) in enumerate(columns):
            max_content = max((len(str(r[idx])) for r in rows), default=0)
            col_widths.append(max(len(heading), max_content))

        border = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

        def _fmt_row(values: tuple[str, ...]) -> str:
            cells = []
            for value, (_, align), width_val in zip(values, columns, col_widths):
                val_str = str(value)
                cells.append(f" {val_str:{align}{width_val}} ")
            return "|" + "|".join(cells) + "|"

        lines = [header, border]
        header_cells = []
        for (heading, align), width_val in zip(columns, col_widths):
            header_cells.append(f" {heading:{align}{width_val}} ")
        lines.append("|" + "|".join(header_cells) + "|")
        lines.append(border)
        for row in rows:
            lines.append(_fmt_row(row))
        lines.append(border)
        return "\n".join(lines)


class MonitoredQueue(Queue):
    """Queue that notifies a monitor on size changes without changing semantics."""

    def __init__(
        self,
        name: str,
        maxsize: int = 0,
        monitor: QueueMonitor | None = None,
    ) -> None:
        super().__init__(maxsize=maxsize)
        self.name = name
        self.monitor = monitor
        self.put_count = 0
        self.get_count = 0
        if monitor:
            monitor.register(name, self)

    def _notify(self) -> None:
        if self.monitor:
            try:
                self.monitor.touch(self.name)
            except Exception:
                pass

    def put(self, item, block=True, timeout=None):
        result = super().put(item, block=block, timeout=timeout)
        self.put_count += 1
        self._notify()
        return result

    def get(self, block=True, timeout=None):
        item = super().get(block=block, timeout=timeout)
        self.get_count += 1
        self._notify()
        return item

    def put_nowait(self, item):
        result = super().put_nowait(item)
        self.put_count += 1
        self._notify()
        return result

    def get_nowait(self):
        item = super().get_nowait()
        self.get_count += 1
        self._notify()
        return item


__all__ = ["QueueMonitor", "MonitoredQueue"]


class TimedItem(Generic[T]):
    """Payload wrapper carrying enqueue time for wait-time measurement."""

    def __init__(self, obj: T, enqueued_time: float | None = None) -> None:
        self.obj = obj
        self.enqueued_time = enqueued_time
