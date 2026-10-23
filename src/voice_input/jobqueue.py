"""Bounded job queue with explicit overflow policy and one worker thread."""
import queue
import threading
from typing import Callable

DROP_NEWEST = "drop-newest"
DROP_OLDEST = "drop-oldest"
POLICIES = (DROP_NEWEST, DROP_OLDEST)


class BoundedJobQueue:
    """put() never blocks. When full, the policy decides which utterance is dropped,
    and the drop is counted and reported via on_drop (never silent)."""

    def __init__(self, maxsize: int, policy: str = DROP_NEWEST,
                 on_drop: Callable[[], None] = lambda: None):
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        self._q: queue.Queue = queue.Queue(maxsize)
        self._lock = threading.Lock()
        self.policy = policy
        self.on_drop = on_drop
        self.dropped = 0

    def put(self, item) -> bool:
        """Returns True if item was queued (possibly evicting an older one)."""
        with self._lock:
            try:
                self._q.put_nowait(item)
                return True
            except queue.Full:
                pass
            if self.policy == DROP_NEWEST:
                self._drop()
                return False
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            self._drop()
            self._q.put_nowait(item)
            return True

    def _drop(self) -> None:
        self.dropped += 1
        self.on_drop()

    def get(self, timeout=None):
        return self._q.get(timeout=timeout)

    def qsize(self) -> int:
        return self._q.qsize()

    def clear(self) -> int:
        n = 0
        with self._lock:
            while True:
                try:
                    self._q.get_nowait()
                    n += 1
                except queue.Empty:
                    return n


class Worker:
    """Single thread that calls handler(item) for each queued item."""

    def __init__(self, jobs: BoundedJobQueue, handler: Callable,
                 on_error: Callable[[Exception], None]):
        self.jobs = jobs
        self.handler = handler
        self.on_error = on_error
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="transcription-worker")

    @property
    def started(self) -> bool:
        return self._thread.is_alive()

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self.jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self.handler(item)
            except Exception as exc:  # keep the worker alive
                self.on_error(exc)

    def stop(self, drain: bool, timeout: float = 30.0) -> None:
        """drain=True processes remaining jobs first; otherwise they are discarded."""
        if drain:
            deadline_hit = threading.Event()
            timer = threading.Timer(timeout, deadline_hit.set)
            timer.start()
            while self.jobs.qsize() and self._thread.is_alive() and not deadline_hit.is_set():
                deadline_hit.wait(0.05)
            timer.cancel()
        else:
            self.jobs.clear()
        self._stop.set()
        self._thread.join()
