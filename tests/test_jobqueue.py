import threading
import unittest

from voice_input.jobqueue import BoundedJobQueue, Worker


class QueueTests(unittest.TestCase):
    def test_drop_newest(self):
        drops = []
        q = BoundedJobQueue(2, "drop-newest", lambda: drops.append(1))
        self.assertEqual([q.put(i) for i in range(3)], [True, True, False])
        self.assertEqual((q.get(), q.get(), q.dropped, len(drops)), (0, 1, 1, 1))

    def test_drop_oldest(self):
        q = BoundedJobQueue(2, "drop-oldest")
        for i in range(3):
            self.assertTrue(q.put(i))
        self.assertEqual((q.get(), q.get(), q.dropped), (1, 2, 1))

    def test_invalid_args(self):
        with self.assertRaises(ValueError):
            BoundedJobQueue(0)
        with self.assertRaises(ValueError):
            BoundedJobQueue(1, "bogus")

    def test_worker_drains_and_survives_errors(self):
        q = BoundedJobQueue(5)
        done, errs = [], []

        def handler(x):
            if x == 1:
                raise RuntimeError("boom")
            done.append(x)

        w = Worker(q, handler, errs.append)
        for i in range(3):
            q.put(i)
        w.start()
        w.stop(drain=True)
        self.assertEqual(done, [0, 2])
        self.assertEqual(len(errs), 1)

    def test_put_does_not_block_while_worker_busy(self):
        q = BoundedJobQueue(1)
        gate = threading.Event()
        w = Worker(q, lambda x: gate.wait(5), lambda e: None)
        w.start()
        q.put("a")
        while q.qsize():
            pass
        q.put("b")
        self.assertFalse(q.put("c"))  # returns immediately
        gate.set()
        w.stop(drain=False)
