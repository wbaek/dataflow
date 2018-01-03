import threading
from contextlib import contextmanager

import tensorflow as tf
import tensorpack.dataflow as df
from tensorflow.python.training.monitored_session import _MonitoredSession

import logging
logger = logging.getLogger(__name__)


class QueueInput(threading.Thread):
    def __init__(self, ds, placeholders, repeat_infinite=False, queue_size=50):
        super(QueueInput, self).__init__()
        self.daemon = True

        self.ds = df.RepeatedData(ds, -1) if repeat_infinite else ds
        self.placeholders = placeholders
        self.queue = tf.FIFOQueue(
            queue_size,
            [ph.dtype for ph in placeholders], shapes=[ph.get_shape() for ph in placeholders]
        )

        self.op = self.queue.enqueue(self.placeholders)
        self.close_op = self.queue.close(cancel_pending_enqueues=True)

        self._itr = None
        self._sess = None
        self._lock = threading.Lock()

    def run(self):
        with self.default_sess():
            try:
                self.reinitialize()
                while True:
                    # pausable loop
                    self._lock.acquire()
                    self._lock.release()

                    dp = next(self._itr)
                    feed = dict(zip(self.placeholders, dp))
                    self.op.run(feed_dict=feed)
            except (tf.errors.CancelledError, tf.errors.OutOfRangeError, df.DataFlowTerminated):
                pass
            except Exception as e:
                if isinstance(e, RuntimeError) and 'closed Session' in str(e):
                    pass
                else:
                    logger.exception("Exception in {}:".format(self.name))
            finally:
                try:
                    self.close_op.run()
                except Exception:
                    pass
                logger.info("{} Exited.".format(self.name))

    def dequeue(self):
        return self.queue.dequeue()

    def tensors(self):
        return self.dequeue()

    def queue_size(self):
        return self.queue.size()

    def reinitialize(self):
        self._itr = self.ds.get_data()

    def pause(self):
        self._lock.acquire()

    def resume(self):
        self._lock.release()

    @contextmanager
    def default_sess(self):
        if self._sess:
            with self._sess.as_default():
                yield
        else:
            logger.warning("QueueInput {} wasn't under a default session!".format(self.name))
            yield

    def start(self, session):
        self._sess = session._tf_sess() if isinstance(session, _MonitoredSession) else session
        super(QueueInput, self).start()
