"""Thread synchronization primitives for concurrent access."""

from contextlib import contextmanager
from threading import Condition, Lock


class ReadWriteLock:
    """
    A fair reader-writer lock.

    Allows multiple concurrent readers OR one exclusive writer.
    Writers are prioritized to prevent starvation.

    Usage:
        lock = ReadWriteLock()

        # Multiple readers can execute concurrently
        with lock.read():
            data = read_from_resource()

        # Writers get exclusive access
        with lock.write():
            write_to_resource(data)
    """

    def __init__(self):
        """Initialize the read-write lock."""
        self._lock = Lock()
        self._readers = 0  # Current number of active readers
        self._writers = 0  # Current number of active writers (0 or 1)
        self._waiting_writers = 0  # Number of writers waiting for access

        # Condition variables for coordinating access
        self._can_read = Condition(self._lock)
        self._can_write = Condition(self._lock)

    @contextmanager
    def read(self):
        """
        Acquire a read lock (shared access).

        Multiple threads can hold read locks simultaneously.
        Blocks if a writer is active or waiting.

        Yields:
            None
        """
        self._acquire_read()
        try:
            yield
        finally:
            self._release_read()

    @contextmanager
    def write(self):
        """
        Acquire a write lock (exclusive access).

        Only one thread can hold a write lock at a time.
        Blocks until all readers and writers are done.

        Yields:
            None
        """
        self._acquire_write()
        try:
            yield
        finally:
            self._release_write()

    def _acquire_read(self):
        """Acquire read lock, waiting if necessary."""
        with self._lock:
            # Wait while any writer is active or waiting
            # (prioritize writers to prevent starvation)
            while self._writers > 0 or self._waiting_writers > 0:
                self._can_read.wait()

            self._readers += 1

    def _release_read(self):
        """Release read lock and notify waiting writers."""
        with self._lock:
            self._readers -= 1

            # If this was the last reader, wake up a waiting writer
            if self._readers == 0:
                self._can_write.notify()

    def _acquire_write(self):
        """Acquire write lock, waiting if necessary."""
        with self._lock:
            self._waiting_writers += 1

            # Wait until no readers or writers are active
            while self._readers > 0 or self._writers > 0:
                self._can_write.wait()

            self._waiting_writers -= 1
            self._writers = 1

    def _release_write(self):
        """Release write lock and notify all waiting threads."""
        with self._lock:
            self._writers = 0

            # Wake up all waiting readers first (fairness)
            self._can_read.notify_all()

            # Then wake up one waiting writer
            self._can_write.notify()

    @property
    def active_readers(self) -> int:
        """Get the number of active readers (for testing/debugging)."""
        with self._lock:
            return self._readers

    @property
    def active_writers(self) -> int:
        """Get the number of active writers (for testing/debugging)."""
        with self._lock:
            return self._writers

    @property
    def waiting_writers(self) -> int:
        """Get the number of waiting writers (for testing/debugging)."""
        with self._lock:
            return self._waiting_writers
