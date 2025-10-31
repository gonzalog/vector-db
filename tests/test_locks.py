"""Tests for thread synchronization primitives."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from vector_db.core.locks import ReadWriteLock


class TestReadWriteLock:
    """Tests for ReadWriteLock implementation."""

    def test_multiple_concurrent_readers(self):
        """Test that multiple threads can hold read locks simultaneously."""
        lock = ReadWriteLock()
        readers_active = []
        max_concurrent_readers = 0

        def reader(reader_id: int):
            nonlocal max_concurrent_readers
            with lock.read():
                readers_active.append(reader_id)
                max_concurrent = len(readers_active)
                if max_concurrent > max_concurrent_readers:
                    max_concurrent_readers = max_concurrent
                time.sleep(0.1)  # Hold lock briefly
                readers_active.remove(reader_id)

        # Launch 5 concurrent readers
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(reader, i) for i in range(5)]
            for future in futures:
                future.result()

        # All readers should have been able to execute concurrently
        assert max_concurrent_readers == 5

    def test_write_lock_exclusive(self):
        """Test that write lock provides exclusive access."""
        lock = ReadWriteLock()
        writers_active = 0
        max_concurrent_writers = 0

        def writer():
            nonlocal writers_active, max_concurrent_writers
            with lock.write():
                writers_active += 1
                max_concurrent = writers_active
                if max_concurrent > max_concurrent_writers:
                    max_concurrent_writers = max_concurrent
                time.sleep(0.05)  # Hold lock briefly
                writers_active -= 1

        # Launch 3 writers
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(writer) for _ in range(3)]
            for future in futures:
                future.result()

        # Only one writer should have been active at a time
        assert max_concurrent_writers == 1

    def test_writers_block_readers(self):
        """Test that active writer blocks readers."""
        lock = ReadWriteLock()
        write_started = threading.Event()
        write_finished = threading.Event()
        read_started = threading.Event()

        def writer():
            with lock.write():
                write_started.set()
                time.sleep(0.2)  # Hold write lock
                assert not read_started.is_set(), "Reader started during write!"
            write_finished.set()

        def reader():
            write_started.wait()  # Wait for writer to start
            with lock.read():
                read_started.set()

        # Start writer and reader
        writer_thread = threading.Thread(target=writer)
        reader_thread = threading.Thread(target=reader)

        writer_thread.start()
        reader_thread.start()

        writer_thread.join()
        reader_thread.join()

        assert write_finished.is_set()
        assert read_started.is_set()

    def test_readers_block_writers(self):
        """Test that active readers block writers."""
        lock = ReadWriteLock()
        read_started = threading.Event()
        read_finished = threading.Event()
        write_started = threading.Event()

        def reader():
            with lock.read():
                read_started.set()
                time.sleep(0.2)  # Hold read lock
                assert not write_started.is_set(), "Writer started during read!"
            read_finished.set()

        def writer():
            read_started.wait()  # Wait for reader to start
            with lock.write():
                write_started.set()

        # Start reader and writer
        reader_thread = threading.Thread(target=reader)
        writer_thread = threading.Thread(target=writer)

        reader_thread.start()
        writer_thread.start()

        reader_thread.join()
        writer_thread.join()

        assert read_finished.is_set()
        assert write_started.is_set()

    def test_writer_priority_prevents_starvation(self):
        """Test that waiting writers are prioritized over new readers."""
        lock = ReadWriteLock()
        writer_got_lock = threading.Event()
        reader_started = threading.Event()

        def long_reader():
            """First reader holds lock."""
            with lock.read():
                reader_started.set()
                time.sleep(0.2)

        def writer():
            """Writer waits for first reader."""
            reader_started.wait()
            time.sleep(0.05)  # Let writer start waiting
            with lock.write():
                writer_got_lock.set()

        def late_reader():
            """Reader tries to acquire after writer is waiting."""
            reader_started.wait()
            time.sleep(0.1)  # Ensure writer is waiting
            with lock.read():
                # This should only succeed after writer completes
                assert writer_got_lock.is_set(), "Late reader got lock before writer!"

        # Start threads
        threads = [
            threading.Thread(target=long_reader),
            threading.Thread(target=writer),
            threading.Thread(target=late_reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def test_properties(self):
        """Test that lock properties report correct state."""
        lock = ReadWriteLock()

        # Initially no active readers or writers
        assert lock.active_readers == 0
        assert lock.active_writers == 0
        assert lock.waiting_writers == 0

        # Acquire read lock
        with lock.read():
            assert lock.active_readers == 1
            assert lock.active_writers == 0

        # After release
        assert lock.active_readers == 0

        # Acquire write lock
        with lock.write():
            assert lock.active_readers == 0
            assert lock.active_writers == 1

        # After release
        assert lock.active_writers == 0

    def test_nested_read_locks_same_thread(self):
        """Test that same thread can acquire multiple read locks."""
        lock = ReadWriteLock()

        with lock.read():
            assert lock.active_readers == 1
            with lock.read():
                # Same thread, nested read lock
                assert lock.active_readers == 2

        assert lock.active_readers == 0

    def test_exception_releases_lock(self):
        """Test that exceptions properly release locks."""
        lock = ReadWriteLock()

        # Read lock
        try:
            with lock.read():
                raise ValueError("Test exception")
        except ValueError:
            pass

        assert lock.active_readers == 0

        # Write lock
        try:
            with lock.write():
                raise ValueError("Test exception")
        except ValueError:
            pass

        assert lock.active_writers == 0

    def test_high_concurrency(self):
        """Test lock behavior under high concurrency."""
        lock = ReadWriteLock()
        shared_counter = 0
        read_count = 0
        write_count = 0

        def reader():
            nonlocal read_count
            for _ in range(10):
                with lock.read():
                    # Multiple readers can read concurrently
                    _ = shared_counter
                    read_count += 1
                    time.sleep(0.001)

        def writer():
            nonlocal shared_counter, write_count
            for _ in range(5):
                with lock.write():
                    # Exclusive write access
                    shared_counter += 1
                    write_count += 1
                    time.sleep(0.001)

        # Launch many readers and fewer writers
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            # 10 readers
            for _ in range(10):
                futures.append(executor.submit(reader))
            # 5 writers
            for _ in range(5):
                futures.append(executor.submit(writer))

            for future in futures:
                future.result()

        # Verify all operations completed
        assert read_count == 100  # 10 readers × 10 reads
        assert write_count == 25  # 5 writers × 5 writes
        assert shared_counter == 25  # All writes succeeded
