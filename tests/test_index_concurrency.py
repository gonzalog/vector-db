"""Tests for concurrent access to vector indexes."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from vector_db.indexes import FlatIndex, HNSWIndex, LSHIndex
from vector_db.models import Chunk


class TestFlatIndexConcurrency:
    """Test concurrent access to FlatIndex."""

    def test_multiple_concurrent_searches(self):
        """Test that multiple threads can search simultaneously."""
        index = FlatIndex()

        # Add some test data
        chunks = [
            Chunk(
                document_id="00000000-0000-0000-0000-000000000001",
                text=f"test {i}",
                embedding=[float(i), 0.0, 0.0],
            )
            for i in range(100)
        ]
        index.add_batch(chunks)

        searches_active = []
        max_concurrent_searches = 0

        def search_operation(search_id: int):
            nonlocal max_concurrent_searches
            searches_active.append(search_id)
            max_concurrent = len(searches_active)
            if max_concurrent > max_concurrent_searches:
                max_concurrent_searches = max_concurrent

            # Perform search (this should hold read lock)
            results = index.search([5.0, 0.0, 0.0], k=5)
            assert len(results) > 0

            time.sleep(0.1)  # Hold lock briefly
            searches_active.remove(search_id)

        # Launch 5 concurrent searches
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(search_operation, i) for i in range(5)]
            for future in futures:
                future.result()

        # All searches should have been able to execute concurrently
        assert max_concurrent_searches == 5

    def test_write_operations_are_exclusive(self):
        """Test that write operations are exclusive."""
        index = FlatIndex()

        writers_active = 0
        max_concurrent_writers = 0
        lock = threading.Lock()

        # Monkey-patch the _lock.write() context manager to track concurrent writers
        original_write = index._lock.write

        def tracked_write():
            nonlocal writers_active, max_concurrent_writers
            context = original_write()

            class TrackedContext:
                def __enter__(self):
                    result = context.__enter__()
                    with lock:
                        nonlocal writers_active, max_concurrent_writers
                        writers_active += 1
                        if writers_active > max_concurrent_writers:
                            max_concurrent_writers = writers_active
                    time.sleep(0.05)  # Simulate work while holding lock
                    return result

                def __exit__(self, *args):
                    result = context.__exit__(*args)
                    with lock:
                        nonlocal writers_active
                        writers_active -= 1
                    return result

            return TrackedContext()

        index._lock.write = tracked_write

        def add_operation():
            chunk = Chunk(
                document_id="00000000-0000-0000-0000-000000000001",
                text="test",
                embedding=[1.0, 0.0, 0.0],
            )
            index.add(chunk)

        # Launch 3 concurrent add operations
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(add_operation) for _ in range(3)]
            for future in futures:
                future.result()

        # Only one writer should have been active at a time
        assert max_concurrent_writers == 1

    def test_writers_block_readers(self):
        """Test that active writer blocks readers."""
        index = FlatIndex()

        # Add initial data
        chunks = [
            Chunk(
                document_id="00000000-0000-0000-0000-000000000001",
                text=f"test {i}",
                embedding=[float(i), 0.0, 0.0],
            )
            for i in range(10)
        ]
        index.add_batch(chunks)

        write_started = threading.Event()
        read_attempted = threading.Event()
        read_succeeded = threading.Event()

        # Monkey-patch to track when write lock is held
        original_write = index._lock.write

        def tracked_write():
            context = original_write()

            class TrackedContext:
                def __enter__(self):
                    result = context.__enter__()
                    write_started.set()
                    time.sleep(0.2)  # Hold lock for a bit
                    # If read was attempted, it should still be blocked
                    if read_attempted.is_set():
                        assert not read_succeeded.is_set(), "Read succeeded while write lock held!"
                    return result

                def __exit__(self, *args):
                    return context.__exit__(*args)

            return TrackedContext()

        index._lock.write = tracked_write

        def writer():
            chunk = Chunk(
                document_id="00000000-0000-0000-0000-000000000001",
                text="new chunk",
                embedding=[100.0, 0.0, 0.0],
            )
            index.add(chunk)

        def reader():
            write_started.wait()  # Wait for writer to acquire lock
            read_attempted.set()
            index.search([0.0, 0.0, 0.0], k=5)
            read_succeeded.set()

        # Start writer and reader
        writer_thread = threading.Thread(target=writer)
        reader_thread = threading.Thread(target=reader)

        writer_thread.start()
        reader_thread.start()

        writer_thread.join()
        reader_thread.join()

        # Read should have succeeded eventually (after writer released)
        assert read_succeeded.is_set()

    def test_readers_block_writers(self):
        """Test that active readers block writers."""
        index = FlatIndex()

        # Add initial data
        chunks = [
            Chunk(
                document_id="00000000-0000-0000-0000-000000000001",
                text=f"test {i}",
                embedding=[float(i), 0.0, 0.0],
            )
            for i in range(10)
        ]
        index.add_batch(chunks)

        read_started = threading.Event()
        write_attempted = threading.Event()
        write_succeeded = threading.Event()

        # Monkey-patch to track when read lock is held
        original_read = index._lock.read

        def tracked_read():
            context = original_read()

            class TrackedContext:
                def __enter__(self):
                    result = context.__enter__()
                    read_started.set()
                    time.sleep(0.2)  # Hold lock for a bit
                    # If write was attempted, it should still be blocked
                    if write_attempted.is_set():
                        assert not write_succeeded.is_set(), "Write succeeded while read lock held!"
                    return result

                def __exit__(self, *args):
                    return context.__exit__(*args)

            return TrackedContext()

        index._lock.read = tracked_read

        def reader():
            index.search([0.0, 0.0, 0.0], k=5)

        def writer():
            read_started.wait()  # Wait for reader to acquire lock
            write_attempted.set()
            chunk = Chunk(
                document_id="00000000-0000-0000-0000-000000000001",
                text="new chunk",
                embedding=[100.0, 0.0, 0.0],
            )
            index.add(chunk)
            write_succeeded.set()

        # Start reader and writer
        reader_thread = threading.Thread(target=reader)
        writer_thread = threading.Thread(target=writer)

        reader_thread.start()
        writer_thread.start()

        reader_thread.join()
        writer_thread.join()

        # Write should have succeeded eventually (after reader released)
        assert write_succeeded.is_set()


class TestLSHIndexConcurrency:
    """Test concurrent access to LSHIndex."""

    def test_multiple_concurrent_searches(self):
        """Test that multiple threads can search LSH index simultaneously."""
        index = LSHIndex(n_hash_tables=3, n_hash_bits=4)

        # Add some test data
        chunks = [
            Chunk(
                document_id="00000000-0000-0000-0000-000000000001",
                text=f"test {i}",
                embedding=[float(i), 0.0, 0.0],
            )
            for i in range(50)
        ]
        index.add_batch(chunks)

        searches_active = []
        max_concurrent_searches = 0

        def search_operation(search_id: int):
            nonlocal max_concurrent_searches
            searches_active.append(search_id)
            max_concurrent = len(searches_active)
            if max_concurrent > max_concurrent_searches:
                max_concurrent_searches = max_concurrent

            # Perform search
            index.search([5.0, 0.0, 0.0], k=5)

            time.sleep(0.1)  # Hold lock briefly
            searches_active.remove(search_id)

        # Launch 5 concurrent searches
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(search_operation, i) for i in range(5)]
            for future in futures:
                future.result()

        # All searches should have been able to execute concurrently
        assert max_concurrent_searches == 5

    def test_write_operations_are_exclusive(self):
        """Test that LSH write operations are exclusive."""
        index = LSHIndex(n_hash_tables=3, n_hash_bits=4)

        writers_active = 0
        max_concurrent_writers = 0
        lock = threading.Lock()

        # Monkey-patch the _lock.write() context manager
        original_write = index._lock.write

        def tracked_write():
            nonlocal writers_active, max_concurrent_writers
            context = original_write()

            class TrackedContext:
                def __enter__(self):
                    result = context.__enter__()
                    with lock:
                        nonlocal writers_active, max_concurrent_writers
                        writers_active += 1
                        if writers_active > max_concurrent_writers:
                            max_concurrent_writers = writers_active
                    time.sleep(0.05)
                    return result

                def __exit__(self, *args):
                    result = context.__exit__(*args)
                    with lock:
                        nonlocal writers_active
                        writers_active -= 1
                    return result

            return TrackedContext()

        index._lock.write = tracked_write

        def add_operation():
            chunk = Chunk(
                document_id="00000000-0000-0000-0000-000000000001",
                text="test",
                embedding=[1.0, 0.0, 0.0],
            )
            index.add(chunk)

        # Launch 3 concurrent add operations
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(add_operation) for _ in range(3)]
            for future in futures:
                future.result()

        # Only one writer should have been active at a time
        assert max_concurrent_writers == 1


class TestHNSWIndexConcurrency:
    """Test concurrent access to HNSWIndex."""

    def test_multiple_concurrent_searches(self):
        """Test that multiple threads can search HNSW index simultaneously."""
        index = HNSWIndex(M=8, ef_construction=50, ef_search=20)

        # Add some test data
        chunks = [
            Chunk(
                document_id="00000000-0000-0000-0000-000000000001",
                text=f"test {i}",
                embedding=[float(i), 0.0, 0.0],
            )
            for i in range(50)
        ]
        index.add_batch(chunks)

        searches_active = []
        max_concurrent_searches = 0

        def search_operation(search_id: int):
            nonlocal max_concurrent_searches
            searches_active.append(search_id)
            max_concurrent = len(searches_active)
            if max_concurrent > max_concurrent_searches:
                max_concurrent_searches = max_concurrent

            # Perform search
            results = index.search([5.0, 0.0, 0.0], k=5)
            assert len(results) > 0

            time.sleep(0.1)  # Hold lock briefly
            searches_active.remove(search_id)

        # Launch 5 concurrent searches
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(search_operation, i) for i in range(5)]
            for future in futures:
                future.result()

        # All searches should have been able to execute concurrently
        assert max_concurrent_searches == 5

    def test_write_operations_are_exclusive(self):
        """Test that HNSW write operations are exclusive."""
        index = HNSWIndex(M=8, ef_construction=50, ef_search=20)

        writers_active = 0
        max_concurrent_writers = 0
        lock = threading.Lock()

        # Monkey-patch the _lock.write() context manager
        original_write = index._lock.write

        def tracked_write():
            nonlocal writers_active, max_concurrent_writers
            context = original_write()

            class TrackedContext:
                def __enter__(self):
                    result = context.__enter__()
                    with lock:
                        nonlocal writers_active, max_concurrent_writers
                        writers_active += 1
                        if writers_active > max_concurrent_writers:
                            max_concurrent_writers = writers_active
                    time.sleep(0.05)
                    return result

                def __exit__(self, *args):
                    result = context.__exit__(*args)
                    with lock:
                        nonlocal writers_active
                        writers_active -= 1
                    return result

            return TrackedContext()

        index._lock.write = tracked_write

        def add_operation(i: int):
            chunk = Chunk(
                document_id="00000000-0000-0000-0000-000000000001",
                text=f"test {i}",
                embedding=[float(i), 0.0, 0.0],
            )
            index.add(chunk)

        # Launch 3 concurrent add operations
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(add_operation, i) for i in range(3)]
            for future in futures:
                future.result()

        # Only one writer should have been active at a time
        assert max_concurrent_writers == 1

    def test_high_concurrency_mixed_operations(self):
        """Test HNSW with many concurrent readers and occasional writers."""
        index = HNSWIndex(M=8, ef_construction=50, ef_search=20)

        # Pre-populate with some data
        initial_chunks = [
            Chunk(
                document_id="00000000-0000-0000-0000-000000000001",
                text=f"initial {i}",
                embedding=[float(i), 0.0, 0.0],
            )
            for i in range(20)
        ]
        index.add_batch(initial_chunks)

        search_count = 0
        write_count = 0
        lock = threading.Lock()

        def reader():
            nonlocal search_count
            for _ in range(10):
                results = index.search([5.0, 0.0, 0.0], k=3)
                assert len(results) > 0
                with lock:
                    search_count += 1
                time.sleep(0.001)

        def writer(i: int):
            nonlocal write_count
            for j in range(3):
                chunk = Chunk(
                    document_id="00000000-0000-0000-0000-000000000001",
                    text=f"writer {i} chunk {j}",
                    embedding=[float(i * 10 + j), 0.0, 0.0],
                )
                index.add(chunk)
                with lock:
                    write_count += 1
                time.sleep(0.001)

        # Launch many readers and fewer writers
        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = []
            # 10 readers
            for _ in range(10):
                futures.append(executor.submit(reader))
            # 5 writers
            for i in range(5):
                futures.append(executor.submit(writer, i))

            for future in futures:
                future.result()

        # Verify all operations completed
        assert search_count == 100  # 10 readers × 10 searches
        assert write_count == 15  # 5 writers × 3 writes
