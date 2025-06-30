"""Test cases for FaissStore functionality."""

import os
import numpy as np
import tempfile
import shutil
from main.vector_store.faiss_indexer import FaissStore

# Create a FAISS store,
# Add embeddings and documents,
# Search,
# Save/load,
# Validate everything works as expected.



def test_faiss_store():
    temp_dir = tempfile.mkdtemp()
    index_file = os.path.join(temp_dir, "test.index")
    metadata_file = os.path.join(temp_dir, "test_metadata.npy")

    try:
        dim = 384
        store = FaissStore(dim)

        embeddings = np.random.rand(3, dim).astype("float32")
        documents = ["doc1", "doc2", "doc3"]

        store.add(embeddings, documents)
        assert store.index.ntotal == 3
        assert len(store.metadata) == 3

        # Search for first embedding, expect at least 1 result
        results = store.search(embeddings[0], k=2)
        assert len(results) > 0
        assert any(doc == "doc1" for doc, _ in results)

        # Save and load index
        store.save(index_file, metadata_file)
        assert os.path.exists(index_file)
        assert os.path.exists(metadata_file)

        new_store = FaissStore(dim)
        new_store.load(index_file, metadata_file)

        # Validate loaded index
        assert new_store.index.ntotal == 3
        assert len(new_store.metadata) == 3

        loaded_results = new_store.search(embeddings[1], k=2)
        assert len(loaded_results) > 0
        assert set(doc for doc, _ in loaded_results).issubset(set(documents))

    finally:
        shutil.rmtree(temp_dir)
