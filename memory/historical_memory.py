import os
import json
import time
import numpy as np
import config


class HistoricalMemory:
    """Optional store for RAW text: full conversations, paragraphs, lectures, anything.
    Only searched when semantic + episodic + procedural can't answer."""

    def __init__(self, embedder=None):
        os.makedirs(config.DATA_FOLDER, exist_ok=True)
        self.file_path = os.path.join(config.DATA_FOLDER, "history.json")

        # FIX: same as SemanticMemory — reuse a shared embedder instead of
        # loading SentenceTransformer from scratch every time this class is created.
        if embedder is None:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        self.embedder = embedder

        self.chunks = []
        self.vectors = []
        self.timestamps = []

        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as file:
                data = json.load(file)
                self.chunks = data.get("chunks", [])
                self.vectors = data.get("vectors", [])
                self.timestamps = data.get("timestamps", [time.time()] * len(self.chunks))

    def add_chunk(self, text, save_now=True):
        vector = self.embedder.encode(text).tolist()
        self.chunks.append(text)
        self.vectors.append(vector)
        self.timestamps.append(time.time())

        # FIX: same batching idea — don't rewrite the whole JSON file per chunk.
        if save_now:
            self.save()

    def add_text(self, full_text, chunk_size=1000):
        """Use this for anything long: a lecture, a full history, a paragraph."""
        for i in range(0, len(full_text), chunk_size):
            # FIX: save_now=False here — we save ONCE after the loop instead of
            # once per chunk. For a long conversation this turns N disk writes
            # into 1.
            self.add_chunk(full_text[i:i + chunk_size], save_now=False)
        self.save()

    def search(self, query, top_k):
        if len(self.chunks) == 0:
            return []

        query_vector = self.embedder.encode(query)
        all_vectors = np.array(self.vectors)
        scores = all_vectors @ query_vector
        scores = scores / (np.linalg.norm(all_vectors, axis=1) * np.linalg.norm(query_vector))

        best_positions = np.argsort(scores)[::-1][:top_k]

        results = []
        for position in best_positions:
            if scores[position] >= config.HISTORICAL_MIN_SCORE:
                results.append(self.chunks[position])
        return results

    def save(self):
        with open(self.file_path, "w") as file:
            json.dump(
                {"chunks": self.chunks, "vectors": self.vectors, "timestamps": self.timestamps},
                file,
            )