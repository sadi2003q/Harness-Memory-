import os
import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer
import config


class SemanticMemory:
    """FACTS and knowledge, searched by meaning. Keeps up to N recent instances
    per topic instead of blocking or overwriting similar facts."""

    def __init__(self):
        os.makedirs(config.DATA_FOLDER, exist_ok=True)
        self.file_path = os.path.join(config.DATA_FOLDER, "facts.json")
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)

        self.facts = []
        self.vectors = []
        self.timestamps = []

        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as file:
                data = json.load(file)
                self.facts = data["facts"]
                self.vectors = data["vectors"]
                self.timestamps = data.get("timestamps", [time.time()] * len(self.facts))

    def add_fact(self, text, max_instances=None):
        max_instances = max_instances or config.SEMANTIC_MAX_INSTANCES

        vector = self.embedder.encode(text).tolist()

        # find facts that are basically about the same topic (very high similarity)
        if len(self.facts) > 0:
            all_vectors = np.array(self.vectors)
            q = np.array(vector)
            scores = all_vectors @ q
            scores = scores / (np.linalg.norm(all_vectors, axis=1) * np.linalg.norm(q))
            family = [i for i, s in enumerate(scores) if s >= config.SEMANTIC_DUPLICATE_SCORE]

            # if this topic already has too many instances, drop the OLDEST one
            if len(family) >= max_instances:
                oldest_index = min(family, key=lambda i: self.timestamps[i])
                self._remove(oldest_index)

        self.facts.append(text)
        self.vectors.append(vector)
        self.timestamps.append(time.time())
        self.save()

    def _remove(self, index):
        del self.facts[index]
        del self.vectors[index]
        del self.timestamps[index]

    def search(self, query, top_k):
        if len(self.facts) == 0:
            return []

        query_vector = self.embedder.encode(query)
        all_vectors = np.array(self.vectors)

        # cosine similarity = how close in meaning
        scores = all_vectors @ query_vector
        scores = scores / (np.linalg.norm(all_vectors, axis=1) * np.linalg.norm(query_vector))

        # rank by score, break near-ties by newest first
        ranked = sorted(
            range(len(self.facts)),
            key=lambda i: (round(scores[i], 2), self.timestamps[i]),
            reverse=True,
        )

        results = []
        for i in ranked[:top_k]:
            if scores[i] >= config.SEMANTIC_MIN_SCORE:
                results.append(self.facts[i])
        return results

    def save(self):
        with open(self.file_path, "w") as file:
            json.dump(
                {"facts": self.facts, "vectors": self.vectors, "timestamps": self.timestamps},
                file,
            )