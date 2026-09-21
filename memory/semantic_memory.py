import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import config


class SemanticMemory:
    """FACTS and knowledge, searched by meaning (a simple vector DB)."""

    def __init__(self):
        os.makedirs(config.DATA_FOLDER, exist_ok=True)
        self.file_path = os.path.join(config.DATA_FOLDER, "facts.json")
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)

        self.facts = []      # the texts
        self.vectors = []    # one number-list per text

        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as file:
                data = json.load(file)
                self.facts = data["facts"]
                self.vectors = data["vectors"]

    def add_fact(self, text):
        vector = self.embedder.encode(text).tolist()
        self.facts.append(text)
        self.vectors.append(vector)
        self.save()

    def search(self, query, top_k):
        if len(self.facts) == 0:
            return []

        query_vector = self.embedder.encode(query)
        all_vectors = np.array(self.vectors)

        # cosine similarity = how close in meaning
        scores = all_vectors @ query_vector
        scores = scores / (np.linalg.norm(all_vectors, axis=1) * np.linalg.norm(query_vector))

        best_positions = np.argsort(scores)[::-1][:top_k]

        results = []
        for position in best_positions:
            if scores[position] >= config.SEMANTIC_MIN_SCORE:
                results.append(self.facts[position])
        return results

    def save(self):
        with open(self.file_path, "w") as file:
            json.dump({"facts": self.facts, "vectors": self.vectors}, file)