import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
import json


class SemanticMemory:
    def __init__(self, persist_directory: str = "./memory"):
        """Initialize the semantic memory system"""
        # Initialize embedding model
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize Chroma client with persistence
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Create collections for different memory types
        self.immediate = self.client.get_or_create_collection(
            name="immediate_memory",
            metadata={"type": "immediate"}
        )

        self.working = self.client.get_or_create_collection(
            name="working_memory",
            metadata={"type": "working"}
        )

        self.long_term = self.client.get_or_create_collection(
            name="long_term_memory",
            metadata={"type": "long_term"}
        )

    def store(self, content: str, context: Optional[Dict] = None) -> str:
        """Store new memory with optional context"""
        # Generate timestamp
        timestamp = datetime.now().isoformat()

        # Create memory metadata
        metadata = {
            "timestamp": timestamp,
            "type": "memory"
        }

        # Add optional context
        if context:
            metadata.update(context)

        # Generate unique ID
        memory_id = f"mem_{timestamp}"

        # Store in immediate memory
        self.immediate.add(
            documents=[content],
            metadatas=[metadata],
            ids=[memory_id]
        )

        # Trigger pattern consolidation
        self._consolidate_patterns()

        return memory_id

    def recall(self,
               query: str,
               n_results: int = 5,
               context_filter: Optional[Dict] = None) -> List[Dict]:
        """Recall memories similar to query"""
        # Search all memory layers
        results = []

        # Query immediate memory
        immediate_results = self.immediate.query(
            query_texts=[query],
            n_results=n_results,
            where=context_filter if context_filter and len(context_filter) > 0 else None
        )
        results.extend(self._format_results(immediate_results, "immediate"))

        # Query working memory
        working_results = self.working.query(
            query_texts=[query],
            n_results=n_results,
            where=context_filter if context_filter and len(context_filter) > 0 else None
        )
        results.extend(self._format_results(working_results, "working"))

        # Query long-term memory
        long_term_results = self.long_term.query(
            query_texts=[query],
            n_results=n_results,
            where=context_filter if context_filter and len(context_filter) > 0 else None
        )
        results.extend(self._format_results(long_term_results, "long_term"))

        # Sort by relevance
        results.sort(key=lambda x: x['similarity'], reverse=True)

        return results[:n_results]

    def _consolidate_patterns(self):
        """Consolidate patterns across memory layers"""
        # Get all immediate memories
        immediate_memories = self.immediate.get()

        if not immediate_memories['documents']:
            return

        # Find clusters in immediate memory
        embeddings = self.encoder.encode(immediate_memories['documents'])
        clusters = self._cluster_embeddings(embeddings)

        # For each cluster
        for cluster_idx, cluster in enumerate(clusters):
            # Extract pattern
            pattern = self._extract_pattern(
                [immediate_memories['documents'][i] for i in cluster],
                [immediate_memories['metadatas'][i] for i in cluster]
            )

            # Store pattern in working memory
            pattern_id = f"pattern_{datetime.now().isoformat()}_{cluster_idx}"
            self.working.add(
                documents=[pattern['content']],
                metadatas=[pattern['metadata']],
                ids=[pattern_id]
            )

        # Clean up immediate memory
        if len(immediate_memories['ids']) > 100:  # Keep last 100 memories
            self.immediate.delete(ids=immediate_memories['ids'][:-100])

    def _cluster_embeddings(self, embeddings: np.ndarray,
                            threshold: float = 0.8) -> List[List[int]]:
        """Cluster embeddings using cosine similarity"""
        # Calculate similarity matrix
        similarity_matrix = np.inner(embeddings, embeddings)

        # Find clusters
        clusters = []
        used_indices = set()

        for i in range(len(embeddings)):
            if i in used_indices:
                continue

            # Find similar embeddings
            cluster = [i]
            used_indices.add(i)

            for j in range(i + 1, len(embeddings)):
                if j in used_indices:
                    continue

                if similarity_matrix[i][j] > threshold:
                    cluster.append(j)
                    used_indices.add(j)

            clusters.append(cluster)

        return clusters

    def _extract_pattern(self,
                         documents: List[str],
                         metadatas: List[Dict]) -> Dict:
        """Extract pattern from cluster of similar memories"""
        # For now, use most recent memory as pattern representative
        latest_idx = max(range(len(metadatas)),
                         key=lambda i: metadatas[i]['timestamp'])

        pattern_metadata = {
            "type": "pattern",
            "timestamp": datetime.now().isoformat(),
            "source_count": len(documents),
            "source_timestamps": json.dumps([m['timestamp'] for m in metadatas])  # Serialize list to string
        }

        return {
            "content": documents[latest_idx],
            "metadata": pattern_metadata
        }

    def _format_results(self,
                        results: Dict,
                        memory_type: str) -> List[Dict]:
        """Format query results"""
        formatted = []

        if not results['documents']:
            return formatted

        documents = results['documents'][0]  # First query results
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            formatted.append({
                'content': doc,
                'metadata': meta,
                'similarity': 1 - dist,  # Convert distance to similarity
                'memory_type': memory_type
            })

        return formatted

    def get_stats(self) -> Dict:
        """Get memory system statistics"""
        return {
            "immediate_count": self.immediate.count(),
            "working_count": self.working.count(),
            "long_term_count": self.long_term.count()
        }


# Example usage
if __name__ == "__main__":
    # Initialize memory system
    memory = SemanticMemory("./semantic_memory")

    # Store some memories
    memory.store(
        "The sky was particularly blue today",
        context={"category": "observation"}
    )

    memory.store(
        "I learned about quantum computing",
        context={"category": "learning"}
    )

    # Recall memories
    results = memory.recall(
        "Do I know about quantum computing?",
        n_results=3
    )

    for result in results:
        print(f"Memory: {result['content']}")
        print(f"Type: {result['memory_type']}")
        print(f"Similarity: {result['similarity']:.2f}")
        print("---")

    # Recall memories
    results = memory.recall(
        "What is the color of the sky?",
        n_results=3
    )

    for result in results:
        print(f"Memory: {result['content']}")
        print(f"Type: {result['memory_type']}")
        print(f"Similarity: {result['similarity']:.2f}")
        print("---")