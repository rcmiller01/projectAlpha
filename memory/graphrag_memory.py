"""
GraphRAG Memory System - Enhanced semantic entity linking and reasoning

This module implements a memory graph using NetworkX for semantic entity linking,
enabling autonomous AI agents to build and query conceptual relationships.

Key Features:
- Directed graph with concepts as nodes, semantic relations as edges
- Thread-safe operations with synchronization locks
- JSON persistence with metadata preservation
- Structured querying for conductor/arbiter integration
- UUID-based request tracing for concurrent operations

Author: ProjectAlpha Team
Compatible with: HRM stack, future SLiM agent integration
"""

import threading
import json
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import networkx as nx
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EdgeMetadata:
    """Metadata for graph edges with confidence and provenance tracking"""
    source: str
    confidence: float
    timestamp: str
    relation_type: str
    context: str = ""
    request_id: str = ""

@dataclass
class QueryResult:
    """Structured result for memory queries"""
    request_id: str
    query_node: str
    depth: int
    related_concepts: List[Dict[str, Any]]
    execution_time_ms: float
    timestamp: str

class GraphRAGMemory:
    """
    Thread-safe GraphRAG memory system for semantic entity linking and reasoning.

    This class provides a directed graph-based memory system where:
    - Nodes represent concepts, entities, or semantic units
    - Edges represent relationships with metadata (confidence, source, type)
    - All operations are thread-safe using locks
    - Persistent storage via JSON serialization
    """

    def __init__(self, memory_file: Optional[str] = None):
        """
        Initialize GraphRAG memory system.

        Args:
            memory_file: Path to JSON file for persistence. If None, uses default location.
        """
        self.graph = nx.DiGraph()
        self._lock = threading.Lock()
        self.memory_file = memory_file or "data/graphrag_memory.json"
        self._ensure_data_directory()
        self.load_memory()

        logger.info(f"GraphRAG Memory initialized with {self.graph.number_of_nodes()} nodes, "
                   f"{self.graph.number_of_edges()} edges")

    def _ensure_data_directory(self):
        """Ensure the data directory exists for memory persistence"""
        Path(self.memory_file).parent.mkdir(parents=True, exist_ok=True)

    def add_fact(self, subject: str, relation: str, object_node: str,
                 confidence: float = 1.0, source: str = "unknown",
                 context: str = "") -> str:
        """
        Add a semantic fact to the memory graph.

        Args:
            subject: Source concept/entity
            relation: Type of relationship
            object_node: Target concept/entity
            confidence: Confidence score (0.0 to 1.0)
            source: Source of this information
            context: Additional context for the relationship

        Returns:
            request_id: UUID for tracking this operation
        """
        request_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        with self._lock:
            try:
                # Add nodes if they don't exist
                if not self.graph.has_node(subject):
                    self.graph.add_node(subject,
                                      node_type="concept",
                                      created_at=timestamp,
                                      access_count=0)

                if not self.graph.has_node(object_node):
                    self.graph.add_node(object_node,
                                      node_type="concept",
                                      created_at=timestamp,
                                      access_count=0)

                # Create edge metadata
                edge_metadata = EdgeMetadata(
                    source=source,
                    confidence=confidence,
                    timestamp=timestamp,
                    relation_type=relation,
                    context=context,
                    request_id=request_id
                )

                # Add edge with metadata
                self.graph.add_edge(subject, object_node, **asdict(edge_metadata))

                # Update access counts
                self.graph.nodes[subject]['access_count'] += 1
                self.graph.nodes[object_node]['access_count'] += 1

                logger.info(f"Added fact: {subject} -[{relation}]-> {object_node} "
                           f"(confidence: {confidence}, request_id: {request_id})")

                return request_id

            except Exception as e:
                logger.error(f"Error adding fact: {e}")
                raise

    def query_related(self, node: str, depth: int = 2,
                     min_confidence: float = 0.1) -> QueryResult:
        """
        Query related concepts from a given node within specified depth.

        Args:
            node: Starting concept/entity
            depth: Maximum traversal depth
            min_confidence: Minimum confidence threshold for relationships

        Returns:
            QueryResult: Structured result with related concepts and metadata
        """
        request_id = str(uuid.uuid4())
        start_time = datetime.now()

        with self._lock:
            try:
                if not self.graph.has_node(node):
                    logger.warning(f"Node '{node}' not found in memory graph")
                    return QueryResult(
                        request_id=request_id,
                        query_node=node,
                        depth=depth,
                        related_concepts=[],
                        execution_time_ms=0.0,
                        timestamp=start_time.isoformat()
                    )

                # Update access count for queried node
                self.graph.nodes[node]['access_count'] += 1

                related_concepts = []
                visited = set()

                def _traverse(current_node: str, current_depth: int, path: List[str]):
                    """Recursive traversal with path tracking"""
                    if current_depth > depth or current_node in visited:
                        return

                    visited.add(current_node)

                    # Get outgoing edges (what this node relates to)
                    for neighbor in self.graph.successors(current_node):
                        edge_data = self.graph.edges[current_node, neighbor]
                        confidence = edge_data.get('confidence', 0.0)

                        if confidence >= min_confidence:
                            concept_info = {
                                'concept': neighbor,
                                'relation_type': edge_data.get('relation_type', 'unknown'),
                                'confidence': confidence,
                                'source': edge_data.get('source', 'unknown'),
                                'context': edge_data.get('context', ''),
                                'depth': current_depth + 1,
                                'path': path + [current_node, neighbor],
                                'node_metadata': dict(self.graph.nodes[neighbor])
                            }
                            related_concepts.append(concept_info)

                            # Continue traversal
                            _traverse(neighbor, current_depth + 1, path + [current_node])

                    # Get incoming edges (what relates to this node)
                    for predecessor in self.graph.predecessors(current_node):
                        edge_data = self.graph.edges[predecessor, current_node]
                        confidence = edge_data.get('confidence', 0.0)

                        if confidence >= min_confidence:
                            concept_info = {
                                'concept': predecessor,
                                'relation_type': f"inverse_{edge_data.get('relation_type', 'unknown')}",
                                'confidence': confidence,
                                'source': edge_data.get('source', 'unknown'),
                                'context': edge_data.get('context', ''),
                                'depth': current_depth + 1,
                                'path': path + [current_node, predecessor],
                                'node_metadata': dict(self.graph.nodes[predecessor])
                            }
                            related_concepts.append(concept_info)

                # Start traversal
                _traverse(node, 0, [])

                # Sort by confidence and depth
                related_concepts.sort(key=lambda x: (x['depth'], -x['confidence']))

                execution_time = (datetime.now() - start_time).total_seconds() * 1000

                result = QueryResult(
                    request_id=request_id,
                    query_node=node,
                    depth=depth,
                    related_concepts=related_concepts,
                    execution_time_ms=execution_time,
                    timestamp=start_time.isoformat()
                )

                logger.info(f"Query completed: {node} -> {len(related_concepts)} related concepts "
                           f"(depth: {depth}, time: {execution_time:.2f}ms, request_id: {request_id})")

                return result

            except Exception as e:
                logger.error(f"Error querying related concepts: {e}")
                raise

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory graph statistics"""
        with self._lock:
            try:
                num_nodes = self.graph.number_of_nodes()
                num_edges = self.graph.number_of_edges()

                if num_nodes == 0:
                    return {
                        'total_nodes': 0,
                        'total_edges': 0,
                        'average_degree': 0.0,
                        'most_connected_nodes': [],
                        'memory_file': self.memory_file
                    }

                # Calculate degree statistics
                degree_dict = {}
                for node in self.graph.nodes():
                    degree_dict[node] = len(list(self.graph.neighbors(node))) + len(list(self.graph.predecessors(node)))

                total_degree = sum(degree_dict.values())
                avg_degree = total_degree / num_nodes

                # Get most connected nodes
                most_connected = sorted(
                    degree_dict.items(), key=lambda x: x[1], reverse=True
                )[:10]

                return {
                    'total_nodes': num_nodes,
                    'total_edges': num_edges,
                    'average_degree': avg_degree,
                    'most_connected_nodes': most_connected,
                    'memory_file': self.memory_file
                }
            except Exception as e:
                logger.error(f"Error computing memory stats: {e}")
                return {
                    'total_nodes': 0,
                    'total_edges': 0,
                    'average_degree': 0.0,
                    'most_connected_nodes': [],
                    'memory_file': self.memory_file
                }

    def save_memory(self) -> bool:
        """
        Save memory graph to JSON file.

        Returns:
            bool: Success status
        """
        with self._lock:
            try:
                # Convert graph to JSON-serializable format
                graph_data = nx.node_link_data(self.graph)

                # Create complete data structure with metadata
                data = dict(graph_data)
                data['metadata'] = {
                    'saved_at': datetime.now().isoformat(),
                    'total_nodes': self.graph.number_of_nodes(),
                    'total_edges': self.graph.number_of_edges(),
                    'version': '1.0'
                }

                with open(self.memory_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                logger.info(f"Memory saved to {self.memory_file}")
                return True

            except Exception as e:
                logger.error(f"Error saving memory: {e}")
                return False

    def load_memory(self) -> bool:
        """
        Load memory graph from JSON file.

        Returns:
            bool: Success status
        """
        try:
            if not Path(self.memory_file).exists():
                logger.info(f"Memory file {self.memory_file} not found, starting with empty graph")
                return True

            with open(self.memory_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Remove metadata before loading graph
            metadata = data.pop('metadata', {})

            with self._lock:
                self.graph = nx.node_link_graph(data)

            logger.info(f"Memory loaded from {self.memory_file} "
                       f"({self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges)")
            return True

        except Exception as e:
            logger.error(f"Error loading memory: {e}")
            return False

    def find_shortest_path(self, start: str, end: str) -> Optional[List[str]]:
        """Find shortest path between two concepts"""
        with self._lock:
            try:
                if not (self.graph.has_node(start) and self.graph.has_node(end)):
                    return None
                path = nx.shortest_path(self.graph, start, end)
                return list(path) if isinstance(path, list) else None
            except (nx.NetworkXNoPath, Exception):
                return None

    def get_concept_neighborhood(self, concept: str, radius: int = 1) -> Dict[str, Any]:
        """Get immediate neighborhood of a concept"""
        with self._lock:
            if not self.graph.has_node(concept):
                return {}

            subgraph = nx.ego_graph(self.graph, concept, radius=radius)
            return {
                'center_concept': concept,
                'neighborhood_size': subgraph.number_of_nodes(),
                'nodes': list(subgraph.nodes(data=True)),
                'edges': list(subgraph.edges(data=True))
            }


# Example usage and testing functions
def example_usage():
    """Example usage of GraphRAG Memory system"""

    # Initialize memory system
    memory = GraphRAGMemory("data/example_graphrag_memory.json")

    # Add some example facts
    print("Adding example facts...")

    memory.add_fact("user", "prefers", "chocolate", confidence=0.9, source="conversation",
                   context="User mentioned liking chocolate ice cream")

    memory.add_fact("chocolate", "is_type_of", "dessert", confidence=1.0, source="knowledge_base")

    memory.add_fact("dessert", "follows", "dinner", confidence=0.8, source="cultural_knowledge")

    memory.add_fact("user", "dislikes", "spicy_food", confidence=0.7, source="conversation")

    memory.add_fact("spicy_food", "contains", "capsaicin", confidence=1.0, source="knowledge_base")

    # Query related concepts
    print("\nQuerying related concepts for 'user'...")
    result = memory.query_related("user", depth=3)

    print(f"Request ID: {result.request_id}")
    print(f"Found {len(result.related_concepts)} related concepts:")

    for concept in result.related_concepts[:5]:  # Show first 5
        print(f"  - {concept['concept']} (relation: {concept['relation_type']}, "
              f"confidence: {concept['confidence']:.2f}, depth: {concept['depth']})")

    # Show memory stats
    print(f"\nMemory Statistics:")
    stats = memory.get_memory_stats()
    for key, value in stats.items():
        if key != 'most_connected_nodes':
            print(f"  {key}: {value}")

    # Save memory
    memory.save_memory()
    print("\nMemory saved successfully!")


if __name__ == "__main__":
    example_usage()
