"""
Core IR data structures for the RL IR Semantic Kernel.
This module defines the fundamental building blocks of the computation graph:
Nodes, Edges, and the Graph container itself.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, NewType, Set, Optional
from core.schema import Schema

# Type alias for Node identifiers
NodeId = NewType("NodeId", str)

# Node type constants
NODE_TYPE_ACTOR = "Actor"
NODE_TYPE_TRANSFORM = "Transform"
NODE_TYPE_SOURCE = "Source"
NODE_TYPE_SINK = "Sink"
NODE_TYPE_CONTROL = "Control"
NODE_TYPE_REPLAY_QUERY = "ReplayQuery"
NODE_TYPE_SCHEDULE = "Schedule"
NODE_TYPE_TRANSFER = "Transfer"
NODE_TYPE_METRICS_SINK = "MetricsSink"
NODE_TYPE_TARGET_SYNC = "TargetSync"
NODE_TYPE_EXPLORATION = "Exploration"


class EdgeType(Enum):
    """Types of edges representing different flow semantics."""

    DATA = "data"
    CONTROL = "control"
    EFFECT = "effect"


@dataclass(frozen=True)
class Node:
    """
    An atomic executable unit in the IR.

    Attributes:
        node_id: Unique identifier for the node.
        node_type: Functional type of the node (e.g., Actor, Transform).
        schema_in: Data schema for input ports.
        schema_out: Data schema for output ports.
        params: Opaque dictionary of configuration parameters.
        tags: Metadata tags for grouping and selection.
    """

    node_id: NodeId
    node_type: str
    schema_in: Schema = field(default_factory=lambda: Schema(fields=[]))
    schema_out: Schema = field(default_factory=lambda: Schema(fields=[]))
    params: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": str(self.node_id),
            "node_type": self.node_type,
            "schema_in": self.schema_in.to_dict(),
            "schema_out": self.schema_out.to_dict(),
            "params": self.params,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Node":
        return cls(
            node_id=NodeId(data["node_id"]),
            node_type=data["node_type"],
            schema_in=Schema.from_dict(data["schema_in"]),
            schema_out=Schema.from_dict(data["schema_out"]),
            params=data.get("params", {}),
            tags=data.get("tags", []),
        )


@dataclass(frozen=True)
class Edge:
    """
    A connection between two nodes in the graph.

    Attributes:
        src: Source NodeId.
        dst: Destination NodeId.
        edge_type: The semantic type of the connection.
        dst_port: Optional named port on the destination node for this input.
    """

    src: NodeId
    dst: NodeId
    edge_type: EdgeType = EdgeType.DATA
    dst_port: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "src": str(self.src),
            "dst": str(self.dst),
            "edge_type": self.edge_type.value,
            "dst_port": self.dst_port,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Edge":
        return cls(
            src=NodeId(data["src"]),
            dst=NodeId(data["dst"]),
            edge_type=EdgeType(data["edge_type"]),
            dst_port=data.get("dst_port"),
        )


class Graph:
    """
    A static IR container representing a computation graph.

    The Graph maintains a collection of nodes and directed edges between them.
    It provides methods for graph construction and traversal.
    """

    def __init__(self) -> None:
        """Initialize an empty Graph."""
        self.nodes: Dict[NodeId, Node] = {}
        self.edges: List[Edge] = []
        self._adjacency: Dict[NodeId, Set[NodeId]] = {}

    def add_node(
        self,
        node_id: str,
        node_type: str,
        schema_in: Schema = None,
        schema_out: Schema = None,
        params: Dict[str, Any] = None,
        tags: List[str] = None,
    ) -> Node:
        """
        Add a new node to the graph.

        Args:
            node_id: Unique identifier string.
            node_type: Type of node (e.g., 'Actor', 'Transform').
            schema_in: Optional input schema.
            schema_out: Optional output schema.
            params: Optional parameters.
            tags: Optional tags.

        Returns:
            The created Node instance.
        """
        nid = NodeId(node_id)
        assert nid not in self.nodes, f"Node with id {node_id} already exists"
        # TODO: this should probably fail fast instead
        # Defensive check: ensure schemas are Schema objects
        from core.schema import TensorSpec, Field

        if schema_in is not None and isinstance(schema_in, TensorSpec):
            schema_in = Schema(fields=[Field("default", schema_in)])
        if schema_out is not None and isinstance(schema_out, TensorSpec):
            schema_out = Schema(fields=[Field("default", schema_out)])

        node = Node(
            node_id=nid,
            node_type=node_type,
            schema_in=schema_in or Schema(fields=[]),
            schema_out=schema_out or Schema(fields=[]),
            params=params or {},
            tags=tags or [],
        )
        self.nodes[nid] = node
        if nid not in self._adjacency:
            self._adjacency[nid] = set()
        return node

    def add_edge(
        self,
        src: str,
        dst: str,
        edge_type: EdgeType = EdgeType.DATA,
        dst_port: Optional[str] = None,
    ) -> Edge:
        """
        Add a directed edge between two existing nodes.

        Args:
            src: Source node ID.
            dst: Destination node ID.
            edge_type: Type of the edge.
            dst_port: Optional named port on the destination node.

        Returns:
            The created Edge instance.

        Raises:
            AssertionError: If either source or destination node does not exist.
        """
        snid = NodeId(src)
        dnid = NodeId(dst)
        assert snid in self.nodes, f"Source node {src} not found"
        assert dnid in self.nodes, f"Destination node {dst} not found"

        edge = Edge(src=snid, dst=dnid, edge_type=edge_type, dst_port=dst_port)
        self.edges.append(edge)
        self._adjacency[snid].add(dnid)
        return edge

    @property
    def adjacency_list(self) -> Dict[NodeId, Set[NodeId]]:
        """
        Get the graph adjacency list.

        Returns:
            A dictionary mapping each NodeId to a set of its successor NodeIds.
        """
        return self._adjacency

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the graph to a dictionary."""
        return {
            "nodes": {str(nid): node.to_dict() for nid, node in self.nodes.items()},
            "edges": [edge.to_dict() for edge in self.edges],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Graph":
        """Reconstruct a graph from a dictionary."""
        graph = cls()
        for nid, node_data in data["nodes"].items():
            node = Node.from_dict(node_data)
            graph.nodes[node.node_id] = node
            if node.node_id not in graph._adjacency:
                graph._adjacency[node.node_id] = set()

        for edge_data in data["edges"]:
            edge = Edge.from_dict(edge_data)
            graph.edges.append(edge)
            graph._adjacency[edge.src].add(edge.dst)

        return graph
