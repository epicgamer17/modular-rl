"""
Node definitions and registry for the RL IR.
Defines static node types (NodeDef) and their instances (NodeInstance).
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Type
from core.schema import Schema
from core.graph import NodeId, NODE_TYPE_ACTOR, NODE_TYPE_TRANSFORM

@dataclass(frozen=True)
class NodeDef:
    """
    Static definition of a node's behavior and interface.
    Equivalent to an 'Operator' in the design doc.
    """
    node_type: str
    input_schema: Schema
    output_schema: Schema
    description: str = ""

@dataclass(frozen=True)
class NodeInstance:
    """
    A specific instance of a NodeDef in a graph.
    Contains runtime parameters and unique identification.
    """
    node_id: NodeId
    node_def: NodeDef
    params: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    @property
    def node_type(self) -> str:
        return self.node_def.node_type

    @property
    def input_schema(self) -> Schema:
        return self.node_def.input_schema

    @property
    def output_schema(self) -> Schema:
        return self.node_def.output_schema

# --- Concrete Node Definitions ---

def create_policy_actor_def(obs_schema: Schema, action_schema: Schema) -> NodeDef:
    """Creates a definition for a Policy Actor node."""
    return NodeDef(
        node_type=NODE_TYPE_ACTOR,
        input_schema=obs_schema,
        output_schema=action_schema,
        description="Executes a policy to produce actions from observations."
    )

def create_gae_def(input_schema: Schema, output_schema: Schema) -> NodeDef:
    """Creates a definition for a GAE (Generalized Advantage Estimation) node."""
    return NodeDef(
        node_type=NODE_TYPE_TRANSFORM,
        input_schema=input_schema,
        output_schema=output_schema,
        description="Computes generalized advantage estimates."
    )

def create_replay_query_def(output_schema: Schema) -> NodeDef:
    """Creates a definition for a ReplayQuery node."""
    from core.graph import NODE_TYPE_REPLAY_QUERY
    from core.schema import Schema
    return NodeDef(
        node_type=NODE_TYPE_REPLAY_QUERY,
        input_schema=Schema(fields=[]), # Often takes no direct graph inputs, relies on params
        output_schema=output_schema,
        description="Queries the replay buffer with constraints."
    )

def create_schedule_def() -> NodeDef:
    """Creates a definition for a Schedule node."""
    from core.graph import NODE_TYPE_SCHEDULE
    from core.schema import Schema
    return NodeDef(
        node_type=NODE_TYPE_SCHEDULE,
        input_schema=Schema(fields=[]),
        output_schema=Schema(fields=[]),
        description="Defines a declarative execution schedule for actors and learners."
    )

def create_target_sync_def() -> NodeDef:
    """Creates a definition for a TargetSync node."""
    from core.graph import NODE_TYPE_TARGET_SYNC
    from core.schema import Schema
    return NodeDef(
        node_type=NODE_TYPE_TARGET_SYNC,
        input_schema=Schema(fields=[]),
        output_schema=Schema(fields=[]),
        description="Synchronizes parameters between online and target networks."
    )

# --- Registry ---

class NodeRegistry:
    """Registry for mapping abstract node categories to specific definitions."""
    def __init__(self):
        self._registry: Dict[str, NodeDef] = {}

    def register(self, name: str, node_def: NodeDef):
        self._registry[name] = node_def

    def get(self, name: str) -> NodeDef:
        assert name in self._registry, f"Node definition '{name}' not found in registry."
        return self._registry[name]

# Default global registry
registry = NodeRegistry()
