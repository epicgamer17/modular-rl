"""
Standard transfer operators for the RL IR.
Handles data movement between devices and processes.
"""

import torch
from typing import Dict, Any, Optional
from core.graph import Node
from runtime.dataref import DataRef, StorageLocation
from runtime.context import ExecutionContext

def op_transfer_to_device(node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None) -> DataRef:
    data = list(inputs.values())[0]
    device_id = node.params.get("device_id")
    
    # If it's already a DataRef, move it
    if isinstance(data, DataRef):
        data.move_to(StorageLocation.GPU, device_id=device_id)
        return data
    
    # Otherwise, wrap and move
    ref = DataRef(data, location=StorageLocation.CPU)
    ref.move_to(StorageLocation.GPU, device_id=device_id)
    return ref

def op_transfer_to_cpu(node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None) -> DataRef:
    data = list(inputs.values())[0]
    
    if isinstance(data, DataRef):
        data.move_to(StorageLocation.CPU)
        return data
        
    # Already on CPU likely, just wrap
    return DataRef(data, location=StorageLocation.CPU)

def op_prefetch(node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None) -> Any:
    """
    Prefetch hint node. 
    In sequential executor, this is a no-op that passes data through.
    In async executor, this triggers non-blocking transfer.
    """
    return list(inputs.values())[0]

def op_serialize(node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None) -> bytes:
    data = list(inputs.values())[0]
    # Use torch.save or pickle for serialization
    import io
    buf = io.BytesIO()
    torch.save(data.data if isinstance(data, DataRef) else data, buf)
    return buf.getvalue()

def op_deserialize(node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None) -> DataRef:
    raw_data = list(inputs.values())[0]
    import io
    buf = io.BytesIO(raw_data)
    data = torch.load(buf)
    return DataRef(data, location=StorageLocation.CPU)

def register_transfer_operators(register_fn):
    register_fn("TransferToDevice", op_transfer_to_device)
    register_fn("TransferToCPU", op_transfer_to_cpu)
    register_fn("Prefetch", op_prefetch)
    register_fn("Serialize", op_serialize)
    register_fn("Deserialize", op_deserialize)
