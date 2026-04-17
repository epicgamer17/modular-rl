"""
Ingestion layer: adapters that convert raw environment data (Sequences,
Transitions, compressed observations) into tensorized form suitable for
the replay buffer.

These objects must NEVER be passed into the buffer, sampler, or writer
directly.  They live upstream of the storage boundary.
"""

from .sequence import Sequence, TimeStep
from .transition import Transition, TransitionBatch
from .compression import (
    ObservationCompressionProcessor,
    LazyDecompressedBuffer,
    ObservationDecompressionProcessor,
)
from .sequence_processor import SequenceTensorProcessor
