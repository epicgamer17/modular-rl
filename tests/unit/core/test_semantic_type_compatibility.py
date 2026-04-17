import pytest
from core.contracts import SemanticType, Policy, Probs, Logits, Action

pytestmark = pytest.mark.unit

def test_semantic_type_compatibility_abstract_to_concrete():
    """Verify that an abstract type (e.g. Policy) accepts a concrete representation (e.g. Policy[Probs])."""
    assert Policy.is_compatible(Policy[Probs]), "Policy should be compatible with Policy[Probs]"
    assert Policy.is_compatible(Policy[Logits]), "Policy should be compatible with Policy[Logits]"

def test_semantic_type_compatibility_concrete_to_abstract():
    """Verify that a concrete representation is NOT satisfied by an abstract requirement (Strict Requirement)."""
    assert not Policy[Probs].is_compatible(Policy), "Concrete requirement Policy[Probs] should NOT be satisfied by abstract provider Policy"

def test_semantic_type_compatibility_identical_concrete():
    """Verify that identical concrete representations are compatible."""
    assert Policy[Probs].is_compatible(Policy[Probs]), "Identical concrete types must be compatible"

def test_semantic_type_compatibility_different_concrete():
    """Verify that different concrete representations of the same semantic type are NOT compatible."""
    assert not Policy[Probs].is_compatible(Policy[Logits]), "Policy[Probs] and Policy[Logits] must be incompatible"

def test_semantic_type_compatibility_different_base():
    """Verify that different base semantic types are never compatible."""
    assert not Policy.is_compatible(Action), "Policy and Action must be incompatible"
    assert not Policy[Probs].is_compatible(Action), "Policy[Probs] and Action must be incompatible"
