def test_latent_consistency_head_custom_strategy():
    """Verifies that a custom strategy overrides the default projection dimension."""
    torch.manual_seed(42)
    arch_config = ArchitectureConfig({"noisy_sigma": 0.0})

    # FIX: Changed 'num_bins' to 'output_size' to match ScalarStrategy __init__
    strategy = ScalarStrategy(output_size=4)

    head = LatentConsistencyHead(
        arch_config=arch_config,
        input_shape=(16,),
        strategy=strategy,
        projection_dim=256,  # This should be overridden by the strategy's bin count
    )

    x = torch.randn(2, 16)
    logits, state, projected = head(x)
    assert projected.shape == (2, 4)
