from hipe.config import build_model_config, default_layer_threshold


def test_default_bipartite_splits():
    assert default_layer_threshold(8) == 3
    assert default_layer_threshold(16) == 7


def test_main_method_configuration():
    config = build_model_config("60M", vocab_size=128, sequence_length=1024, sigma=500.0)
    assert config.use_scaled_rope1 is True
    assert config.rope_scaling_threshold == 3
    assert config.decay_func == "gaussian"
