from curate.descriptors import descriptor_features


def test_descriptor_features_basic():
    metrics = {
        "perplexity": {"overall": 9.0},
        "toxicity": {"tss_95th": 0.8},
        "helpfulness": {"avg_information_density": 0.4},
        "diversity": {"self_bleu": 0.2},
        "compliance": {"compliance_rate": 1.0},
    }
    feats = descriptor_features(metrics)
    assert feats["ppl_inv"] == 0.1
    assert feats["self_bleu_inv"] == 0.8
    assert feats["tss95"] == 0.8
