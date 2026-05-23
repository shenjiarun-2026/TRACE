from trace.pairwise import rank_to_score, combined_rank_score, make_pair_features


def test_rank_to_score():
    assert rank_to_score(1, 5) == 1.0
    assert rank_to_score(5, 5) == 0.0


def test_combined_rank_score():
    score = combined_rank_score(1, 2, 5, 0.7)
    assert 0.0 <= score <= 1.0


def test_make_pair_features_concat():
    vals, names = make_pair_features({"a": 1.0}, {"a": 2.0}, ["a"], "concat")
    assert vals == [1.0, 2.0]
    assert names == ["d1__a", "d2__a"]
