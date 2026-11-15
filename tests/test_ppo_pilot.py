import pytest

from prism.ppo_pilot import development_split


def test_development_split_is_order_independent_disjoint_and_complete():
    prompts = [{"id": str(i), "messages": [{"role": "user", "content": str(i)}]} for i in range(2000)]
    split = development_split(prompts)
    assert list(map(len, split)) == [64, 64, 1872]
    assert split == development_split(list(reversed(prompts)))
    assert set(row["id"] for part in split for row in part) == set(map(str, range(2000)))
    prompts[1]["messages"] = prompts[0]["messages"]
    with pytest.raises(ValueError, match="distinct"):
        development_split(prompts)
