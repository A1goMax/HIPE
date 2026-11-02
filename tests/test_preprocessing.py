from datasets import Dataset
from hipe.data import tokenize_and_chunk


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        ids = [ord(c) % 16 for c in text]
        return [99] + ids if add_special_tokens else ids

    def __call__(self, texts, truncation=False, add_special_tokens=False):
        return {"input_ids": [self.encode(text, add_special_tokens) for text in texts]}


def test_c4_chunk_boundaries_match_original_single_map():
    data = Dataset.from_dict({"text": ["ab", "", "cdef", "ghi"]})
    actual = tokenize_and_chunk(data, Tokenizer(), 3, protocol="legacy_c4")
    ids = [ord(c) % 16 for c in "abcdefghi"]
    assert actual["input_ids"] == [ids[:4], ids[4:8]]


def test_wiki_special_token_policy_is_explicit():
    data = Dataset.from_dict({"text": ["abc", "def"]})
    legacy = tokenize_and_chunk(data, Tokenizer(), 3, protocol="legacy_wiki")
    release = tokenize_and_chunk(data, Tokenizer(), 3, protocol="release")
    assert legacy["input_ids"][0][0] == 99
    assert release["input_ids"][0][0] != 99
    assert all(len(x) == 4 for x in legacy["input_ids"])
