from __future__ import annotations

import re
from types import SimpleNamespace

from watermark_suite.attacks.synonym_replacement import SynonymReplacer


class FakeTokenizer:
    mask_token = "[MASK]"
    mask_token_id = 999
    model_max_length = 10

    def __init__(self) -> None:
        self._token_to_id = {self.mask_token: self.mask_token_id}
        self._id_to_token = {self.mask_token_id: self.mask_token}

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        tokens = re.findall(r"\[MASK\]|\w+|[^\w\s]", text)
        result = []
        for token in tokens:
            if token not in self._token_to_id:
                identity = len(self._token_to_id) + 1
                self._token_to_id[token] = identity
                self._id_to_token[identity] = token
            result.append(self._token_to_id[token])
        return result

    @staticmethod
    def num_special_tokens_to_add(*, pair):
        assert pair is False
        return 2

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens,
        clean_up_tokenization_spaces,
    ):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return " ".join(self._id_to_token[value] for value in token_ids)


class FakeUnmasker:
    def __init__(self) -> None:
        self.model = SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=10)
        )
        self.requests = []

    def __call__(self, texts, *, top_k, batch_size):
        assert top_k == 15
        assert batch_size == 4
        self.requests.extend(texts)
        predictions = [
            {"token_str": "animals", "score": 0.6},
            {"token_str": "pursue", "score": 0.2},
            {"token_str": "softly", "score": 0.1},
            {"token_str": "bright", "score": 0.1},
        ]
        return [list(predictions) for _ in texts]


def fake_pos_tagger(tokens):
    tags = {
        "Cats": "NNS",
        "cats": "NNS",
        "chase": "VBP",
        "mice": "NNS",
        "quickly": "RB",
        "animals": "NNS",
        "pursue": "VBP",
        "softly": "RB",
        "bright": "JJ",
    }
    return [(token, tags.get(token, ".")) for token in tokens]


def test_replacement_is_seeded_and_preserves_original_formatting():
    tokenizer = FakeTokenizer()
    unmasker = FakeUnmasker()
    replacer = SynonymReplacer(
        tokenizer,
        unmasker,
        pos_tagger=fake_pos_tagger,
    )
    original = "Cats  chase mice,\nquickly."

    first = replacer.replace_many(
        [original],
        seeds=[42],
        replacement_rate=0.75,
        top_k=15,
        candidate_sampling="score-weighted",
        batch_size=4,
    )[0]
    second = replacer.replace_many(
        [original],
        seeds=[42],
        replacement_rate=0.75,
        top_k=15,
        candidate_sampling="score-weighted",
        batch_size=4,
    )[0]

    assert first == second
    assert first.text != original
    assert re.fullmatch(r"\w+  \w+ \w+,\n\w+\.", first.text)
    assert first.eligible_word_count == 4
    assert first.selected_word_count == 3
    assert first.replaced_word_count == 3
    assert first.failed_replacement_count == 0
    assert first.realized_replacement_rate == 0.75
    assert first.mean_selected_candidate_rank is not None


def test_long_text_uses_mask_centered_windows_without_duplication():
    tokenizer = FakeTokenizer()
    unmasker = FakeUnmasker()
    replacer = SynonymReplacer(
        tokenizer,
        unmasker,
        pos_tagger=fake_pos_tagger,
    )
    original = " ".join(["cats"] * 40)

    result = replacer.replace_many(
        [original],
        seeds=[7],
        replacement_rate=0.1,
        top_k=15,
        candidate_sampling="score-weighted",
        batch_size=4,
    )[0]

    assert len(result.text.split()) == 40
    assert result.selected_word_count == 4
    assert result.replaced_word_count == 4
    assert len(unmasker.requests) == 4
    assert all(len(tokenizer.encode(text, add_special_tokens=False)) <= 8
               for text in unmasker.requests)


def test_failed_candidates_are_reported_without_changing_text():
    class InvalidUnmasker(FakeUnmasker):
        def __call__(self, texts, *, top_k, batch_size):
            del top_k, batch_size
            return [
                [{"token_str": "123", "score": 1.0}]
                for _ in texts
            ]

    replacer = SynonymReplacer(
        FakeTokenizer(),
        InvalidUnmasker(),
        pos_tagger=fake_pos_tagger,
    )
    original = "cats chase mice quickly"

    result = replacer.replace_many(
        [original],
        seeds=[9],
        replacement_rate=0.5,
        top_k=15,
        candidate_sampling="score-weighted",
        batch_size=4,
    )[0]

    assert result.text == original
    assert result.selected_word_count == 2
    assert result.replaced_word_count == 0
    assert result.failed_replacement_count == 2
    assert result.mean_selected_candidate_rank is None
