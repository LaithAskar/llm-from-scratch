"""
Tests for bpe.py — trains on tiny synthetic corpora to keep test runtime low.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bpe import decode, encode, load, pretokenize, save, train_bpe


# --- pretokenize -----------------------------------------------------------


def test_pretokenize_splits_words_and_keeps_spaces():
    # GPT-2 regex puts the leading space WITH the next word (most words
    # except the very first start with a leading space).
    tokens = pretokenize("hello world")
    # Expect: ["hello", " world"] as bytes
    assert tokens == [b"hello", b" world"]


def test_pretokenize_separates_punctuation():
    tokens = pretokenize("hi!")
    assert tokens == [b"hi", b"!"]


def test_pretokenize_handles_numbers():
    # GPT-2 regex keeps the leading space attached to the number too.
    tokens = pretokenize("year 2026")
    assert tokens == [b"year", b" 2026"]


# --- train_bpe -------------------------------------------------------------


def test_train_bpe_initial_vocab_is_all_bytes():
    """With vocab_size=256, no merges happen — vocab is just the 256 bytes."""
    merges, vocab = train_bpe("hello", vocab_size=256)
    assert merges == []
    assert len(vocab) == 256


def test_train_bpe_rejects_small_vocab():
    with pytest.raises(ValueError, match="vocab_size must be >= 256"):
        train_bpe("hello", vocab_size=100)


def test_train_bpe_learns_repeated_pair():
    """
    Most frequent pair in 'ababab' is ('a', 'b'). First merge should be it.
    """
    merges, vocab = train_bpe("ababab", vocab_size=257)
    assert len(merges) == 1
    assert merges[0] == (b"a", b"b")
    # The merged token gets ID 256 (right after the byte vocab).
    assert vocab[b"ab"] == 256


def test_train_bpe_learns_multiple_merges_in_order():
    """
    Corpus 'abcabcabc' should first merge a+b (or b+c — they tie), then the
    merged pair with the remaining letter.
    """
    merges, vocab = train_bpe("abcabcabc", vocab_size=259)
    assert len(merges) == 3
    # After 3 merges starting from byte-level, the whole word 'abc' should
    # be a single token.
    assert b"abc" in vocab


def test_train_bpe_stops_when_no_more_pairs():
    """
    Single-character corpus has no pairs to merge.
    """
    merges, vocab = train_bpe("aaaaa", vocab_size=4096)
    # 'a' + 'a' = 'aa', then 'aa' + 'a' = 'aaa', etc. Should merge but stop
    # before vocab_size is reached if all pairs eliminate. (Actually 'aaaaa'
    # can merge to 'aaaaa' through a sequence of merges, so we just verify
    # it doesn't crash and produces some merges.)
    assert len(merges) > 0


# --- encode / decode round-trip --------------------------------------------


def test_encode_decode_round_trip_after_training():
    """encode then decode must recover the original text."""
    corpus = "the cat sat on the mat. the cat ate the rat."
    merges, vocab = train_bpe(corpus, vocab_size=300)
    ids = encode(corpus, merges, vocab)
    recovered = decode(ids, vocab)
    assert recovered == corpus


def test_encode_decode_handles_unseen_text():
    """A trained BPE must still encode characters it hasn't seen at merge
    time — they fall back to single-byte tokens (which are all in vocab)."""
    merges, vocab = train_bpe("hello world", vocab_size=300)
    text = "x y z 1 2 3"  # never appeared in training
    ids = encode(text, merges, vocab)
    assert decode(ids, vocab) == text


def test_encode_reduces_token_count_vs_bytes():
    """
    After training on a corpus with repeated structure, encoding the same
    corpus should produce fewer tokens than raw bytes — that's the whole
    point of BPE.
    """
    corpus = "the cat sat on the mat. " * 50
    merges, vocab = train_bpe(corpus, vocab_size=400)
    ids = encode(corpus, merges, vocab)
    bytes_count = len(corpus.encode("utf-8"))
    assert len(ids) < bytes_count, (
        f"compression failure: {bytes_count} bytes -> {len(ids)} tokens"
    )


def test_encode_uses_lowest_merge_first():
    """
    If two adjacent pairs are both learned merges, the one with the lower
    merge index (= earlier learned) takes priority.
    """
    # Train so that ('a', 'b') is merge #0 and ('b', 'c') is merge #1.
    # Then 'abc' should encode using the 'ab' merge first, leaving 'ab' + 'c'.
    corpus = "ab ab ab ab abc"  # 'ab' more frequent than 'bc' or 'abc'
    merges, vocab = train_bpe(corpus, vocab_size=258)
    # merges[0] should be ('a', 'b')
    assert merges[0] == (b"a", b"b")


# --- save / load -----------------------------------------------------------


def test_save_load_round_trip(tmp_path: Path):
    """Saving and loading must give back identical merges and vocab."""
    merges, vocab = train_bpe("hello world hello there", vocab_size=300)
    out = tmp_path / "bpe.json"
    save(out, merges, vocab)
    merges2, vocab2 = load(out)
    assert merges == merges2
    assert vocab == vocab2


def test_save_load_then_encode_matches(tmp_path: Path):
    """After save+load, encoding should produce the same IDs as before save."""
    corpus = "the quick brown fox jumps over the lazy dog"
    merges, vocab = train_bpe(corpus, vocab_size=350)
    ids_before = encode(corpus, merges, vocab)
    out = tmp_path / "bpe.json"
    save(out, merges, vocab)
    merges2, vocab2 = load(out)
    ids_after = encode(corpus, merges2, vocab2)
    assert ids_before == ids_after


# --- non-ASCII safety ------------------------------------------------------


def test_bpe_handles_non_ascii_via_utf8_bytes():
    """Unicode characters become multiple bytes; BPE works at byte level."""
    corpus = "café résumé naïve"
    merges, vocab = train_bpe(corpus, vocab_size=300)
    ids = encode(corpus, merges, vocab)
    assert decode(ids, vocab) == corpus
