"""Byte-level BPE tokenizer for Pip. Pure Python, no extra dependencies.

Trained on the training text itself. The base vocabulary is all 256 bytes, so any
character (including unseen ones) can be encoded. Words are split with a simple
regex so merges never cross word boundaries (the leading space stays with its word).
"""
import heapq
import json
import re
from collections import Counter, defaultdict

PAT = re.compile(r"\s?\S+|\s+")


def _merge_word(word, pair, new_id):
    a, b = pair
    out = []
    i = 0
    n = len(word)
    while i < n:
        if i < n - 1 and word[i] == a and word[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


class BPETokenizer:
    def __init__(self, merges=None):
        self.merges = [tuple(m) for m in (merges or [])]
        self.ranks = {p: i for i, p in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for k, (a, b) in enumerate(self.merges):
            self.vocab[256 + k] = self.vocab[a] + self.vocab[b]
        self._cache = {}

    @property
    def vocab_size(self):
        return 256 + len(self.merges)

    @classmethod
    def train(cls, text, vocab_size):
        num_merges = max(0, vocab_size - 256)
        word_freq = Counter(tuple(w.encode("utf-8")) for w in PAT.findall(text))

        pair_counts = defaultdict(int)
        pair_words = defaultdict(set)
        for w, f in word_freq.items():
            for p in zip(w, w[1:]):
                pair_counts[p] += f
                pair_words[p].add(w)

        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        merges = []
        while len(merges) < num_merges and heap:
            neg, pair = heapq.heappop(heap)
            cur = pair_counts.get(pair, 0)
            if cur <= 0:
                continue
            if cur != -neg:
                # stale heap entry: re-insert with the current count
                heapq.heappush(heap, (-cur, pair))
                continue

            new_id = 256 + len(merges)
            merges.append(pair)
            for w in list(pair_words[pair]):
                f = word_freq.pop(w, 0)
                if not f:
                    continue
                for p in zip(w, w[1:]):
                    pair_counts[p] -= f
                    pair_words[p].discard(w)
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                nw = _merge_word(w, pair, new_id)
                word_freq[nw] = word_freq.get(nw, 0) + f
                for p in zip(nw, nw[1:]):
                    pair_counts[p] += f
                    pair_words[p].add(nw)
                    heapq.heappush(heap, (-pair_counts[p], p))
            pair_words.pop(pair, None)

        return cls(merges)

    def _bpe(self, key):
        ids = list(key)
        while len(ids) >= 2:
            best, best_rank = None, None
            for p in zip(ids, ids[1:]):
                r = self.ranks.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = p, r
            if best is None:
                break
            ids = list(_merge_word(tuple(ids), best, 256 + best_rank))
        return ids

    def encode(self, text):
        out = []
        for w in PAT.findall(text):
            key = w.encode("utf-8")
            if key not in self._cache:
                self._cache[key] = self._bpe(key)
            out.extend(self._cache[key])
        return out

    def decode(self, ids):
        return b"".join(self.vocab[int(i)] for i in ids).decode("utf-8", errors="replace")

    def state(self):
        return {"merges": [list(m) for m in self.merges]}

    @classmethod
    def from_state(cls, state):
        return cls([tuple(m) for m in state["merges"]])

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.state(), f)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_state(json.load(f))
