"""Byte-level BPE tokenizer, built from tokenizer.json alone.

Pipeline (encode):  special-token split -> normalize (if the file asks) -> regex pre-splits -> bytes
                    -> stand-in chars -> BPE merges by priority -> vocab lookup [-> BOS in front, if asked].
Decode runs the BPE steps backwards.
"""
import json
import unicodedata
from pathlib import Path

import regex  # third-party engine: needed for \p{L} / \p{N} classes that the stdlib `re` lacks

# HF tokenizers runs these regexes with Oniguruma, whose \w differs from the `regex` module's on 8 characters: ZWNJ
# and ZWJ are word characters only in `regex`; ² ³ ¹ ¼ ½ ¾ only in Oniguruma. Tiny Aya's digit split ends in \b, so
# a number written before a ZWNJ (common in Persian: ۱۹۷۰‌ها) would be grouped differently. \b is rewritten to
# Oniguruma's word boundary; \w, \W and \B would need the same and are refused.
_ONIG_W = r"(?:(?![\u200c\u200d])\w|[\u00b2\u00b3\u00b9\u00bc-\u00be])"
_ONIG_B = rf"(?:(?<={_ONIG_W})(?!{_ONIG_W})|(?<!{_ONIG_W})(?={_ONIG_W}))"


def oniguruma_compatible(pattern: str) -> str:
    r"""The pattern with every \b outside a character class replaced by Oniguruma's word boundary."""
    out, i, in_class = [], 0, False
    while i < len(pattern):
        c = pattern[i]
        if c == "\\" and i + 1 < len(pattern):
            e = pattern[i + 1]
            if e in "wWB" or (e == "b" and in_class):
                raise NotImplementedError(f"\\{e} in a pre-tokenizer regex: {pattern!r}")
            out.append(_ONIG_B if e == "b" else c + e)
            i += 2
            continue
        in_class = (in_class and c != "]") or (not in_class and c == "[")
        out.append(c)
        i += 1
    return "".join(out)


def build_byte_alphabet() -> tuple[dict[int, str], dict[str, int]]:
    """The 256 atoms: every byte value 0..255 gets a printable stand-in character.
    Printable bytes stand for themselves; the rest are shifted up to 256+ so they stay visible
    (space -> 'Ġ', newline -> 'Ċ'). This is GPT-2's bytes_to_unicode, which Qwen inherits."""
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    byte_to_char: dict[int, str] = {}
    nxt = 256
    for b in range(256):
        if b in printable:
            byte_to_char[b] = chr(b)
        else:
            byte_to_char[b] = chr(nxt)
            nxt += 1
    char_to_byte = {c: b for b, c in byte_to_char.items()}
    return byte_to_char, char_to_byte


class Tokenizer:
    def __init__(self, path: str):
        spec = json.load(open(path, encoding="utf-8"))
        model = spec["model"]

        self.vocab: dict[str, int] = model["vocab"]                       # token string -> id
        self.id_to_token: dict[int, str] = {i: t for t, i in self.vocab.items()}

        # merges are either "a b" strings or [a, b] pairs depending on file version; rank = position
        self.merge_rank: dict[tuple[str, str], int] = {}
        for rank, m in enumerate(model["merges"]):
            a, b = m.split(" ") if isinstance(m, str) else m
            self.merge_rank[(a, b)] = rank

        if any(t.get(f) for t in spec["added_tokens"] for f in ("lstrip", "rstrip", "single_word", "normalized")):
            raise NotImplementedError("added tokens with lstrip / rstrip / single_word / normalized set")
        self.special: dict[str, int] = {t["content"]: t["id"] for t in spec["added_tokens"]}
        # Some checkpoints declare extra added tokens only in tokenizer_config.json (Qwen3.5: 248070..248076)
        cfg_path = Path(path).with_name("tokenizer_config.json")
        if cfg_path.exists():
            for tid, t in json.load(open(cfg_path, encoding="utf-8")).get("added_tokens_decoder", {}).items():
                self.special.setdefault(t["content"], int(tid))
        self.id_to_token.update({i: t for t, i in self.special.items()})
        self.special_re = regex.compile("(" + "|".join(regex.escape(s) for s in self.special) + ")") if self.special \
            else None

        # Stage 1, normalization: Qwen3.5's file asks for NFC; Tiny Aya's has none (text is used as given).
        norm = spec["normalizer"]
        if norm not in (None, {"type": "NFC"}):
            raise NotImplementedError(f"normalizer {norm}")
        self.nfc = norm is not None

        # Stage 2, pre-tokenization: a Sequence of regex Splits, then ByteLevel. Each Split is "Isolated": a match
        # becomes its own piece and the text between matches stays as pieces too. Qwen3.5 has one Split whose regex
        # covers every character; Tiny Aya first splits off digit groups (3 at a time, from the right), then splits
        # every piece with a GPT-4o-style regex. ByteLevel with use_regex=false only maps bytes to stand-in chars.
        # The regexes are taken from the file verbatim (the `regex` engine supports (?i:...)), except that \b gets
        # Oniguruma's meaning (see oniguruma_compatible). Known, rare differences from HF that remain: `regex` uses
        # Unicode 17 tables and Oniguruma Unicode 16, so ~4,700 code points new in Unicode 17 are letters or digits
        # only here; and Python's NFC (Qwen3.5) composes around a few dozen combining marks newer than tokenizers' NFC
        # tables, which treat them as starters.
        # Note: for Qwen3.5 checkpoints, transformers' AutoTokenizer resolves to Qwen2Tokenizer (per the checkpoint's
        # tokenizer_config.json) and substitutes the older Qwen2 regex, which lacks \p{M}; it then splits combining
        # marks (Hindi, Tamil, Thai, Arabic) differently from this file. The file and transformers' own
        # Qwen3_5Tokenizer agree with each other, and this engine follows them.
        pt = spec["pre_tokenizer"]
        steps = pt["pretokenizers"] if pt["type"] == "Sequence" else [pt]
        *splits, last = steps
        if last["type"] != "ByteLevel" or last["use_regex"] or last["add_prefix_space"]:
            raise NotImplementedError(f"pre-tokenizer must end with a plain ByteLevel step, got {last}")
        for p in splits:
            if p["type"] != "Split" or p["behavior"] != "Isolated" or p["invert"]:
                raise NotImplementedError(f"pre-tokenizer step {p}")
        self.splits = [regex.compile(oniguruma_compatible(p["pattern"]["Regex"])) for p in splits]

        if model["type"] != "BPE" or model.get("ignore_merges") or model.get("byte_fallback") or model.get("dropout"):
            raise NotImplementedError("only plain byte-level BPE is implemented")

        # After BPE: Tiny Aya's post-processor (TemplateProcessing) puts <BOS_TOKEN> in front of every sequence;
        # Qwen3.5's (ByteLevel) only adjusts offsets. A Sequence may hold ByteLevel steps and one template.
        pp = spec.get("post_processor")
        procs = (pp["processors"] if pp["type"] == "Sequence" else [pp]) if pp else []
        templates = [x for x in procs if x["type"] == "TemplateProcessing"]
        if len(templates) > 1 or any(x["type"] not in ("ByteLevel", "TemplateProcessing") for x in procs):
            raise NotImplementedError(f"post-processor {pp}")
        single = templates[0]["single"] if templates else [{"Sequence": {"id": "A"}}]
        if [next(iter(x)) for x in single] == ["SpecialToken", "Sequence"]:
            self.bos_id: int | None = templates[0]["special_tokens"][single[0]["SpecialToken"]["id"]]["ids"][0]
        elif [next(iter(x)) for x in single] == ["Sequence"]:
            self.bos_id = None
        else:
            raise NotImplementedError(f"post-processor template {single}")

        self.byte_to_char, self.char_to_byte = build_byte_alphabet()

    # ---------- Stage 2: regex pre-splits ----------
    def _pieces(self, text: str) -> list[str]:
        """Apply every Split in turn to every piece so far. Isolated: each match is its own piece, and the text
        between matches is kept as pieces too (a plain findall would silently drop it)."""
        pieces = [text]
        for rx in self.splits:
            out = []
            for s in pieces:
                last = 0
                for m in rx.finditer(s):
                    if m.start() > last:
                        out.append(s[last:m.start()])     # the gap before this match
                    if m.end() > m.start():
                        out.append(m.group())             # the match itself (empty matches add nothing)
                    last = m.end()
                if last < len(s):
                    out.append(s[last:])                  # the tail after the last match
            pieces = out
        return pieces

    # ---------- Stage 3: BPE merging for ONE chunk ----------
    def _bpe(self, chunk: str) -> list[str]:
        symbols = list(chunk)                                  # start as single stand-in characters
        while len(symbols) > 1:
            # find the adjacent pair with the best (lowest) merge rank
            best, best_rank = None, None
            for pair in zip(symbols, symbols[1:]):
                r = self.merge_rank.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = pair, r
            if best is None:
                break                                          # nothing left to merge
            # glue every occurrence of that pair
            merged, out, i = best[0] + best[1], [], 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best:
                    out.append(merged)
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            symbols = out
        return symbols

    # ---------- Encode: text -> ids ----------
    def encode(self, text: str, add_bos: bool = False) -> list[int]:
        """add_bos: put the model's BOS in front (only if its post-processor has one), as for a raw prompt.
        Off by default: a chat template writes BOS itself, and a continuation (a decision option, a next turn)
        must never get one."""
        ids: list[int] = [self.bos_id] if add_bos and self.bos_id is not None else []
        for part in self.special_re.split(text) if self.special_re else [text]:   # Stage 0: peel off specials
            if not part:
                continue
            if part in self.special:
                ids.append(self.special[part])
                continue
            if self.nfc:
                part = unicodedata.normalize("NFC", part)                # Stage 1
            for chunk in self._pieces(part):                              # Stage 2: word-like chunks
                mapped = "".join(self.byte_to_char[b] for b in chunk.encode("utf-8"))  # Stage 2b
                for piece in self._bpe(mapped):                           # Stage 3
                    ids.append(self.vocab[piece])                         # vocab lookup
        return ids

    # ---------- Decode: ids -> text ----------
    def decode(self, ids: list[int]) -> str:
        out = bytearray()
        for i in ids:
            tok = self.id_to_token[i]
            if tok in self.special:
                out += tok.encode("utf-8")
            else:
                out += bytes(self.char_to_byte[c] for c in tok)           # stand-in char -> byte
        return out.decode("utf-8", errors="replace")

    def vocab_size(self) -> int:
        return len(self.id_to_token)
