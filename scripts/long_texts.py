"""Long public-domain texts in eight languages, for Tiny Aya's long-context checks (M5: past its 4096-token window).

They are downloaded on demand into models/long_texts/ (gitignored) and never committed. Each Project Gutenberg file
is checked against a pinned SHA-256. Each Wikisource text is fetched at pinned page revisions (rendered by Wikisource,
so the text of the scanned pages it transcludes is included) and the extracted text is checked against a pinned
SHA-256. The same tokens therefore come back every time, and a changed source fails loudly instead.

build(tok, n) takes about n tokens of each work's own text (from an anchor sentence where it begins, past prefaces,
tables of contents and transcribers' notes), in a fixed order, separated by blank lines.

usage: long_texts.py [--tokens-per-language 600]   (downloads what is missing, prints each section's token count)
"""
import argparse
import hashlib
import html
import json
import re
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "models" / "long_texts"
AGENT = {"User-Agent": "metal-inference-engine long-context test (github.com/MeKaustubh07/metal-inference-engine)"}

# language -> (source, what, SHA-256, start). Gutenberg: the ebook number; Wikisource: (site, page revisions). start:
# the work's first words (its text begins there) or a number of characters to skip. All public domain: the authors died
# before 1930 or the works are centuries old.
SOURCES = {
    "en": (1342, "Pride and Prejudice (Jane Austen, 1813)",
           "3f6bb9d6f78e0293b56acd4714dd68cb7d6d1d293402031ce9d5a216bcaf9d75",
           "It is a truth universally acknowledged"),
    "fr": (17489, "Les Misérables, tome I : Fantine (Victor Hugo, 1862)",
           "a5de514ba7b9f2e1790e7e259c4e8b7a35ae1d29e4bf9a5f8767039c58b80503",
           "En 1815, M. Charles-François-Bienvenu Myriel"),
    "de": (22367, "Die Verwandlung (Franz Kafka, 1915)",
           "359d3f5983c812393f4bd7ee49a350ffffc7d476015e58d6b84b6c848dd2b0e9", "Als Gregor Samsa eines Morgens"),
    "es": (2000, "Don Quijote (Miguel de Cervantes, 1605)",
           "534f41d59f7142163fa0964076ac6351845c006ea6433778be637fac6d5b04d7", "En un lugar de la Mancha"),
    "hi": (("hi.wikisource.org", (469932, 469936, 469940, 469944, 469953, 469956)),
           "अंधेर नगरी (Bharatendu Harishchandra, 1881), its six acts as in भारतेंदु-नाटकावली (1935)",
           "e2a49d8662097a2566547727f2ed2f7082c90948822f967366689d8eaeeea932", "पहला अंक"),
    "ar": (("ar.wikisource.org", (529885,)), "ألف ليلة وليلة، الجزء الأول (One Thousand and One Nights, vol. 1)",
           "a3d9f7f1bc6af9fd92b0a4e00c9f89579870bb56572122ee9b44be8cf23aa207", "حكي والله أعلم"),
    "zh": (23962, "西遊記 (Wu Cheng'en, 16th century)",
           "af3c9e408c0c58595b666ed9981b6fa1e9343f4bbc78309b1cb0818c32fc1f58", 2000),
    "ja": (1982, "羅生門 (Akutagawa Ryūnosuke, 1915)",
           "7585b90b3c25951420ddf4857a53964a3b4b892c8aab5a0200c3633b53bb2a1e", "或日の暮方の事である"),
}


def _get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=AGENT), timeout=120) as r:
        return r.read()


def _checked(data: bytes, sha: str | None, what: str) -> bytes:
    got = hashlib.sha256(data).hexdigest()
    if sha is not None and got != sha:
        raise ValueError(f"{what}: SHA-256 {got}, pinned {sha}: the source changed, check it and re-pin")
    return data


def _plain(page_html: str) -> str:
    """Wikisource's rendered HTML -> running text: no scripts, styles, page numbers or [edit] links."""
    h = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", page_html, flags=re.S)
    h = re.sub(r'<span class="pagenum[^"]*"[^>]*>.*?</span>', " ", h, flags=re.S)
    h = re.sub(r'<span class="mw-editsection">.*?</span>\s*</span>', " ", h, flags=re.S)
    t = html.unescape(re.sub(r"<[^>]+>", " ", h))
    t = re.sub(r"[ \t\u00a0\u200b\u2060\ufeff]+", " ", t)
    lines = re.sub(r"\s*\n\s*", "\n", t).strip().split("\n")
    return "\n".join(ln for ln in lines if "←" not in ln and "→" not in ln     # navigation header lines
                     and not re.match(r"\d{5,} ", ln))                         # "<page id> <book> <year> <author>"


def section(lang: str) -> str:
    """One language's text, downloaded if missing and checked against its pin, from where the work begins."""
    source, what, sha, start = SOURCES[lang]
    if isinstance(source, int):                                    # Project Gutenberg
        path = CACHE / f"pg{source}.txt"
        if not path.exists():
            data = _checked(_get(f"https://www.gutenberg.org/cache/epub/{source}/pg{source}.txt"), sha, what)
            CACHE.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        text = _checked(path.read_bytes(), sha, str(path)).decode("utf-8")
        text = text.split("*** START OF", 1)[1].split("\n", 1)[1].split("*** END OF", 1)[0]   # without the licence
    else:                                                          # Wikisource, at pinned revisions
        site, revs = source
        path = CACHE / f"{lang}-wikisource-{'-'.join(map(str, revs))}.txt"
        if not path.exists():
            parts = []
            for r in revs:                                         # one page a second: Wikisource rate-limits
                parts.append(_plain(json.loads(_get(f"https://{site}/w/api.php?action=parse&oldid={r}&prop=text"
                                                    "&format=json&formatversion=2"))["parse"]["text"]))
                time.sleep(1)
            data = _checked("\n\n".join(parts).encode("utf-8"), sha, what)
            CACHE.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        text = _checked(path.read_bytes(), sha, str(path)).decode("utf-8")
    text = re.sub(r"\n{3,}", "\n\n", text.replace("\r\n", "\n")).strip()
    return text[text.index(start):] if isinstance(start, str) else text[start:]


def parts(tok, n: int, langs=tuple(SOURCES)) -> list[str]:
    """About n tokens of each language: the shortest prefix of its section reaching n tokens with tok.encode, in the
    order of SOURCES. tok: anything with encode(text) -> list of ids."""
    out = []
    for lang in langs:
        text = section(lang)
        lo, hi = 1, len(text)
        while lo < hi:
            mid = (lo + hi) // 2
            lo, hi = (mid + 1, hi) if len(tok.encode(text[:mid])) < n else (lo, mid)
        out.append(text[:lo].strip())
    return out


def build(tok, n: int, langs=tuple(SOURCES)) -> str:
    """parts(), joined by blank lines: one long multilingual text."""
    return "\n\n".join(parts(tok, n, langs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens-per-language", type=int, default=600)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    hf = Tokenizer.from_file(str(ROOT / "models/tiny-aya-global/tokenizer.json"))

    class Tok:
        @staticmethod
        def encode(text):
            return hf.encode(text, add_special_tokens=False).ids
    ps = parts(Tok, a.tokens_per_language)
    for lang, part in zip(SOURCES, ps):
        print(f"{lang}: {len(Tok.encode(part))} tokens, {len(part)} characters | {SOURCES[lang][1]}")
    print(f"all: {len(Tok.encode(build(Tok, a.tokens_per_language)))} tokens")


if __name__ == "__main__":
    main()
