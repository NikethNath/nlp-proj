"""Deterministic Brahmic -> ASCII-Latin transliteration for Malayalam and Tamil.

This is the *canonical* romanization (condition L2): one spelling per word, no
diacritics, no digraphs a phone keyboard user wouldn't type. It is deliberately
lossy in the same directions real users are lossy - retroflex/dental contrasts
collapse, long vowels stay marked only where Indians conventionally mark them
(aa/ee/oo). It is NOT ISO 15919; a reversible scholarly scheme would be the
wrong baseline for a study about what happens when people type in Latin script.

Both scripts are abugidas: a bare consonant carries an inherent /a/, a vowel
sign replaces it, and the virama (chandrakkala / pulli) deletes it.
"""
from __future__ import annotations

import re
import unicodedata

ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍﻿"), None)

# --- Malayalam -------------------------------------------------------------
ML_VIRAMA = "്"
ML_INDEPENDENT_VOWELS = {
    "അ": "a", "ആ": "aa", "ഇ": "i", "ഈ": "ee", "ഉ": "u", "ഊ": "oo",
    "ഋ": "ru", "ൠ": "ru", "ഌ": "lu", "ൡ": "lu",
    "എ": "e", "ഏ": "e", "ഐ": "ai", "ഒ": "o", "ഓ": "o", "ഔ": "au",
}
ML_VOWEL_SIGNS = {
    "ാ": "aa", "ി": "i", "ീ": "ee", "ു": "u", "ൂ": "oo",
    "ൃ": "ru", "ൄ": "ru", "ൢ": "lu", "ൣ": "lu",
    "െ": "e", "േ": "e", "ൈ": "ai",
    "ൊ": "o", "ോ": "o", "ൌ": "au", "ൗ": "au",
}
ML_CONSONANTS = {
    "ക": "k", "ഖ": "kh", "ഗ": "g", "ഘ": "gh", "ങ": "ng",
    "ച": "ch", "ഛ": "chh", "ജ": "j", "ഝ": "jh", "ഞ": "nj",
    "ട": "t", "ഠ": "th", "ഡ": "d", "ഢ": "dh", "ണ": "n",
    "ത": "th", "ഥ": "th", "ദ": "d", "ധ": "dh", "ന": "n",
    "പ": "p", "ഫ": "f", "ബ": "b", "ഭ": "bh", "മ": "m",
    "യ": "y", "ര": "r", "ല": "l", "വ": "v",
    "ശ": "sh", "ഷ": "sh", "സ": "s", "ഹ": "h",
    "ള": "l", "ഴ": "zh", "റ": "r", "ൺ": "n", "ൻ": "n", "ർ": "r",
    "ൽ": "l", "ൾ": "l", "ൿ": "k", "ഺ": "t", "ൔ": "m", "ൕ": "y", "ൖ": "zh",
}
# Chillus are consonants that are already vowel-less; no inherent /a/ follows.
ML_CHILLU = set("ൺൻർൽൾൿൔൕൖ")
ML_SIGNS = {"ം": "m", "ഃ": "h", "ഁ": "m"}  # anusvara, visarga, candrabindu
ML_DIGITS = {chr(0x0D66 + i): str(i) for i in range(10)}

# --- Tamil -----------------------------------------------------------------
TA_VIRAMA = "்"
TA_INDEPENDENT_VOWELS = {
    "அ": "a", "ஆ": "aa", "இ": "i", "ஈ": "ee", "உ": "u", "ஊ": "oo",
    "எ": "e", "ஏ": "e", "ஐ": "ai", "ஒ": "o", "ஓ": "o", "ஔ": "au",
}
TA_VOWEL_SIGNS = {
    "ா": "aa", "ி": "i", "ீ": "ee", "ு": "u", "ூ": "oo",
    "ெ": "e", "ே": "e", "ை": "ai",
    "ொ": "o", "ோ": "o", "ௌ": "au", "ௗ": "au",
}
TA_CONSONANTS = {
    "க": "k", "ங": "ng", "ச": "s", "ஞ": "nj", "ட": "t", "ண": "n",
    "த": "th", "ந": "n", "ப": "p", "ம": "m",
    "ய": "y", "ர": "r", "ல": "l", "வ": "v", "ழ": "zh", "ள": "l",
    "ற": "r", "ன": "n",
    "ஜ": "j", "ஷ": "sh", "ஸ": "s", "ஹ": "h", "ஶ": "sh",
}
TA_SIGNS = {"ஂ": "m", "்": ""}
TA_AYTHAM = {"ஃ": "h"}
TA_DIGITS = {chr(0x0BE6 + i): str(i) for i in range(10)}

# Tamil writes one grapheme for a voicing contrast that speakers do make and
# typists do write: intervocalic plosives voice. Users type "pogum", not
# "pokum". Applying this makes L2 a fairer stand-in for careful typed Tamil.
TA_VOICING = {"k": "g", "t": "d", "th": "dh", "p": "b"}
TA_NASAL_VOICING = {**TA_VOICING, "ch": "j"}
# Word-initially the same grapheme is an affricate: Chennai, not Sennai.
TA_INITIAL = {"ச": "ch"}

# Malayalam keeps voiced and voiceless stops as separate letters, so it needs
# far less of this - except after a homorganic nasal, where the cluster is
# always voiced (nda, mba, nja), and for intervocalic ta, which carries the
# genitive/dative inflections everyone writes as "-ude", "-idam".
ML_NASAL_VOICING = {"k": "g", "ch": "j", "t": "d", "p": "b"}
ML_INTERVOCALIC = {"t": "d"}
ML_NASALS = {"ng", "nj", "n", "m"}
# Conjuncts whose sound is not the sum of their letters. Measured against
# Dakshina: Malayalam rra+rra is typed "tt" 76% of the time (never "rr"), and
# na+rra "nt" 99%; Tamil rra+rra is "ttr" 52% / "tr" 46%, and na+rra "ndr" 70%.
# Newer Malayalam text may encode na+rra with an atomic chillu before the virama.
ML_CLUSTERS = {"റ്റ": "tt", "ന്റ": "nt", "ൻ്റ": "nt"}
TA_CLUSTERS = {"ற்ற": "ttr", "ன்ற": "ndr"}

# Word-final virama in Malayalam is the half-u (samvruthokaram): veedu, not veet.
ML_FINAL_SCHWA = "u"
# Geminate nasals are conventionally written single: njangal, not njangngal.
_GEMINATE_NASAL = re.compile(r"(ng|nj)\1")
_NASAL_VOICED = re.compile(r"ngg")

_VOWEL_CHARS = set("aeiou")


class _Script:
    def __init__(self, virama, vowels, signs, consonants, marks, digits,
                 chillu=frozenset(), voicing=None, nasal_voicing=None,
                 initial=None, final_schwa="", clusters=None):
        self.virama = virama
        self.vowels = vowels
        self.vowel_signs = signs
        self.consonants = consonants
        self.marks = marks
        self.digits = digits
        self.chillu = chillu
        self.voicing = voicing or {}
        self.nasal_voicing = nasal_voicing or {}
        self.initial = initial or {}
        self.final_schwa = final_schwa
        # Longest first, so a longer conjunct is never shadowed by a prefix.
        self.clusters = sorted((clusters or {}).items(), key=lambda kv: -len(kv[0]))


MALAYALAM = _Script(ML_VIRAMA, ML_INDEPENDENT_VOWELS, ML_VOWEL_SIGNS,
                    ML_CONSONANTS, ML_SIGNS, ML_DIGITS, ML_CHILLU,
                    voicing=ML_INTERVOCALIC, nasal_voicing=ML_NASAL_VOICING,
                    final_schwa=ML_FINAL_SCHWA, clusters=ML_CLUSTERS)
TAMIL = _Script(TA_VIRAMA, TA_INDEPENDENT_VOWELS, TA_VOWEL_SIGNS,
                TA_CONSONANTS, {**TA_SIGNS, **TA_AYTHAM}, TA_DIGITS,
                voicing=TA_VOICING, nasal_voicing=TA_NASAL_VOICING,
                initial=TA_INITIAL, clusters=TA_CLUSTERS)

SCRIPTS = {"ml": MALAYALAM, "ta": TAMIL}


def _transliterate_word(word: str, sc: _Script, voice: bool) -> str:
    out: list[str] = []
    prev_was_nasal_cluster = False   # last emission was a virama-killed nasal
    i, n = 0, len(word)
    while i < n:
        ch = word[i]
        cluster = next(((seq, rom) for seq, rom in sc.clusters if word.startswith(seq, i)), None)
        if cluster or ch in sc.consonants:
            if cluster:
                # A conjunct behaves as one consonant: it takes the following
                # vowel sign, inherent vowel or virama, and never voices.
                width, cons = len(cluster[0]), cluster[1]
            else:
                width = 1
                cons = sc.initial[ch] if (not out and ch in sc.initial) else sc.consonants[ch]
            last = i + width - 1
            nxt = word[last + 1] if last + 1 < n else ""
            final_schwa_here = bool(sc.final_schwa) and nxt == sc.virama and last + 2 >= n

            if voice and not cluster and (nxt != sc.virama or final_schwa_here):
                if prev_was_nasal_cluster and cons in sc.nasal_voicing:
                    cons = sc.nasal_voicing[cons]
                elif (cons in sc.voicing and out and out[-1][-1:] in _VOWEL_CHARS):
                    cons = sc.voicing[cons]

            out.append(cons)
            i = last
            if not cluster and ch in sc.chillu:
                prev_was_nasal_cluster = cons in ML_NASALS
                i += 1
                continue
            if nxt == sc.virama:
                # Virama kills the inherent vowel. Word-finally, Malayalam
                # still voices a half-u rather than ending on a bare consonant.
                if final_schwa_here:
                    out.append(sc.final_schwa)
                    prev_was_nasal_cluster = False
                else:
                    prev_was_nasal_cluster = cons in ML_NASALS
                i += 2
                continue
            prev_was_nasal_cluster = False
            if nxt in sc.vowel_signs:
                out.append(sc.vowel_signs[nxt])
                i += 2
                continue
            out.append("a")     # inherent vowel
            i += 1
            continue

        prev_was_nasal_cluster = False
        if ch in sc.vowels:
            out.append(sc.vowels[ch])
        elif ch in sc.vowel_signs:
            out.append(sc.vowel_signs[ch])   # orphaned sign; emit bare
        elif ch in sc.marks:
            out.append(sc.marks[ch])
        elif ch in sc.digits:
            out.append(sc.digits[ch])
        elif ch == sc.virama:
            pass
        else:
            out.append(ch)      # Latin, punctuation, digits pass through
        i += 1
    word_out = _GEMINATE_NASAL.sub(r"\1", "".join(out))
    return _NASAL_VOICED.sub("ng", word_out)


_TOKEN = re.compile(r"(\s+)")


def transliterate(text: str, lang: str, voice: bool | None = None) -> str:
    """Romanize `text` from `lang` ('ml'|'ta') into the canonical L2 spelling."""
    if lang not in SCRIPTS:
        raise ValueError(f"lang must be one of {sorted(SCRIPTS)}, got {lang!r}")
    sc = SCRIPTS[lang]
    if voice is None:
        voice = bool(sc.voicing)
    text = unicodedata.normalize("NFC", text).translate(ZERO_WIDTH)
    return "".join(
        part if part.isspace() else _transliterate_word(part, sc, voice)
        for part in _TOKEN.split(text)
    )
