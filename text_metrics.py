"""Word error rate between a reference text and a transcript (used by the STT and TTS notebooks).

Both texts are normalised first: lower-case, punctuation removed, hyphens split, and spelled-out numbers turned into
digits ("twenty five" -> "25"), so formatting differences don't count as errors.
"""
import difflib
import re

UNITS = {w: i for i, w in enumerate('zero one two three four five six seven eight nine ten eleven twelve thirteen '
                                    'fourteen fifteen sixteen seventeen eighteen nineteen'.split())}
TENS = {w: 10 * i for i, w in enumerate('twenty thirty forty fifty sixty seventy eighty ninety'.split(), 2)}


def words_to_digits(words):
    out, num, active = [], 0, False
    for w in words:
        if w in UNITS:
            num, active = num + UNITS[w], True
        elif w in TENS:
            num, active = num + TENS[w], True
        elif w in ('hundred', 'thousand') and active:
            num *= 100 if w == 'hundred' else 1000
        elif w == 'and' and active:
            continue
        else:
            if active:
                out.append(str(num))
                num, active = 0, False
            out.append(w)
    if active:
        out.append(str(num))
    return out


def normalize(text):
    text = str(text).lower().replace('’', "'").replace('-', ' ')
    text = re.sub(r'\.(?!\d)', '', text)              # U.K. -> uk, keep decimals
    text = re.sub(r"[^a-z0-9' ]+", ' ', text)
    text = re.sub(r'(\d)([a-z])\b', r'\1 \2', text)   # 172n -> 172 n
    return words_to_digits(text.split())


def word_errors(reference, transcript):
    """(reference word count, substitutions + deletions + insertions) after normalising both texts."""
    ref, hyp = normalize(reference), normalize(transcript)
    errors = 0
    for op, a0, a1, b0, b1 in difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False).get_opcodes():
        if op != 'equal':
            errors += max(a1 - a0, b1 - b0)
    return len(ref), errors
