#!/usr/bin/python
import math
import re
import sys
import xml.sax.saxutils

"""
This script was adapted from the original version by hieuhoang1972 which is part of MOSES. 
"""

# $Id: bleu.py 1307 2007-03-14 22:22:36Z hieuhoang1972 $

"""Provides:

cook_refs(refs, n=4): Transform a list of reference sentences as strings into a form usable by cook_test().
cook_test(test, refs, n=4): Transform a test sentence as a string (together with the cooked reference sentences) into 
a form usable by score_cooked().
score_cooked(alltest, n=4): Score a list of cooked test sentences.

score_set(s, testid, refids, n=4): Interface with dataset.py; calculate BLEU score of testid against refids.

The reason for breaking the BLEU computation into three phases cook_refs(), cook_test(), and score_cooked() is to allow
the caller to calculate BLEU scores for multiple test sets as efficiently as possible.
"""

# Added to bypass NIST-style pre-processing of hyp and ref files -- wade
nonorm = 0

preserve_case = False
eff_ref_len = "shortest"

normalize1 = [
    ("<skipped>", ""),  # strip "skipped" tags
    (r"-\n", ""),  # strip end-of-line hyphenation and join lines
    (r"\n", " "),  # join lines
    #    (r'(\d)\s+(?=\d)', r'\1'), # join digits
]
normalize1_regex = [(re.compile(pattern), replace) for (pattern, replace) in normalize1]

normalize2 = [
    (
        r"([\{-\~\[-\` -\&\(-\+\:-\@\/])",
        r" \1 ",
    ),  # tokenize punctuation. apostrophe is missing
    (
        r"([^0-9])([\.,])",
        r"\1 \2 ",
    ),  # tokenize period and comma unless preceded by a digit
    (
        r"([\.,])([^0-9])",
        r" \1 \2",
    ),  # tokenize period and comma unless followed by a digit
    (r"([0-9])(-)", r"\1 \2 "),  # tokenize dash when preceded by a digit
]
normalize2_regex = [(re.compile(pattern), replace) for (pattern, replace) in normalize2]


def normalize(s):
    """Normalize and tokenize text. This is lifted from NIST mteval-v11a.pl."""
    # Added to bypass NIST-style pre-processing of hyp and ref files -- wade
    if nonorm:
        return s.split()
    if type(s) is not str:
        s = " ".join(s)
    # language-independent part:
    for pattern, replace in normalize1:
        s = re.sub(pattern, replace, s)
    s = xml.sax.saxutils.unescape(s, {"&quot;": '"'})
    # language-dependent part (assuming Western languages):
    s = " %s " % s
    if not preserve_case:
        s = s.lower()  # this might not be identical to the original
    for pattern, replace in normalize2:
        s = re.sub(pattern, replace, s)
    return s.split()


def count_ngrams(words, n=4):
    counts = {}
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i: i + k])
            counts[ngram] = counts.get(ngram, 0) + 1
    return counts


def cook_refs(refs, n=4):
    """Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them."""

    refs = [normalize(ref) for ref in refs]
    max_counts = {}
    for ref in refs:
        counts = count_ngrams(ref, n)
        for ngram, count in counts.items():
            max_counts[ngram] = max(max_counts.get(ngram, 0), count)
    return [len(ref) for ref in refs], max_counts


def cook_test(test, item, n=4):
    """Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it."""
    (ref_lens, ref_max_counts) = item
    test = normalize(test)
    result = {"testlen": len(test)}

    # Calculate effective reference sentence length.

    if eff_ref_len == "shortest":
        result["reflen"] = min(ref_lens)
    elif eff_ref_len == "average":
        result["reflen"] = float(sum(ref_lens)) / len(ref_lens)
    elif eff_ref_len == "closest":
        min_diff = None
        for ref_len in ref_lens:
            if min_diff is None or abs(ref_len - len(test)) < min_diff:
                min_diff = abs(ref_len - len(test))
                result["reflen"] = ref_len

    result["guess"] = [max(len(test) - k + 1, 0) for k in range(1, n + 1)]

    result["correct"] = [0] * n
    counts = count_ngrams(test, n)
    for ngram, count in counts.items():
        result["correct"][len(ngram) - 1] += min(ref_max_counts.get(ngram, 0), count)

    return result


def score_cooked(all_comps, n=4, ground=0, smooth=1):
    total_comps = {"testlen": 0, "reflen": 0, "guess": [0] * n, "correct": [0] * n}
    for comps in all_comps:
        for key in ["testlen", "reflen"]:
            total_comps[key] += comps[key]
        for key in ["guess", "correct"]:
            for k in range(n):
                total_comps[key][k] += comps[key][k]
    log_bleu = 0.0
    all_bleus = []
    for k in range(n):
        correct = total_comps["correct"][k]
        guess = total_comps["guess"][k]
        add_smooth = 0
        if smooth == 1 and k > 0:
            add_smooth = 1
        log_bleu += math.log(correct + add_smooth + sys.float_info.min) - math.log(
            guess + add_smooth + sys.float_info.min
        )
        if guess == 0:
            all_bleus.append(-10000000)
        else:
            all_bleus.append(math.log(correct + sys.float_info.min) - math.log(guess))

    log_bleu /= float(n)
    all_bleus.insert(0, log_bleu)

    brev_penalty = min(
        0., 1 - float(total_comps["reflen"] + 1) / (total_comps["testlen"] + 1)
    )
    for i in range(len(all_bleus)):
        if i == 0:
            all_bleus[i] += brev_penalty
        all_bleus[i] = math.exp(all_bleus[i])
    return all_bleus


def bleu(refs, candidate, ground=0, smooth=1):
    refs = cook_refs(refs)
    test = cook_test(candidate, refs)
    return score_cooked([test], ground=ground, smooth=smooth)


def split_puncts(line):
    return " ".join(re.findall(r"\w+|[^\s\w]", line))


def compute_maps(predictions, gold_file):
    prediction_map = {}
    gold_map = {}
    gf = open(gold_file, "r")

    for row in predictions:
        cols = row.strip().split("\t")
        if len(cols) == 1:
            (rid, pred) = (cols[0], "")
        else:
            (rid, pred) = (cols[0], cols[1])
        prediction_map[rid] = [split_puncts(pred.strip().lower())]

    for row in gf:
        (rid, pred) = row.split("\t")
        if rid in prediction_map:  # Only insert if the id exists for the method
            if rid not in gold_map:
                gold_map[rid] = []
            gold_map[rid].append(split_puncts(pred.strip().lower()))

    sys.stderr.write("Total: " + str(len(gold_map)) + "\n")
    return gold_map, prediction_map


# m1 is the reference map
# m2 is the prediction map
def bleu_from_maps(m1, m2):
    score = [0] * 5
    num = 0.0

    for key in m1:
        if key in m2:
            bl = bleu(m1[key], m2[key][0])
            score = [score[i] + bl[i] for i in range(0, len(bl))]
            num += 1
    return [s * 100.0 / num for s in score]


if __name__ == "__main__":
    reference_file = sys.argv[1]
    predictions = []
    for row in sys.stdin:
        predictions.append(row)
    (gold_map, prediction_map) = compute_maps(predictions, reference_file)
    print(bleu_from_maps(gold_map, prediction_map)[0])
