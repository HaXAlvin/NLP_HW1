"""
This script applies to IOB2 or IOBES tagging scheme.
If you are using a different scheme, please convert to IOB2 or IOBES.

IOB2:
- B = begin,
- I = inside but not the first,
- O = outside

e.g.
John   lives in New   York  City  .
B-PER  O     O  B-LOC I-LOC I-LOC O

IOBES:
- B = begin,
- E = end,
- S = singleton,
- I = inside but not the first or the last,
- O = outside

e.g.
John   lives in New   York  City  .
S-PER  O     O  B-LOC I-LOC E-LOC O

prefix: IOBES
chunk_type: PER, LOC, etc.
"""
from collections import defaultdict


def split_tag(chunk_tag):
    """
    split chunk tag into IOBES prefix and chunk_type
    e.g.
    B-PER -> (B, PER)
    O -> (O, None)
    """
    if chunk_tag == "O":
        return ("O", None)
    return chunk_tag.split("-", maxsplit=1)


def is_chunk_end(prev_tag, tag):
    """
    check if the previous chunk ended between the previous and current word
    e.g.
    (B-PER, I-PER) -> False
    (B-LOC, O)  -> True

    Note: in case of contradicting tags, e.g. (B-PER, I-LOC)
    this is considered as (B-PER, B-LOC)
    """
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)

    if prefix1 == "O":
        return False
    if prefix2 == "O":
        return prefix1 != "O"

    if chunk_type1 != chunk_type2:
        return True

    return prefix2 in ["B", "S"] or prefix1 in ["E", "S"]


def is_chunk_start(prev_tag, tag):
    """
    check if a new chunk started between the previous and current word
    """
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)

    if prefix2 == "O":
        return False
    if prefix1 == "O":
        return prefix2 != "O"

    if chunk_type1 != chunk_type2:
        return True

    return prefix2 in ["B", "S"] or prefix1 in ["E", "S"]


def calc_metrics(tp, p, t, percent=True):
    """
    compute overall precision, recall and FB1 (default values are 0.0)
    if percent is True, return 100 * original decimal value
    """
    precision = tp / p if p else 0
    recall = tp / t if t else 0
    fb1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    rate = 100 if percent else 1
    return rate * precision, rate * recall, rate * fb1


def count_chunks(true_seqs, pred_seqs):
    """
    true_seqs: a list of true tags
    pred_seqs: a list of predicted tags

    return:
    correct_chunks: a dict (counter),
                    key = chunk types,
                    value = number of correctly identified chunks per type
    true_chunks:    a dict, number of true chunks per type
    pred_chunks:    a dict, number of identified chunks per type

    correct_counts, true_counts, pred_counts: similar to above, but for tags
    """
    correct_chunks = defaultdict(int)
    true_chunks = defaultdict(int)
    pred_chunks = defaultdict(int)

    correct_counts = defaultdict(int)
    true_counts = defaultdict(int)
    pred_counts = defaultdict(int)

    prev_true_tag, prev_pred_tag = "O", "O"
    correct_chunk = None

    for true_tag, pred_tag in zip(true_seqs, pred_seqs):
        if true_tag == pred_tag:
            correct_counts[true_tag] += 1
        true_counts[true_tag] += 1
        pred_counts[pred_tag] += 1

        _, true_type = split_tag(true_tag)
        _, pred_type = split_tag(pred_tag)

        if correct_chunk is not None:
            true_end = is_chunk_end(prev_true_tag, true_tag)
            pred_end = is_chunk_end(prev_pred_tag, pred_tag)

            if pred_end and true_end:
                correct_chunks[correct_chunk] += 1
                correct_chunk = None
            elif pred_end != true_end or true_type != pred_type:
                correct_chunk = None

        true_start = is_chunk_start(prev_true_tag, true_tag)
        pred_start = is_chunk_start(prev_pred_tag, pred_tag)

        if true_start and pred_start and true_type == pred_type:
            correct_chunk = true_type
        if true_start:
            true_chunks[true_type] += 1
        if pred_start:
            pred_chunks[pred_type] += 1

        prev_true_tag, prev_pred_tag = true_tag, pred_tag
    if correct_chunk is not None:
        correct_chunks[correct_chunk] += 1

    return (correct_chunks, true_chunks, pred_chunks, correct_counts, true_counts, pred_counts)


def get_result(correct_chunks, true_chunks, pred_chunks, correct_counts, true_counts, pred_counts, verbose=True):
    """
    if verbose, print overall performance, as well as performance per chunk type;
    otherwise, simply return overall prec, rec, f1 scores
    """
    # sum counts
    sum_correct_chunks = sum(correct_chunks.values())
    sum_true_chunks = sum(true_chunks.values())
    sum_pred_chunks = sum(pred_chunks.values())

    sum_correct_counts = sum(correct_counts.values())
    sum_true_counts = sum(true_counts.values())

    nonO_correct_counts = sum(v for k, v in correct_counts.items() if k != "O")
    nonO_true_counts = sum(v for k, v in true_counts.items() if k != "O")

    chunk_types = sorted(set(list(true_chunks) + list(pred_chunks)))

    # compute overall precision, recall and FB1 (default values are 0.0)
    prec, rec, f1 = calc_metrics(sum_correct_chunks, sum_pred_chunks, sum_true_chunks)

    def output_result(prec, rec, f1):
        # print overall performance, and performance per chunk type

        print(f"processed {sum_true_counts} tokens with {sum_true_chunks} phrases; ", end="")
        print(f"found: {sum_pred_chunks} phrases; correct: {sum_correct_chunks}.\n", end="")

        print(f"accuracy: {100 * nonO_correct_counts / nonO_true_counts:6.2f}%; (non-O)")
        print(f"accuracy: {100 * sum_correct_counts / sum_true_counts:6.2f}%; ", end="")
        print(f"precision: {prec:6.2f}%; recall: {rec:6.2f}%; FB1: {f1:6.2f}")

        # for each chunk type, compute precision, recall and FB1 (default values are 0.0)
        for t in chunk_types:
            t_prec, t_rec, t_f1 = calc_metrics(correct_chunks[t], pred_chunks[t], true_chunks[t])
            print(f"{t:17s}: ", end="")
            print(f"precision: {t_prec:6.2f}%; recall: {t_rec:6.2f}; FB1: {t_f1:6.2f}", end="")
            print(f"  {pred_chunks[t]}")

    if verbose:
        output_result(prec, rec, f1)
    return prec, rec, f1, 100 * nonO_correct_counts / nonO_true_counts, 100 * sum_correct_counts / sum_true_counts


def evaluate(true_seqs, pred_seqs, verbose=True):
    return get_result(*count_chunks(true_seqs, pred_seqs), verbose=verbose)
