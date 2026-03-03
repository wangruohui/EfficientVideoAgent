#!/usr/bin/env python3
import json
import sys
from collections import Counter

def _require_number(item, key: str, lineno: int):
    if key not in item:
        raise ValueError(f"line {lineno}: missing required field `{key}`")
    value = item[key]
    if value is None:
        raise ValueError(f"line {lineno}: field `{key}` is None")
    if not isinstance(value, (int, float)):
        raise ValueError(f"line {lineno}: field `{key}` must be number, got {type(value).__name__}")
    return value


def main(infile, print_=False):
    eval_total = 0
    eval_correct = 0
    fail_empty_answer = 0
    round_sum = 0.0
    token_sum = 0.0
    reason_counter = Counter()

    with open(infile, "r") as f:
        for lineno, line in enumerate(f, start=1):
            item = json.loads(line)
            eval_total += 1

            reason = item.get("stop_reason", "missing_stop_reason")
            reason_counter[str(reason)] += 1

            # Strict mode: these fields must exist and be numeric.
            round_sum += _require_number(item, "num_rounds", lineno)
            token_sum += _require_number(item, "total_tokens", lineno)

            gt_s = str(item.get("gt", "")).strip().upper()
            ans_s = str(item.get("answer", "")).strip().upper()
            if ans_s == "":
                fail_empty_answer += 1
                continue

            if gt_s == ans_s:
                eval_correct += 1

    if eval_total == 0:
        raise ValueError(f"{infile} is empty")

    total_accuracy = eval_correct / eval_total
    avg_num_rounds = round_sum / eval_total
    avg_total_tokens = token_sum / eval_total
    if print_:
        print(f"Evaluating {infile}...")
        print("summary:")
        print(f"  total entries: {eval_total}")
        print(f"  accuracy: {eval_correct}/{eval_total} = {total_accuracy:.4%}")
        print(f"  failed (empty answer): {fail_empty_answer}")
        print(f"  avg rounds: {avg_num_rounds:.4f}")
        print(f"  avg total_tokens: {avg_total_tokens:.4f}")
        print("  stop_reason stats:")
        for reason, cnt in reason_counter.most_common():
            print(f"    {reason}: {cnt}")

    return total_accuracy, eval_total


if __name__ == "__main__":
    infile = sys.argv[1]
    main(infile, print_=True)
