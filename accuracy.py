#!/usr/bin/env python3
import json
import re
import sys
from collections import Counter

pat = re.compile(r"<answer>([\w\s]+)</answer>")
pat2 = re.compile(r"<\|im_start\|>assistant\n\s*([ABCDE])", re.DOTALL|re.MULTILINE|re.S)

def id_to_duration(i:int):
    if i < 900:
        return "short"
    elif i < 1800:
        return "medium"
    else:
        return "long"

def main(infile, print_=False):
    # correct, total = 0, 0
    correct = {"short": 0, "medium": 0, "long": 0}
    total = {"short": 0, "medium": 0, "long": 0}
    visual_token_counts = {"short": 0, "medium": 0, "long": 0}
    token_counts = {"short": 0, "medium": 0, "long": 0}
    num_rounds = 0
    answer_not_found = 0
    answer_format2 = 0
    c = Counter()

    with open(infile, "r") as f:

        for i, line in enumerate(f):
            duration = id_to_duration(i)

            item = json.loads(line)
            total[duration] += 1

            try:
                output = item["output"]
            except:
                # print(i)
                pass
            c.update([item["stop_reason"]])

            if "answer" in item:
                answer = item["answer"].strip().upper()
            else:
                m = pat.search(output)
                if not m:
                    # print("Error: No answer found in output:", item)
                    answer_not_found += 1
                    continue
                    # m = pat2.search(output)
                    # if not m:
                    #     print(output[-100:])
                    #     continue
                    # c['answer_found'] += 1
                    # c[item['stop_reason']] -= 1
                    # answer_format2 += 1

                answer = m.group(1).strip().upper()

            gt = item["gt"].strip().upper()
            # print(f"line: {i}, GT: {gt}, Pred: {answer}")

            if answer == gt:
                correct[duration] += 1
            else:
                pass
                # print(f"line: {i}, GT: {gt}, Pred: {answer}, Output: {output}")

            visual_token_counts[duration] += item.get("num_img_tokens", 0)
            token_counts[duration] += item.get("total_tokens", 0)
            num_rounds += item.get("num_round", item.get("num_rounds", 0))
    print(answer_format2)
    avg_accuracy = {k: correct[k] / total[k] if total[k] > 0 else 0 for k in correct}
    total_accuracy = sum(correct.values()) / sum(total.values()) if sum(total.values()) > 0 else 0
    total_avg_visual_token = sum(visual_token_counts.values()) / sum(total.values()) if sum(total.values()) > 0 else 0
    total_avg_token = sum(token_counts.values()) / sum(total.values()) if sum(total.values()) > 0 else 0
    avg_num_rounds = num_rounds / sum(total.values()) if sum(total.values()) > 0 else 0
    if print_:
        print(f"Correct: {correct}, Total: {total}")
        print(f"Average Accuracy: {avg_accuracy}")
        print(f"Total Accuracy: {total_accuracy:.3%}")
        print(f"Average Visual Token: {total_avg_visual_token:.0f}")
        print(f"Average Token: {total_avg_token:.0f}")
        print(f"Average Number of Rounds: {avg_num_rounds:.2f}")
        print("Stop reason counts:", c)
        print(f"Failed rate: {(sum(total.values()) - c['answer_found'])}/{sum(total.values())}")
        print(f"Answer not found {answer_not_found}")

    return total_accuracy, sum(total.values())


if __name__ == "__main__":
    infile = sys.argv[1]
    main(infile, print_=True)
