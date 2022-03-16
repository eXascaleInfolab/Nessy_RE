import argparse
import copy
import json
import numpy as np
import os
from utils.data_loader import load_json


NEG = 'no_relation'


def add_random_noise(data, percent, label):
    noisy_data = copy.deepcopy(data)
    for instance in noisy_data:
        if np.random.rand() <= percent:
            if instance['relation'] == label:
                instance['relation'] = NEG
            else:
                instance['relation'] = label
    return noisy_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to input file")
    parser.add_argument("output", type=str, help="Path to output file")
    parser.add_argument("label", type=str, help="Positive label")
    parser.add_argument("--percent", type=float, default=0.4, help="How much noise")
    parser.add_argument("--seed", type=int, default=31, help="Random seed")
    args = parser.parse_args()
    np.random.seed(args.seed)
    ori_data = load_json(args.input)
    print(f"{len(ori_data)} instances in original data.")
    noisy_data = add_random_noise(ori_data, args.percent, args.label)
    count = 0
    for d_ori, d_noisy in zip(ori_data, noisy_data):
        if d_ori['relation'] != d_noisy['relation']:
            count += 1
    print(f"{count} instances became noisy which is {count/len(ori_data):.2%} of data")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w+") as f:
        json.dump(noisy_data, f)


if __name__ == "__main__":
    main()
