import copy
import json
import os
import random

from tqdm import tqdm


def int_to_bin_list(n, length=8):
    bin_str = format(n, "b").zfill(length)
    return [int(b) for b in bin_str]


def max_number(bits):
    return (1 << bits) - 1


def generate_model_key_data(bit_number, sample_number, output_file, window_size):
    # We need to generate pairs for all numbers from 0 to 15 (inclusive)
    numbers = list(range(1, max_number(bit_number)))

    # process the pairs and assign balanced labels
    data = []
    combined_set = set()  # Use a set to track unique combined data
    # Iterating over all pairs of numbers
    for _ in tqdm(range(sample_number)):
        # create a list of labels for each num, half 0s and half 1s
        labels = [0, 1]
        random.shuffle(labels)
        combined = []
        # Loop over window size
        for _ in range(window_size - 1):
            # random pick number from numbers and ensure unique
            num = random.choice(numbers)
            bin_num = int_to_bin_list(num, bit_number)
            combined.append(bin_num)

        for label in labels:
            combined1 = copy.deepcopy(combined)
            num = random.choice(numbers)
            bin_num = int_to_bin_list(num, bit_number)
            # import ipdb; ipdb.set_trace()
            combined1.append(bin_num)
            # assign the label
            data.append({"data": combined1, "label": label})

    # save to jsonl
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for entry in data:
            f.write(json.dumps(entry))
            f.write("\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--bit_number", type=int, default=8)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--sample_number", type=int, default=50)
    parser.add_argument(
        "--output_file", type=str, default="train_generator_data/data_8_sample.jsonl"
    )
    args = parser.parse_args()
    generate_model_key_data(
        args.bit_number, args.sample_number, args.output_file, args.window_size
    )
