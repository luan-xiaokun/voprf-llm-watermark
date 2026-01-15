import argparse

from watermark_suite.schemes.upv.generate_data import generate_model_key_data
from watermark_suite.schemes.upv.model_key import train_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bit_number", type=int, default=18)
    parser.add_argument("--window_size", type=int, default=4)
    parser.add_argument("--sample_number", type=int, default=5000)
    parser.add_argument("--layers", type=int, default=5, help="bit number")
    parser.add_argument(
        "--output_file", type=str, default="train_generator_data/data_18_sample.jsonl"
    )
    parser.add_argument(
        "--model_dir", type=str, default="model", help="model directory"
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    print("Args:", args)
    print("Generating data...")
    generate_model_key_data(
        args.bit_number,
        args.sample_number,
        args.output_file,
        args.window_size,
    )
    print("Training model...")
    train_model(
        args.output_file,
        args.bit_number,
        args.model_dir,
        args.window_size,
        args.layers,
    )


if __name__ == "__main__":
    main()
