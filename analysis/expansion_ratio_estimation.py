import json

import numpy as np


def load_data():
    costs_w4 = []
    with open("data/plot_data/detection_costs_w4.jsonl") as f:
        for line in f:
            costs_w4.append(json.loads(line))
    return costs_w4


def lstsq(costs: list):
    raw_text_bytes = np.array([r["raw_text_bytes"][0] for r in costs])
    total_communication_bytes = np.array(
        [r["total_communication_bytes"][0] for r in costs]
    )
    return np.linalg.lstsq(
        raw_text_bytes[:, None], total_communication_bytes, rcond=None
    )


def main():
    costs_w4 = load_data()
    print("W4:", lstsq(costs_w4))


if __name__ == "__main__":
    main()
