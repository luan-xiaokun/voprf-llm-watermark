import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ..detector import DetectionCost, DetectionResult, WatermarkDetector
from .generate_data import int_to_bin_list
from .model_key import BinaryClassifier, SubNet
from .watermarking import get_detector_model


@dataclass
class UPVDetectionResult(DetectionResult):
    confidence: float
    predicted: bool
    green_token_mask: list[bool] | None = None


class UPVDetectionCost(DetectionCost):
    pass


class TransformerClassifier(nn.Module):
    def __init__(
        self, bit_number, b_layers, input_dim, hidden_dim, num_classes=1, num_layers=2
    ):
        super(TransformerClassifier, self).__init__()
        self.binary_classifier = SubNet(bit_number, b_layers)
        self.classifier = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, return_sequence=False):
        batch_size, seq_len, _ = x.size()
        x1 = x.view(batch_size * seq_len, -1)
        features = self.binary_classifier(x1)
        features = features.view(batch_size, seq_len, -1)
        output, _ = self.classifier(features)

        if return_sequence:
            output = self.fc_hidden(output)
            output = self.sigmoid(output)
            output = self.fc(output)
            output = self.sigmoid(output)
            return output
        else:
            output = self.fc_hidden(output[:, -1, :])
            output = self.sigmoid(output)
            output = self.fc(output)
            output = self.sigmoid(output)
            return output


class Seq2SeqDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_detector_model(bits_num, b_layers, model_dir):
    model = TransformerClassifier(bits_num, b_layers, 64, 128)
    if model_dir is not None:
        state_dict = torch.load(model_dir)
        print("Loaded state dict from", model_dir)
        model.load_state_dict(state_dict)
    return model


class UPVDetector(WatermarkDetector):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        private_detector_dir: str | Path,
        window_size: int,
        bits_num: int,
        gamma: float = 0.5,
    ):
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.bits_num = bits_num
        self.gamma = gamma
        self.private_detector_dir = Path(private_detector_dir)
        self.provider_detector_model = get_detector_model(
            bits_num,
            5,  # hardcode number of layers
            self.private_detector_dir / "private_detector.pt",
        )
        self.provider_detector_model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.provider_detector_model.to(self.device)

    def detect(
        self,
        text: str,
        token_num: int | None = None,
        step_size: int | None = None,
        return_green_token_mask: bool = False,
        **kwargs,
    ) -> UPVDetectionResult:
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
        if token_num is None:
            token_num = inputs["input_ids"].shape[1]
        inputs_ids = inputs["input_ids"].squeeze()[:token_num]
        inputs_bin = [int_to_bin_list(n, self.bits_num) for n in inputs_ids]

        inputs_tensor = torch.tensor(inputs_bin).unsqueeze(0)  # add batch dimension
        inputs_tensor = inputs_tensor.float().to(self.device)

        milestones = None
        step_p_values = None
        step_scores = None

        if step_size is not None and step_size > 0:
            outputs = self.provider_detector_model(inputs_tensor, return_sequence=True)
            outputs = outputs.squeeze(0).squeeze(-1)  # (seq_len,)

            milestones = list(range(step_size, token_num + 1, step_size))
            if milestones:
                indices = [m - 1 for m in milestones]
                selected_outputs = outputs[indices]
                step_scores = selected_outputs.tolist()
                step_p_values = [(1.0 - s) for s in step_scores]

            confidence = outputs[-1].item()
        else:
            outputs = self.provider_detector_model(inputs_tensor)
            outputs = outputs.reshape([-1])
            confidence = outputs.item()

        predicted = confidence > 0.5

        green_token_mask = None
        if return_green_token_mask:
            # Use the provider detector model with return_sequence=True to get per-token predictions
            with torch.no_grad():
                token_outputs = self.provider_detector_model(
                    inputs_tensor, return_sequence=True
                )
                token_outputs = token_outputs.squeeze(0).squeeze(-1)  # (seq_len,)
                token_predictions = (token_outputs > 0.5).bool().cpu().tolist()

            # Mask out the first window_size tokens (insufficient context for reliable detection)
            green_token_mask = [False] * self.window_size + token_predictions[
                self.window_size :
            ]

        return UPVDetectionResult(
            confidence=confidence,
            predicted=predicted,
            total_token_num=token_num,
            p_value=1 - confidence,
            step_size=step_size,
            milestones=milestones,
            step_p_values=step_p_values,
            step_scores=step_scores,
            green_token_mask=green_token_mask,
        )

    def batch_detect(
        self,
        texts: list[str],
        token_num: int | None = None,
        step_size: int | None = None,
    ) -> tuple[list[UPVDetectionResult], list[UPVDetectionCost]]:
        results = []
        costs = []
        for text in tqdm(texts, desc="UPV Detection"):
            start_time = time.time()
            detection_result = self.detect(
                text,
                token_num=token_num,
                step_size=step_size,
            )
            end_time = time.time()
            detection_cost = UPVDetectionCost(
                token_num=detection_result.total_token_num,
                total_time=end_time - start_time,
            )
            results.append(detection_result)
            costs.append(detection_cost)
        return results, costs


def prepare_data(
    filepath, tokenizer, train_or_test="train", bits_num=18, z_value_threshold=4.0
):
    data = []
    if train_or_test == "train":
        with open(filepath, "r") as f:
            for line in f:
                json_obj = json.loads(line)
                inputs = json_obj["Input"]
                output = json_obj["Output"]
                label = 1 if output > z_value_threshold else 0  # binary classification

                inputs_bin = [int_to_bin_list(n, bits_num) for n in inputs]

                data.append(
                    (torch.tensor(inputs_bin), torch.tensor(label))
                )  # label is a scalar
    else:
        with open(filepath, "r") as f:
            for line in f:
                json_obj = json.loads(line)
                inputs = json_obj["Input"]
                label = json_obj["Tag"]
                z_score = json_obj["Z-score"]

                inputs = tokenizer(inputs, return_tensors="pt", add_special_tokens=True)

                inputs_bin = [
                    int_to_bin_list(n, bits_num) for n in inputs["input_ids"].squeeze()
                ]

                data.append(
                    (
                        torch.tensor(inputs_bin),
                        torch.tensor(label),
                        torch.tensor(z_score),
                    )
                )  # label is a scalar

    return data


def pad_sequence_to_fixed_length(inputs, target_length, padding_value=0):
    padded_inputs = torch.nn.utils.rnn.pad_sequence(
        inputs, batch_first=True, padding_value=padding_value
    )

    original_length = padded_inputs.shape[1]

    if original_length < target_length:
        # If the original sequence is shorter than the target length, we need to further pad the sequences
        pad_size = (0, 0, 0, target_length - original_length)
        padded_inputs = F.pad(padded_inputs, pad_size, value=padding_value)
    elif original_length > target_length:
        # If the original sequence is longer than the target length, we need to truncate the sequences
        padded_inputs = padded_inputs[:, :target_length, :]
    else:
        # If the original sequence is the same as the target length, just return the original inputs
        padded_inputs = padded_inputs

    return padded_inputs


def train_collate_fn(batch):
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    inputs_padded = pad_sequence_to_fixed_length(inputs, 200)

    return inputs_padded, torch.stack(targets)


def test_collate_fn(batch):
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    z_score = [item[2] for item in batch]

    inputs_padded = pad_sequence_to_fixed_length(inputs, 200)

    return inputs_padded, torch.stack(targets), torch.stack(z_score)


def train_private_detector_model(
    tokenizer,
    bits_num,
    data_dir,
    model_dir,
    output_model_dir,
    b_layers,
    z_value_threshold,
):
    # Prepare data
    data_dir = Path(data_dir)

    train_data = prepare_data(
        data_dir / "train_data.jsonl",
        tokenizer,
        train_or_test="train",
        bits_num=bits_num,
        z_value_threshold=z_value_threshold,
    )
    test_data = prepare_data(
        data_dir / "test_data.jsonl",
        tokenizer,
        train_or_test="test",
        bits_num=bits_num,
        z_value_threshold=z_value_threshold,
    )

    train_dataset = Seq2SeqDataset(train_data)
    test_dataset = Seq2SeqDataset(test_data)

    train_dataloader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, collate_fn=train_collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=32, collate_fn=test_collate_fn
    )

    # Initialize model and optimizer
    model_dir = Path(model_dir)
    pretrained_dict = torch.load(model_dir / "sub_net.pt")
    model = TransformerClassifier(bits_num, b_layers, 64, 128)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model_dict = model.binary_classifier.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.binary_classifier.load_state_dict(model_dict, strict=True)
    for param in model.binary_classifier.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    # Define the loss function
    loss_fn = torch.nn.BCELoss()

    print("private detector:")
    # save the average acc, tpr, fpr, tnr, fnr of the last 5 epochs
    acc_avg, tpr_avg, fpr_avg, tnr_avg, fnr_avg, f1_avg = 0, 0, 0, 0, 0, 0
    # Train and evaluate
    epochs = 80
    for epoch in range(epochs):
        model.train()
        train_losses = []
        correct = 0
        total = 0
        for inputs, targets in train_dataloader:
            targets = targets.cuda()
            optimizer.zero_grad()
            outputs = model((inputs.float()).cuda())
            outputs = outputs.reshape([-1])
            loss = loss_fn(outputs, (targets.float()))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            # calculate accuracy
            predicted = (outputs.data > 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        train_accuracy = 100 * correct / total

        model.eval()
        test_losses = []
        correct, total, tp, fp, fn, tn = 0, 0, 0, 0, 0, 0
        with torch.no_grad():
            for inputs, targets, z_score in test_dataloader:
                outputs = model((inputs.float()).cuda()).cuda()
                targets = targets.cuda()
                outputs = outputs.reshape([-1])
                loss = loss_fn(outputs, targets.float())
                test_losses.append(loss.item())

                # calculate acc, tp, fp, fn, tn, f1
                predicted = (outputs.data > 0.5).int()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                tp += (predicted & targets).sum().item()
                fp += (predicted & (~(targets.bool()))).sum().item()
                fn += ((~predicted) & targets).sum().item()
                tn += ((~predicted) & (~(targets.bool()))).sum().item()

        test_accuracy = 100 * correct / total
        test_tpr = 100 * tp / (tp + fn)
        test_fpr = 100 * fp / (fp + tn)
        test_tnr = 100 * tn / (fp + tn)
        test_fnr = 100 * fn / (tp + fn)
        test_f1 = 100 * 2 * tp / (2 * tp + fn + fp)

        print(
            f"Epoch: {epoch}, Train Loss: {sum(train_losses) / len(train_losses)}, Train Accuracy: {train_accuracy}%, Test Loss: {sum(test_losses) / len(test_losses)}, Test Accuracy: {test_accuracy}%, Test TPR: {test_tpr}%, Test FPR: {test_fpr}%, Test TNR: {test_tnr}%, Test FNR: {test_fnr}%, Test F1: {test_f1}%"
        )

        # calculate the average acc, tpr, fpr, tnr, fnr, f1 of the last 5 epochs
        if epochs - 5 <= epoch < epochs:
            acc_avg += test_accuracy
            tpr_avg += test_tpr
            fpr_avg += test_fpr
            tnr_avg += test_tnr
            fnr_avg += test_fnr
            f1_avg += test_f1

    acc_avg /= 5
    tpr_avg /= 5
    fpr_avg /= 5
    tnr_avg /= 5
    fnr_avg /= 5
    f1_avg /= 5

    output_model_dir = Path(output_model_dir)
    output_model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_model_dir / "private_detector.pt")
    print(
        f"Test Accuracy: {acc_avg}%, Test TPR: {tpr_avg}%, Test FPR: {fpr_avg}%, Test TNR: {tnr_avg}%, Test FNR: {fnr_avg}%, Test F1: {f1_avg}%"
    )

    print("public detector:")
    corr_num, tot_num, tp, fp, fn, tn = 0, 0, 0, 0, 0, 0
    with open(Path(data_dir) / "test_data.jsonl", "r") as f:
        for line in f:
            tot_num += 1
            json_obj = json.loads(line)
            label = json_obj["Tag"]
            z_score = json_obj["Z-score"]
            predicted = z_score > z_value_threshold
            if predicted == label:
                corr_num += 1
            if predicted == 1 and label == 1:
                tp += 1
            if predicted == 1 and label == 0:
                fp += 1
            if predicted == 0 and label == 1:
                fn += 1
            if predicted == 0 and label == 0:
                tn += 1
    test_accuracy = 100 * corr_num / tot_num
    test_tpr = 100 * tp / (tp + fn)
    test_fpr = 100 * fp / (fp + tn)
    test_tnr = 100 * tn / (fp + tn)
    test_fnr = 100 * fn / (tp + fn)
    test_f1 = 100 * 2 * tp / (2 * tp + fn + fp)
    print(
        f"Test Accuracy: {test_accuracy}%, Test TPR: {test_tpr}%, Test FPR: {test_fpr}%, Test TNR: {test_tnr}%, Test FNR: {test_fnr}%, Test F1: {test_f1}%"
    )
