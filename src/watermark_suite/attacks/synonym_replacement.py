import random

import nltk
import torch
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline

try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("taggers/punkt_tab")
    nltk.data.find("taggers/averaged_perceptron_tagger")
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except Exception:
    print("Downloading required NLTK data... (punkt, averaged_perceptron_tagger)")
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    print("Download complete.")


class SynonymReplacer:
    """
    A class to replace words in a text with synonyms using a masked language model.
    Models are loaded upon instantiation.
    """

    def __init__(
        self, model_checkpoint: str = "distilbert-base-uncased", device: int = -1
    ):
        """
        Initializes the replacer and loads the necessary models.
        """
        print("Initializing SynonymReplacer and loading models...")

        self.model_max_length = 512
        self.safe_chunk_size = 400
        self.chunk_overlap = 50

        if device == -1:
            computed_device = 0 if torch.cuda.is_available() else -1
        else:
            computed_device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.unmasker = pipeline(
            "fill-mask",
            model=model_checkpoint,
            tokenizer=self.tokenizer,
            device=computed_device,
        )
        self.detokenizer = TreebankWordDetokenizer()

        print(f"Using device: {'GPU' if computed_device == 0 else 'CPU'}")
        print(
            f"Model max length: {self.model_max_length}, "
            f"safe chunk size: {self.safe_chunk_size} (in sub-word tokens)"
        )

    def _preprocess_and_chunk_text(self, long_text: str) -> list[str]:
        """
        (Helper method) Splits text into manageable, overlapping chunks.
        """
        sentences = nltk.sent_tokenize(long_text)
        processed_chunks = []

        for sentence in sentences:
            token_ids = self.tokenizer.encode(sentence, add_special_tokens=False)

            if len(token_ids) <= self.safe_chunk_size:
                processed_chunks.append(sentence)
            else:
                start = 0
                while start < len(token_ids):
                    end = start + self.safe_chunk_size
                    chunk_ids = token_ids[start:end]
                    processed_chunks.append(
                        self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
                    )
                    start += self.safe_chunk_size - self.chunk_overlap
        return processed_chunks

    def _batch_chunk_synonym_replacement(
        self,
        text_chunks: list[str],
        replacement_rate: float,
        top_k: int,
        batch_size: int = 16,
    ) -> list[str]:
        """
        (Helper method) Processes a list of text chunks to replace synonyms.
        """
        all_masked_sentences, mapping_info, output_tokens_list = [], [], []

        for i, chunk in enumerate(text_chunks):
            try:
                tokens = nltk.word_tokenize(chunk)
                output_tokens_list.append(tokens[:])
            except Exception:
                output_tokens_list.append([])
                continue

            content_word_tags = {
                "NN",
                "NNS",
                "NNP",
                "NNPS",
                "VB",
                "VBD",
                "VBG",
                "VBN",
                "VBP",
                "VBZ",
                "JJ",
                "JJR",
                "JJS",
                "RB",
                "RBR",
                "RBS",
            }
            pos_tags = nltk.pos_tag(tokens)
            candidate_indices = [
                k
                for k, (word, tag) in enumerate(pos_tags)
                if tag in content_word_tags and len(word) >= 3
            ]

            num_to_replace = int(len(candidate_indices) * replacement_rate)
            if num_to_replace == 0:
                continue

            indices_to_replace = sorted(
                random.sample(candidate_indices, num_to_replace)
            )

            for index in indices_to_replace:
                masked_tokens = tokens[:]
                masked_tokens[index] = self.tokenizer.mask_token
                all_masked_sentences.append(self.detokenizer.detokenize(masked_tokens))
                mapping_info.append((i, index, tokens[index]))

        if not all_masked_sentences:
            return [
                self.detokenizer.detokenize(tokens) for tokens in output_tokens_list
            ]

        all_predictions = self.unmasker(
            all_masked_sentences, top_k=top_k, batch_size=batch_size
        )

        for i, preds in enumerate(all_predictions):
            if not isinstance(preds, list):
                print(
                    f"\n[WARNING] Skipping prediction due to unexpected format: {preds}"
                )
                continue

            original_chunk_idx, token_idx, original_word = mapping_info[i]

            best_replacement = None
            for pred in preds:
                if (
                    pred["token_str"].lower() != original_word.lower()
                    and pred["token_str"].isalpha()
                ):
                    best_replacement = pred["token_str"]
                    break

            if best_replacement:
                if original_word.isupper():
                    best_replacement = best_replacement.upper()
                elif original_word.istitle():
                    best_replacement = best_replacement.capitalize()
                output_tokens_list[original_chunk_idx][token_idx] = best_replacement

        return [self.detokenizer.detokenize(tokens) for tokens in output_tokens_list]

    def replace_synonyms(
        self, texts: list[str], replacement_rate: float, top_k: int = 15
    ) -> list[str]:
        """
        Public method to perform synonym replacement on a list of texts.
        """
        results = []
        for text in tqdm(texts, desc="Synonym Replacement Processing"):
            safe_chunks = self._preprocess_and_chunk_text(text)
            modified_chunks = self._batch_chunk_synonym_replacement(
                safe_chunks, replacement_rate, top_k
            )
            final_text = " ".join(modified_chunks)
            results.append(final_text)
        return results
