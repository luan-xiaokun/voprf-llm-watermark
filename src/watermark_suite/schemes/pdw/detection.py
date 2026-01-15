import pickle
import time

from petlib.pack import decode
from tqdm import tqdm

from ..detector import DetectionCost, DetectionResult, WatermarkDetector
from .detect import search_for_asymmetric_watermark

DEFAULT_PK_PATH = "data/pdw/pk"
DEFAULT_PARAMS_PATH = "data/pdw/params"


class PDWDetectionResult(DetectionResult):
    pass


class PDWDetector(WatermarkDetector):
    def __init__(
        self,
        pk_path: str = DEFAULT_PK_PATH,
        params_path: str = DEFAULT_PARAMS_PATH,
        signature_segment_length: int = 16,
        bit_size: int = 2,
        message_length: int = 8,
        max_planted_errors: int = 2,
        timing: bool = False,
    ):
        self.params_path = params_path
        self.signature_segment_length = signature_segment_length
        self.bit_size = bit_size
        self.message_length = message_length
        self.max_planted_errors = max_planted_errors
        self.timing = timing

        with open(pk_path, "rb") as f:
            self.pk = decode(pickle.load(f))

        with open(params_path, "rb") as g:
            G = decode(pickle.load(g))
            self.params = (G, G.order(), G.gen1(), G.gen2(), G.pair)

    def batch_detect(
        self,
        texts: list[str],
        token_num: int | None = None,
    ) -> tuple[list[DetectionResult], list[DetectionCost]]:
        results = []
        costs = []
        for text in tqdm(texts, desc="PDW Detection"):
            start = time.perf_counter()
            is_watermarked = search_for_asymmetric_watermark(
                self.pk,
                self.params,
                text,
                self.message_length,
                self.signature_segment_length,
                self.bit_size,
                self.max_planted_errors,
            )
            end = time.perf_counter()
            costs.append(DetectionCost(token_num=1, total_time=end - start))
            results.append(
                PDWDetectionResult(
                    total_token_num=1,
                    p_value=0.0 if is_watermarked else 1.0,
                )
            )
        return results, costs
