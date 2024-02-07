from dataclasses import dataclass
from pathlib import Path
from random import randint
from typing import Optional, Tuple

import numpy as np
import torch
from transformers import BartTokenizerFast


@dataclass
class Preprocessor:
    encodec_base_path: Path
    clap_base_path: Path
    tokenizer: BartTokenizerFast = BartTokenizerFast.from_pretrained(
        "facebook/bart-base"
    )
    max_length: int = 1024
    mcm_masking_prob: float = 0.15
    mcm_masking_span: int = 10
    label_pad_token_id: int = -100
    mask_token_id: int = 1024
    num_eval_captions: int = 5

    def __post_init__(self):
        if isinstance(self.encodec_base_path, str):
            self.encodec_base_path = Path(self.encodec_base_path)
        if isinstance(self.clap_base_path, str):
            self.clap_base_path = Path(self.clap_base_path)
        if isinstance(self.tokenizer, str):
            self.tokenizer = BartTokenizerFast.from_pretrained(self.tokenizer)

    def preprocess_train(self, example):
        path = example["file_path"]
        encodec = np.load(self.encodec_base_path / path)
        clap_embedding = np.load(self.clap_base_path / path)
        encodec_mask = np.array(
            [0, 0] + [1] * min(encodec.shape[0], self.max_length - 3) + [0]
        )
        attention_mask = np.ones(min(encodec.shape[0] + 3, self.max_length)).astype(
            np.int64
        )
        target_text = self.tokenizer(text_target=example["caption"])

        if encodec.shape[0] + 3 > self.max_length:
            start = randint(0, encodec.shape[0] - self.max_length + 3)
            encodec = encodec[start : start + self.max_length - 3]

        mcm_labels = None
        if self.mcm_masking_prob > 0:
            num_rvq = encodec.shape[-1]
            mcm_mask, _ = _compute_mask_indices(
                encodec.T.shape, self.mcm_masking_prob, self.mcm_masking_span
            )
            mcm_mask = mcm_mask.T
            mcm_labels = np.where(mcm_mask, encodec, self.label_pad_token_id)
            mcm_labels = np.concatenate(
                [
                    np.ones((2, num_rvq), dtype=np.int64) * self.label_pad_token_id,
                    mcm_labels,
                    np.ones((1, num_rvq), dtype=np.int64) * self.label_pad_token_id,
                ],
                axis=0,
            )
            encodec[mcm_mask] = self.mask_token_id

        encodec = np.concatenate(
            [
                np.ones((2, num_rvq), dtype=np.int64) * self.tokenizer.bos_token_id,
                encodec,
                np.ones((1, num_rvq), dtype=np.int64) * self.tokenizer.eos_token_id,
            ],
            axis=0,
        )

        return {
            "input_ids": encodec,
            "clap_embedding": clap_embedding,
            "encodec_mask": encodec_mask,
            "attention_mask": attention_mask,
            "mcm_labels": mcm_labels,
            "labels": target_text["input_ids"],
            "decoder_attention_mask": target_text["attention_mask"]
        }

    def preprocess_eval(self, example):
        path = example["file_path"]
        encodec = np.load(self.encodec_base_path / path)
        clap_embedding = np.load(self.clap_base_path / path)
        encodec_mask = np.array(
            [0, 0] + [1] * min(encodec.shape[0], self.max_length - 3) + [0]
        )
        attention_mask = np.ones(min(encodec.shape[0] + 3, self.max_length)).astype(
            np.int64
        )

        if encodec.shape[0] + 3 > self.max_length:
            encodec = encodec[: self.max_length - 3]

        num_rvq = encodec.shape[-1]
        encodec = np.concatenate(
            [
                np.ones((2, num_rvq), dtype=np.int64) * self.tokenizer.bos_token_id,
                encodec,
                np.ones((1, num_rvq), dtype=np.int64) * self.tokenizer.eos_token_id,
            ],
            axis=0,
        )

        captions = []
        for i in range(self.num_eval_captions):
            captions.append(example[f"caption_{i+1}"])

        return {
            "input_ids": encodec,
            "encodec_mask": encodec_mask,
            "attention_mask": attention_mask,
            "clap_embedding": clap_embedding,
            "captions": captions,
        }


def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.LongTensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape. Used to implement [SpecAugment: A Simple Data Augmentation Method for
    ASR](https://arxiv.org/abs/1904.08779). Note that this method is not optimized to run on TPU and should be run on
    CPU as part of the preprocessing during training.

    Args:
        shape: The shape for which to compute masks. This should be of a tuple of size 2 where
               the first element is the batch size and the second element is the length of the axis to span.
        mask_prob:  The percentage of the whole axis (between 0 and 1) which will be masked. The number of
                    independently generated mask spans of length `mask_length` is computed by
                    `mask_prob*shape[1]/mask_length`. Note that due to overlaps, `mask_prob` is an upper bound and the
                    actual percentage will be smaller.
        mask_length: size of the mask
        min_masks: minimum number of masked spans
        attention_mask: A (right-padded) attention mask which independently shortens the feature axis of
                        each batch dimension.
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon is used for probabilistic rounding
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # make sure num masked span <= sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # make sure num_masked span is also <= input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # compute number of masked spans in batch
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )

    # SpecAugment mask to fill
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []

    max_num_masked_span = compute_num_masked_span(sequence_length)

    if max_num_masked_span == 0:
        return spec_aug_mask

    for input_length in input_lengths:
        # compute num of masked spans for this input
        num_masked_span = compute_num_masked_span(input_length)

        # get random indices to mask
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # pick first sampled index that will serve as a dummy index to pad vector
        # to ensure same dimension for all batches due to probabilistic rounding
        # Picking first sample just pads those vectors twice.
        if len(spec_aug_mask_idx) == 0:
            # this case can only happen if `input_length` is strictly smaller then
            # `sequence_length` in which case the last token has to be a padding
            # token which we can use as a dummy mask id
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        spec_aug_mask_idx = np.concatenate(
            [
                spec_aug_mask_idx,
                np.ones(max_num_masked_span - num_masked_span, dtype=np.int32)
                * dummy_mask_idx,
            ]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(
        batch_size, max_num_masked_span * mask_length
    )

    # add offset to the starting indexes so that indexes now create a span
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(
        offsets, (batch_size, max_num_masked_span, mask_length)
    ).reshape(batch_size, max_num_masked_span * mask_length)
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # ensure that we cannot have indices larger than sequence_length
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = (
            sequence_length - 1
        )

    # scatter indices to mask
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return torch.from_numpy(spec_aug_mask), spec_aug_mask_idxs
