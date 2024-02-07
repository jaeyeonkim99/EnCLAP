from typing import Any, Dict

import argparse
import numpy as np
import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
from laion_clap import CLAP_Module
from transformers import AutoTokenizer

from modeling.enclap_bart import EnClapBartConfig, EnClapBartForConditionalGeneration


class EnClap:
    def __init__(
        self,
        ckpt_path: str,
        clap_audio_model: str = "HTSAT-tiny",
        clap_enable_fusion: bool = True,
        clap_ckpt_path: str = None,
        device: str = "cuda",
    ):
        config = EnClapBartConfig.from_pretrained(ckpt_path)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
        self.model = (
            EnClapBartForConditionalGeneration.from_pretrained(ckpt_path)
            .to(self.device)
            .eval()
        )

        self.encodec = EncodecModel.encodec_model_24khz().to(self.device)
        self.encodec.set_target_bandwidth(12.0)
        self.clap_model = CLAP_Module(enable_fusion=clap_enable_fusion, amodel=clap_audio_model, device=self.device)
        self.clap_model.load_ckpt(clap_ckpt_path)

        self.generation_config = {
            "_from_model_config": True,
            "bos_token_id": 0,
            "decoder_start_token_id": 2,
            "early_stopping": True,
            "eos_token_id": 2,
            "forced_bos_token_id": 0,
            "forced_eos_token_id": 2,
            "no_repeat_ngram_size": 3,
            "num_beams": 4,
            "pad_token_id": 1,
            "max_length": 50,
        }
        self.max_seq_len = config.max_position_embeddings - 3

    @torch.no_grad()
    def infer_from_audio_file(
        self, audio_file: str, generation_config: Dict[str, Any] = None
    ) -> str:
        if generation_config is None:
            generation_config = self.generation_config
        audio, res = torchaudio.load(audio_file)
        return self.infer_from_audio(audio[0], res)

    @torch.no_grad()
    def infer_from_audio(
        self, audio: torch.Tensor, res: int, generation_config: Dict[str, Any] = None
    ) -> str:
        if generation_config is None:
            generation_config = self.generation_config
        if audio.dtype == torch.short:
            audio = audio / 2**15
        if audio.dtype == torch.int:
            audio = audio / 2**31
        encodec_audio = (
            convert_audio(
                audio.unsqueeze(0), res, self.encodec.sample_rate, self.encodec.channels
            )
            .unsqueeze(0)
            .to(self.device)
        )
        encodec_frames = self.encodec.encode(encodec_audio)
        encodec_frames = torch.cat(
            [codebook for codebook, _ in encodec_frames], dim=-1
        ).mT

        clap_audio = torchaudio.transforms.Resample(res, 48000)(audio).unsqueeze(0)
        clap_embedding = self.clap_model.get_audio_embedding_from_data(clap_audio, use_tensor=True)

        return self._infer(encodec_frames, clap_embedding, generation_config)

    @torch.no_grad()
    def _infer(
        self,
        encodec_frames: torch.LongTensor,
        clap_embedding: torch.Tensor,
        generation_config: Dict[str, Any] = None,
    ) -> str:
        input_ids = torch.cat(
            [
                torch.ones(
                    (encodec_frames.shape[0], 2, encodec_frames.shape[-1]),
                    dtype=torch.long,
                ).to(self.device)
                * self.tokenizer.bos_token_id,
                encodec_frames[:, : self.max_seq_len],
                torch.ones(
                    (encodec_frames.shape[0], 1, encodec_frames.shape[-1]),
                    dtype=torch.long,
                ).to(self.device)
                * self.tokenizer.eos_token_id,
            ],
            dim=1,
        )
        encodec_mask = torch.LongTensor(
            [[0, 0] + [1] * (input_ids.shape[1] - 3) + [0]]
        ).to(self.device)

        enclap_bart_inputs = {
            "input_ids": input_ids,
            "encodec_mask": encodec_mask,
            "clap_embedding": clap_embedding,
        }

        results = self.model.generate(**enclap_bart_inputs, **generation_config)
        caption = self.tokenizer.batch_decode(results, skip_special_tokens=True)

        return caption

    @torch.no_grad()
    def infer_from_encodec(
        self,
        encodec_path,
        clap_path,
        generation_config: Dict[str, Any] = None,
    ):
        if generation_config is None:
            generation_config = self.generation_config
        encodec_frames = torch.from_numpy(np.load(encodec_path)).unsqueeze(0).cuda()
        clap_embedding = torch.from_numpy(np.load(clap_path)).unsqueeze(0).cuda()

        return self._infer(encodec_frames, clap_embedding, generation_config)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", "-c", type=str)
    parser.add_argument("--clap_ckpt", '-cl', type=str)
    parser.add_argument("--input", "-i", type=str)
    args = parser.parse_args()

    print("> Loading Model...")
    enclap = EnClap(
        ckpt_path=args.ckpt, 
        clap_ckpt_path=args.clap_ckpt, 
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("> Running Inference...")
    prediction = enclap.infer_from_audio_file(args.input)[0]
    print("> Result: ", prediction)