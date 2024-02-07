import argparse
from pathlib import Path

import librosa
import numpy as np
import torch
from laion_clap import CLAP_Module
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        "-d",
        required=True,
        type=str,
        help="Path of the original wav files",
    )
    parser.add_argument(
        "--save_path",
        "-s",
        required=True,
        type=str,
        help="Path to save the clap audio embedding '.npy' files",
    )
    parser.add_argument(
        "--clap_ckpt",
        "-c",
        required=True,
        type=str,
        help="Path of the pretrained clap checkpoint",
    )
    parser.add_argument(
        "--enable_fusion",
        "-e",
        default=True,
        type=bool,
        help="Whether to enable the feature fusion of the clap model. Depends on the clap checkpoint you are using",
    )
    parser.add_argument(
        "--audio_encoder",
        "-a",
        default="HTSAT-tiny",
        type=str,
        help="Audio encoder of the clap model. Depends on the clap checkpoint you are using",
    )
    args = parser.parse_args()

    model = CLAP_Module(enable_fusion=args.enable_fusion, aencoder=args.audio_encoder)
    model.load_ckpt(args.clap_ckpt)
    data_path = Path(args.data_path)
    save_path = Path(args.save_path)

    with torch.no_grad():
        for wav_path in tqdm(
            data_path.glob("**/*.wav"), dynamic_ncols=True, colour="yellow"
        ):
            wav, _ = librosa.load(wav_path, sr=48000)

            clap_embeding = model.get_audio_embedding_from_data(
                x=wav[np.newaxis], use_tensor=False
            )
            clap_embeding = clap_embeding.squeeze(axis=0)

            out_path = save_path / wav_path.with_suffix(".npy").relative_to(data_path)
            out_path.parent.mkdir(exist_ok=True)
            np.save(out_path, clap_embeding)
