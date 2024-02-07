import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path of the original wav files"
    )
    parser.add_argument(
        "--save_path", type=str, required=True, help="Path to save encodec .npy files"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(12.0)
    model = model.to(device)

    data_path = Path(args.data_path)
    save_path = Path(args.save_path)

    with torch.no_grad():
        for wav_path in tqdm(data_path.glob("**/*.wav")):
            wav, sr = torchaudio.load(wav_path)
            wav = convert_audio(wav, sr, model.sample_rate, model.channels)
            wav = wav.unsqueeze(0).to(device)
            encoded_frames = model.encode(wav)

            codes = torch.cat([codebook for codebook, _ in encoded_frames], dim=-1)
            codes = codes.cpu().squeeze(0).transpose(-1, -2).detach().numpy()

            out_path = save_path / wav_path.with_suffix(".npy").relative_to(data_path)
            out_path.parent.mkdir(exist_ok=True)
            np.save(out_path, codes)
