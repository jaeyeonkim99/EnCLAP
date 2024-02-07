import argparse
from typing import Tuple

import gradio as gr
import numpy as np
import torch

from inference import EnClap


def input_toggle(choice: str):
    if choice == "file":
        return gr.update(visible=True), gr.update(visible=False)
    return gr.update(visible=False), gr.update(visible=True)


if __name__ == "__main__":
    import logging
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", "-c", type=str)
    parser.add_argument("--clap_ckpt", '-cl', type=str)
    parser.add_argument("--device", "-d", type=str, choices=["cpu", "cuda"])
    args = parser.parse_args()

    enclap = EnClap(ckpt_path=args.ckpt, clap_ckpt_path=args.clap_ckpt, device=args.device)

    
    def run_enclap(
        input_type: str,
        file_input: Tuple[int, np.ndarray],
        mic_input: Tuple[int, np.ndarray],
        seed: int,
    ) -> str:
        print(input_type, file_input, mic_input)
        input = file_input if input_type == "file" else mic_input
        if input is None:
            raise gr.Error("Input audio was not provided.")
        res, audio = input
        torch.manual_seed(seed)
        return enclap.infer_from_audio(torch.from_numpy(audio), res)[0]

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                radio = gr.Radio(
                    ["file", "mic"],
                    value="file",
                    label="Choose the input method of the audio.",
                )
                file = gr.Audio(label="Input", visible=True)
                mic = gr.Mic(label="Input", visible=False)
                slider = gr.Slider(minimum=0, maximum=100, label="Seed")
                radio.change(fn=input_toggle, inputs=radio, outputs=[file, mic])
                button = gr.Button("Run", label="run")
            with gr.Column():
                output = gr.Text(label="Output")
            button.click(
                fn=run_enclap, inputs=[radio, file, mic, slider], outputs=output
            )

    demo.launch(share=True)
