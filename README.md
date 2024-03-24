# EnCLAP: Combining Neural Audio Codec and Audio-Text Joint Embedding for Automated Audio Captioning [ICASSP 2024]

Jaeyeon Kim, Jaeyoon Jung, Jinjoo Lee, Sang Hoon Woo @ [MAUM AI Inc.](https://github.com/maum-ai), SNU, SSU

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/enclap-combining-neural-audio-codec-and-audio/audio-captioning-on-audiocaps)](https://paperswithcode.com/sota/audio-captioning-on-audiocaps?p=enclap-combining-neural-audio-codec-and-audio)
[![arXiv](https://img.shields.io/badge/arXiv-2401.17690-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2401.17690) [![Gradio](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/enclap-team/enclap) 

**Abstract** : We propose EnCLAP, a novel framework for automated audio captioning. EnCLAP employs two acoustic representation models, EnCodec and CLAP, along with a pretrained language model, BART. We also introduce a new training objective called masked codec modeling that improves acoustic awareness of the pretrained language model. Experimental results on AudioCaps and Clotho demonstrate that our model surpasses the performance of baseline models. Source code will be available at https://github.com/jaeyeonkim99/EnCLAP. An online demo is available at https://huggingface.co/spaces/enclap-team/enclap.

![Overall_procedure](./asset/figure.png)


# Requirements

## Environment
- We used `torch==1.13.0` and `python==3.9.12` for our experiment.
- Please install required depedencies through `pip install -r requirements.txt`.
- `wget`, `java`, `unzip` are required to use `aac_metrics` library. Please run `aac-metrics-download` after installing the library.
- `libsndfile1` is required to use `soundfile`.
- `torchaudio` version should match the version of `torch` and `cuda`. Check [here](https://download.pytorch.org/whl/torchaudio/) for appropriate version.

## CLAP and EnCodec
- In our experiments, we used CLAP checkpoint trained on `LAION-audio-630k` and `AudioSet`([630k-audioset-fusion-best.pt](https://huggingface.co/lukewys/laion_clap/blob/main/630k-audioset-fusion-best.pt)). See details of pretrained CLAP checkpoints in [LAION-AI/CLAP](https://github.com/LAION-AI/CLAP?tab=readme-ov-file).


# How to Use

## Training
1. **Preprocess the dataset**: For training, you must first convert the audio files into their respective CLAP embeddings and EnCodec sequences. Once you have the converted data, you must write CSV files mapping each example to its CLAP embedding and EnCodec sequence files. The example CSV files for AudioCaps and Clotho are provided at [csv](csv/).  See [data/README.md](data/README.md) for further details.
2. **Setup the training config file**: We included a number of training configuration files we used for our experiments at [cfg](cfg/). Feel free to modify them to fit your needs.
```
# Paths to save the experiment results 
output_dir: /output  

# Paths of the train csv file
train_file: csv/audiocaps/train.csv  

# Paths of the validation csv file
validation_file: csv/audiocaps/valid.csv  

# Root directory of the preprocessed EnCodec entries
encodec_base_path: /data/audiocaps/encodec 

# Root directory of the preprocessed CLAP embeddings
clap_base_path: /data/audiocaps/clap 
```

3. **Run the training script**: Our training script is based on [Accelerate](https://huggingface.co/docs/accelerate/index), so you may need to setup Accelerate before running the training script. You can designate the path to the training config by modifying `CFG_PATH` in the training script.

```
accelerate config    # Set up Accelerate
sh train.sh          # Run the training script
```

## Evaluate
- The evaluation dataset may be either preprocessed CLAP embeddings and EnCodec sequences or raw wav files. Similarly to training, you must provide a CSV file containing the dataset. See [csv](csv/) for examples.

```
# From preprocessed data
python evaluate.py --ckpt checkpoint_directory_with_config --clap_ckpt clap_ckpt_path --test_csv test.csv --from_preprocessed --encodec_path data/encodec_embeddings --clap_path data/clap_embeddings


# From wav files
python evaluate.py --ckpt checkpoint_directory_with_config --clap_ckpt clap_ckpt_path --test_csv test.csv --audio_path data/wav_files
```

- You can save the predictions by adding CSV path with `--save_path`. You can evaluate on other AAC metrics by modifying `metric_list` on the top of `evaluate.py`.
- To reproduce our results, we recommend you to evaluate several best checkpoints based on `valid/spider` score (we evalauted 5). The score reported during the training is not 100% accurate due to DDP and dataloader padding. 

## Inference
- You can infer from wav files using the following code.
```
python inference.py  --ckpt checkpoint_directory_with_config --clap_ckpt clap_ckpt_path --input input.wav
```

- You can also run the gradio demo with your own checkpoint.
```
python gradio_app.py --ckpt checkpoint_directory_with_config --clap_ckpt clap_ckpt_path --device cuda
```
# Pretrained Checkpoints
Pretrained checkpoints are available [here](https://drive.google.com/drive/folders/1JOcKyNOlKud0PY93ETGDlUJnWhmSC35m?usp=sharing).
- Each folder contains both the configuration file (`config.json`) and the weight file (`pytorch_model.bin`) required for inference.
- The folders `audiocaps`, `clotho`, `clotho_finetune`, and `both` denote the datasets on which each checkpoint is trained:
  - `clotho_finetune` contains checkpoints pretrained on AudioCaps and finetuned on Clotho.
  - `both` contains checkpoints trained on both Clotho and AudioCaps, used for the official gradio demo.
- The labels `base` and `large` indicate the checkpoints for EnClap-base and EnClap-large, respectively.

# Citation
```
@article{kim2024enclap,
  title={EnCLAP: Combining Neural Audio Codec and Audio-Text Joint Embedding for Automated Audio Captioning},
  author={Kim, Jaeyeon and Jung, Jaeyoon and Lee, Jinjoo and Woo, Sang Hoon},
  journal={arXiv preprint arXiv:2401.17690},
  year={2024}
}
```
```
@INPROCEEDINGS{10446672,
  author={Kim, Jaeyeon and Jung, Jaeyoon and Lee, Jinjoo and Woo, Sang Hoon},
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={EnCLAP: Combining Neural Audio Codec and Audio-Text Joint Embedding for Automated Audio Captioning}, 
  year={2024},
  volume={},
  number={},
  pages={6735-6739},
  keywords={Training;Codecs;Speech coding;Source coding;Signal processing;Acoustics;Task analysis;automated audio captioning;neural audio codec;audio-text joint embedding},
  doi={10.1109/ICASSP48485.2024.10446672}}
```