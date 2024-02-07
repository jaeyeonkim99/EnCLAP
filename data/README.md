# Data Preprocessing for training EnCLAP

## 1. Make EnCodec Embeddings
```
python infer_encodec.py --data_path data/wavs --save_path data/encodec_embeddings
```

## 2. Make CLAP Embeddings
```
python infer_clap.py --data_path data/wavs --save_path data/clap_embeddings --clap_ckpt clap_ckpt_path
```
You may give different options to `--enable_fusion` and `--audio_encoder` flags based on the CLAP checkpoint you are using.

## 3. Make CSV files
You should make CSV files corresponding to train, validation, and test dataset. Examples are given in [csv](../csv/) for AudioCaps and Clotho datasets.
- Train csv files should have columns `file_path` and `caption`. If an audio file is labeled with multiple captions, they should be made listed in separate entries.
- Validation and Test csv files should have columns `file_path`,`caption_1`, `caption_2`,...., `caption_n` if `n` captions are given for a single audio file. 
