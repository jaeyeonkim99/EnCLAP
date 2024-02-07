from aac_metrics import evaluate
from inference import EnClap
from tqdm import tqdm
import os
import pandas as pd 
import argparse
import torch 
import csv

# You may change metrics to evaluate
metric_list = ["meteor", "spider"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", "-c", type=str, required=True)
    parser.add_argument("--clap_ckpt", '-cl', type=str, required=True)
    parser.add_argument('--test_csv', '-ts', required=True, type=str)
    parser.add_argument('--from_preprocessed', '-fp', action="store_true")
    parser.add_argument('--audio_path', '-ap', type=str, required=False)
    parser.add_argument('--encodec_path', '-ep', type=str, required=False)
    parser.add_argument('--clap_path', '-cp', type=str, required=False)
    parser.add_argument('--save_path', '-s', type=str, required=False)
    parser.add_argument('--num_captions', '-n', type=int, default=5)
    args = parser.parse_args()

    enclap = EnClap(
        ckpt_path=args.ckpt, 
        clap_ckpt_path=args.clap_ckpt, 
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    df = pd.read_csv(args.test_csv)

    print(f"> Making Predictions for model {args.ckpt}...")
    predictions = []
    references = []
    for idx in tqdm(range(len(df)), dynamic_ncols=True, colour="BLUE"):
        if args.from_preprocessed:
            wav_path = df.loc[idx]['file_path']
            encodec_path = os.path.join(args.encodec_path, wav_path)
            clap_path = os.path.join(args.clap_path, wav_path)
            prediction = enclap.infer_from_encodec(encodec_path, clap_path)[0]
        else:
            wav_path = df.loc[idx]['file_path'].replace(".npy", ".wav")
            wav_path = os.path.join(args.audio_path, wav_path)
            prediction = enclap.infer_from_audio_file(wav_path)[0]

        predictions.append(prediction)
        reference = []
        for i in range(1, args.num_captions+1):
            reference.append(df.loc[idx][f'caption_{i}'])        
        references.append(reference)

    print("> Evaluating predictions...")
    result = evaluate(predictions, references, metrics=metric_list)
    result = {k: round(v.item(),4) for k, v in result[0].items()}
    print(result)

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        with open(args.save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['file_path', 'prediction'])
            for idx in tqdm(range(len(df))):
                writer.writerow([df.loc[idx]['file_path'], predictions[idx]])

        print("> Saved prediction at ", args.save_path)