"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

run inference for Image Text Retrieval
"""
import argparse
import sys

import torch
from dataset.dataset import WinogroundDataset
from pathlib import Path
import pickle
import joblib

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = Path(args.dataset)

    add_wha = True if args.model == 'uniter' else False
    # Load winoground dataset

    winoground_dataset = WinogroundDataset(dataset_path/"winoground_features.npy",
                                            dataset_path/"winoground_boxes.npy",
                                            dataset_path/"examples_augmented.jsonl",
                                            add_wha, device)
    # winoground_dataloader = DataLoader(winoground_dataset, batch_size=None, shuffle=False)

    if args.model == 'uniter':
        import UNITER_codebase.uniter_code as Uniter
        model, tokenizer = Uniter.setup_model(device)
        run_inference = lambda image, captions: Uniter.run_inference_batched(image, captions, model, tokenizer, device)
    elif args.model == 'lxmert':
        import LXMERT.lxmert_code as Lxmert
        model, tokenizer = Lxmert.setup_model_inference(device)
        run_inference = lambda image, captions: Lxmert.run_inference_batched(image, captions, model, tokenizer, device)
    else:
        raise ValueError(f"Shouldn't reach here; Model {args.model} not supported")
    
    winoground_sets = []
    # Evaluate
    model.eval()
    with torch.no_grad():
        for i in range(0, 400):
            print(i, file=sys.stderr)
            winoground_set = {}
            image_0, caption_0_list, image_1, caption_1_list = winoground_dataset[i]

            captions_0 = [_["variant"] for _ in caption_0_list]
            captions_1 = [_["variant"] for _ in caption_1_list]

            img0_text0_outputs = run_inference(image_0, captions_0)
            img0_text1_outputs = run_inference(image_0, captions_1)
            img1_text0_outputs = run_inference(image_1, captions_0)
            img1_text1_outputs = run_inference(image_1, captions_1)

            winoground_set['caption0'] = img0_text0_outputs['caption'] # Same as img1_text0_outputs['caption']
            winoground_set['caption1'] = img1_text1_outputs['caption'] # Same as img0_text1_outputs['caption']

            winoground_set['output_img_0_cap_0'] = img0_text0_outputs["model_output"]
            winoground_set['output_img_0_cap_1'] = img0_text1_outputs["model_output"]
            winoground_set['output_img_1_cap_0'] = img1_text0_outputs["model_output"]
            winoground_set['output_img_1_cap_1'] = img1_text1_outputs["model_output"]

            winoground_sets.append(winoground_set)

            del winoground_set
            del img0_text0_outputs
            del img0_text1_outputs
            del img1_text0_outputs
            del img1_text1_outputs
            torch.cuda.empty_cache()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with open(Path(args.output_dir)/"feats.pkl", 'wb') as f:
        pickle.dump(winoground_sets, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model", type=str, required=True, choices=["uniter", "lxmert"])
    parser.add_argument("--dataset", default="dataset/", type=str)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    main(args)