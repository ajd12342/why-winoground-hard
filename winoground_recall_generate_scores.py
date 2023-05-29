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
import numpy as np

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = Path(args.dataset)

    add_wha = True if args.model == 'uniter' else False
    # Load winoground dataset
    winoground_dataset = WinogroundDataset(dataset_path/"winoground_features.npy",
                                            dataset_path/"winoground_boxes.npy",
                                            dataset_path/"examples.jsonl",
                                            add_wha, device)
    # winoground_dataloader = DataLoader(winoground_dataset, batch_size=None, shuffle=False)


    text_scores = torch.zeros(len(winoground_dataset))
    image_scores = torch.zeros(len(winoground_dataset))
    group_scores = torch.zeros(len(winoground_dataset))

    accuracies = torch.zeros(len(winoground_dataset)*2)

    if args.model == 'uniter':
        import UNITER_codebase.uniter_code as Uniter
        model, tokenizer = Uniter.setup_model(device)
        get_matching_scores_batched = lambda image, captions: Uniter.get_matching_scores_batched(image, captions, model, tokenizer, device)
    elif args.model == 'lxmert':
        import LXMERT.lxmert_code as Lxmert
        model, tokenizer = Lxmert.setup_model(device)
        get_matching_scores_batched = lambda image, captions: Lxmert.get_matching_scores_batched(image, captions, model, tokenizer, device)
    else:
        raise ValueError(f"Shouldn't reach here; Model {args.model} not supported")
    
    captions = []
    for i in range(len(winoground_dataset)):
        image_0, caption_0, image_1, caption_1 = winoground_dataset[i]
        captions.append(caption_0)
        captions.append(caption_1)

    # Evaluate
    scores = []
    with torch.no_grad():
        for i in range(len(winoground_dataset)):
            print(i, file=sys.stderr)
            image_0, caption_0, image_1, caption_1 = winoground_dataset[i]

            img0_captions_score = get_matching_scores_batched(image_0, captions)
            img1_captions_score = get_matching_scores_batched(image_1, captions)
            scores.append(img0_captions_score.detach().cpu().numpy())
            scores.append(img1_captions_score.detach().cpu().numpy())
    
    scores = np.array(scores)
    with open(f"uniter_scores/scores.npy", "wb") as f:
        np.save(f, scores)
    
    # with open(f"uniter_scores/{args.img_idx}.pkl", 'wb') as f:
    #     pickle.dump([img0_captions_score, img1_captions_score], f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model", type=str, required=True, choices=["uniter", "lxmert"])
    parser.add_argument("--dataset", default="dataset/", type=str)
    parser.add_argument("--img_idx", type=int, default=0)

    args = parser.parse_args()

    main(args)