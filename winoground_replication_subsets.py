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

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = Path(args.dataset)

    name2subset = {
        "All": [i for i in range(400)],
        "MC": [3, 13, 36, 46, 75, 76, 77, 78, 82, 86, 88, 113, 119, 121, 131, 132, 133, 148, 189, 220, 221, 262, 263, 287, 295, 300, 303, 305, 307, 310, 319, 322, 332, 340, 343, 344, 348, 353, 355, 356, 363, 374, 377, 381, 386, 394],
        "NM": [72, 73, 74, 95, 96, 133, 149, 150, 164, 218, 221, 222, 224, 235, 237, 246, 274, 275, 321, 325, 326, 327, 332, 333, 334, 350, 364, 365, 398, 399],
        "WI": [31, 36, 38, 41, 42, 61, 62, 70, 78, 84, 93, 110, 114, 116, 128, 133, 136, 155, 159, 164, 173, 174, 188, 201, 203, 204, 206, 209, 218, 223, 239, 245, 246, 247, 254, 274, 275, 277, 280, 282, 293, 303, 307, 314, 319, 320, 327, 329, 339, 362, 367, 383, 384, 388, 393, 395],
        "WT": [10, 41, 49, 58, 63, 68, 70, 152, 156, 159, 163, 174, 198, 201, 209, 214, 215, 221, 229, 233, 237, 253, 257, 264, 275, 287, 303, 315, 318, 323, 324, 326, 327, 335, 338, 342, 343, 345, 346, 351, 354, 359, 364, 376, 383, 385, 386, 387, 390, 394],
        "SO": [4, 22, 23, 25, 27, 28, 31, 36, 58, 65, 69, 70, 77, 97, 116, 118, 134, 138, 159, 163, 172, 176, 182, 187, 200, 214, 226, 227, 232, 241, 255, 268, 286, 335, 352, 356, 373, 376],
        "CR": [16, 40, 44, 46, 55, 58, 81, 83, 93, 97, 103, 111, 116, 118, 128, 130, 135, 143, 144, 176, 190, 191, 192, 193, 199, 206, 208, 209, 210, 211, 217, 218, 219, 227, 228, 230, 234, 238, 241, 242, 249, 254, 258, 260, 262, 264, 267, 268, 275, 276, 281, 284, 286, 287, 292, 295, 296, 298, 299, 304, 311, 312, 316, 330, 331, 334, 336, 342, 347, 358, 361, 371, 373, 375, 382, 383, 392, 396],
        "None":  [0, 1, 2, 5, 6, 7, 8, 9, 11, 12, 14, 15, 17, 18, 19, 20, 21, 24, 26, 29, 30, 32, 33, 34, 35, 37, 39, 43, 45, 47, 48, 50, 51, 52, 53, 54, 56, 57, 59, 60, 64, 66, 67, 71, 79, 80, 85, 87, 89, 90, 91, 92, 94, 98, 99, 100, 101, 102, 104, 105, 106, 107, 108, 109, 112, 115, 117, 120, 122, 123, 124, 125, 126, 127, 129, 137, 139, 140, 141, 142, 145, 146, 147, 151, 153, 154, 157, 158, 160, 161, 162, 165, 166, 167, 168, 169, 170, 171, 175, 177, 178, 179, 180, 181, 183, 184, 185, 186, 194, 195, 196, 197, 202, 205, 207, 212, 213, 216, 225, 231, 236, 240, 243, 244, 248, 250, 251, 252, 256, 259, 261, 265, 266, 269, 270, 271, 272, 273, 278, 279, 283, 285, 288, 289, 290, 291, 294, 297, 301, 302, 306, 308, 309, 317, 328, 337, 341, 349, 357, 360, 366, 368, 369, 370, 372, 378, 379, 380, 389, 391, 397],
        "Easier": [0, 1, 2, 5, 6, 7, 8, 9, 11, 12, 14, 15, 17, 18, 19, 20, 21, 24, 26, 29, 30, 32, 33, 34, 35, 37, 39, 43, 45, 47, 48, 50, 51, 52, 53, 54, 56, 57, 59, 60, 64, 66, 67, 71, 72, 73, 74, 79, 80, 85, 87, 89, 90, 91, 92, 94, 95, 96, 98, 99, 100, 101, 102, 104, 105, 106, 107, 108, 109, 112, 115, 117, 120, 122, 123, 124, 125, 126, 127, 129, 133, 137, 139, 140, 141, 142, 145, 146, 147, 149, 150, 151, 153, 154, 157, 158, 160, 161, 162, 164, 165, 166, 167, 168, 169, 170, 171, 175, 177, 178, 179, 180, 181, 183, 184, 185, 186, 194, 195, 196, 197, 202, 205, 207, 212, 213, 216, 218, 221, 222, 224, 225, 231, 235, 236, 237, 240, 243, 244, 246, 248, 250, 251, 252, 256, 259, 261, 265, 266, 269, 270, 271, 272, 273, 274, 275, 278, 279, 283, 285, 288, 289, 290, 291, 294, 297, 301, 302, 306, 308, 309, 317, 321, 325, 326, 327, 328, 332, 333, 334, 337, 341, 349, 350, 357, 360, 364, 365, 366, 368, 369, 370, 372, 378, 379, 380, 389, 391, 397, 398, 399],
        "Harder": [3, 4, 10, 13, 16, 22, 23, 25, 27, 28, 31, 36, 38, 40, 41, 42, 44, 46, 49, 55, 58, 61, 62, 63, 65, 68, 69, 70, 75, 76, 77, 78, 81, 82, 83, 84, 86, 88, 93, 97, 103, 110, 111, 113, 114, 116, 118, 119, 121, 128, 130, 131, 132, 134, 135, 136, 138, 143, 144, 148, 152, 155, 156, 159, 163, 172, 173, 174, 176, 182, 187, 188, 189, 190, 191, 192, 193, 198, 199, 200, 201, 203, 204, 206, 208, 209, 210, 211, 214, 215, 217, 219, 220, 223, 226, 227, 228, 229, 230, 232, 233, 234, 238, 239, 241, 242, 245, 247, 249, 253, 254, 255, 257, 258, 260, 262, 263, 264, 267, 268, 276, 277, 280, 281, 282, 284, 286, 287, 292, 293, 295, 296, 298, 299, 300, 303, 304, 305, 307, 310, 311, 312, 313, 314, 315, 316, 318, 319, 320, 322, 323, 324, 329, 330, 331, 335, 336, 338, 339, 340, 342, 343, 344, 345, 346, 347, 348, 351, 352, 353, 354, 355, 356, 358, 359, 361, 362, 363, 367, 371, 373, 374, 375, 376, 377, 381, 382, 383, 384, 385, 386, 387, 388, 390, 392, 393, 394, 395, 396],
    }


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

    matching_scores = torch.zeros(len(winoground_dataset))
    accuracies = torch.zeros(len(winoground_dataset)*2)

    if args.model == 'uniter':
        import UNITER_codebase.uniter_code as Uniter
        model, tokenizer = Uniter.setup_model(device)
        get_matching_score = lambda image, caption: Uniter.get_matching_score(image, caption, model, tokenizer, device)
    elif args.model == 'lxmert':
        import LXMERT.lxmert_code as Lxmert
        model, tokenizer = Lxmert.setup_model(device)
        get_matching_score = lambda image, caption: Lxmert.get_matching_score(image, caption, model, tokenizer, device)
    else:
        raise ValueError(f"Shouldn't reach here; Model {args.model} not supported")
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        for i in range(len(winoground_dataset)):
            print(i, file=sys.stderr)
            image_0, caption_0, image_1, caption_1 = winoground_dataset[i]

            img0_text0_score = get_matching_score(image_0, caption_0)
            img0_text1_score = get_matching_score(image_0, caption_1)
            img1_text0_score = get_matching_score(image_1, caption_0)
            img1_text1_score = get_matching_score(image_1, caption_1)

            text_score = img0_text0_score > img0_text1_score and img1_text1_score > img1_text0_score
            image_score = img0_text0_score > img1_text0_score and img1_text1_score > img0_text1_score
            group_score = text_score and image_score

            matching_score = img0_text0_score*img1_text1_score > img0_text1_score*img1_text0_score
            
            text_scores[i] = int(text_score)
            image_scores[i] = int(image_score)
            group_scores[i] = int(group_score)

            matching_scores[i] = int(matching_score)

            accuracy_0 = img0_text0_score > 0.5
            accuracy_1 = img0_text1_score > 0.5
            accuracies[i*2] = int(accuracy_0)
            accuracies[i*2+1] = int(accuracy_1)

    
    # Scores per subset
    for name, subset in name2subset.items():
        # print(f"{name} Text score: {text_scores[subset].mean():.4f}")
        # print(f"{name} Image score: {image_scores[subset].mean():.4f}")
        # print(f"{name} Group score: {group_scores[subset].mean():.4f}")
        print(f"{name} Matching score: {matching_scores[subset].mean():.4f}")
        # print(f"{name} Accuracy: {accuracies[subset].mean():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model", type=str, required=True, choices=["uniter", "lxmert"])
    parser.add_argument("--dataset", default="dataset/", type=str)

    args = parser.parse_args()

    main(args)
