"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

run inference for Image Text Retrieval
"""
import argparse
import sys
from pathlib import Path
import numpy as np

def compute_recall(scores, at_k=1):
    found = 0
    for i in range(len(scores)):
        if i in np.argsort(scores[i])[::-1][:at_k]:
            found += 1
    
    return found / len(scores)
        

if __name__ == '__main__':
    scores = np.load("uniter_scores/scores.npy")

    # I2T retrieval (X-axis is Image, Y-axis is Text)
    print("I2T", "recall@1", compute_recall(scores, at_k=1))
    print("I2T", "recall@2", compute_recall(scores, at_k=2))
    print("I2T", "recall@5", compute_recall(scores, at_k=5))
    print("I2T", "recall@10", compute_recall(scores, at_k=10))

    scores = scores.T
    # T2I retrieval (X-axis is Text, Y-axis is Image)
    print("T2I", "recall@1", compute_recall(scores, at_k=1))
    print("T2I", "recall@2", compute_recall(scores, at_k=2))
    print("T2I", "recall@5", compute_recall(scores, at_k=5))
    print("T2I", "recall@10", compute_recall(scores, at_k=10))
