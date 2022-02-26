"""
Tests the linguistic similarity between two sentences in English

Usage (BLEU-4): 
python3 bleu_similarity.py --reference="the sun rises in the east" --hypothesis="the sun sets in the west"

Usage (BLEU-1): 
python3 bleu_similarity.py --reference="the sun rises in the east" --hypothesis="the sun sets in the west" --bleuone


"""

import argparse
from nltk.translate.bleu_score import sentence_bleu

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Reference sentence (ground truth)"
    parser.add_argument("-r", "--reference", help=help_)
    help_ = "Hypothesis or test sentence"
    parser.add_argument("-t", "--hypothesis", help=help_)
    help_ = "Use BLEU-1 instead of default BLEU-4"
    parser.add_argument("-o", "--bleuone", help=help_, action='store_true')
    args = parser.parse_args()

    if args.reference is None:
        print("Please provide a reference sentence")
        exit(1)
    else:
        reference = args.reference

    if args.hypothesis is None:
        print("Please provide a hypothesis sentence")
        exit(1)
    else:
        hypothesis = args.hypothesis

    
    reference = [reference.split()]
    hypothesis = hypothesis.split()
    if args.bleuone:
        score = sentence_bleu(reference, hypothesis, weights=(1, 0, 0, 0))
        print("BLEU-1 Score: ", score)
    else:
        score = sentence_bleu(reference, hypothesis)
        print("BLEU-4 Score: ", score)

