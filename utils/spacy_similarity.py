"""
Tests the linguistic similarity between two sentences in English

Usage:
python3 spacy_similarity.py --reference="the sun rises in the east" --hypothesis="the sun sets in the west"

"""

import spacy
import argparse

nlp = spacy.load('en')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Reference sentence (ground truth)"
    parser.add_argument("-r", "--reference", help=help_)
    help_ = "Hypothesis or test sentence"
    parser.add_argument("-t", "--hypothesis", help=help_)
    args = parser.parse_args()

    if args.reference is None:
        print("Please provide a reference sentence")
        exit(1)
    else:
        reference = nlp(args.reference)

    if args.hypothesis is None:
        print("Please provide a hypothesis sentence")
        exit(1)
    else:
        hypothesis = nlp(args.hypothesis)

    score = reference.similarity(hypothesis)

    print("Score: ", score)
