'''
Runs an inference on a single audio file.
Assumption is data file and checkpoint are in the same args.path

Usage:
    python3 kws-infer.py --wav-file test_audio/right.wav --model-path resnet18-kws-best-acc.pt
'''


import torch
import argparse
import torchaudio
import os
import numpy as np
import librosa
from torchvision.transforms import ToTensor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/speech_commands/")
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--win-length", type=int, default=None)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--wav-file", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)

    args = parser.parse_args()
    return args


# main routine
if __name__ == "__main__":
    CLASSES = ['silence', 'unknown', 'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
               'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no',
               'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
               'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

    idx_to_class = {i: c for i, c in enumerate(CLASSES)}
  
    args = get_args()

    model_path = args.model_path if args.model_path else os.path.join(args.path, "checkpoints", "resnet18-kws-best-acc.pt")
    print("Loading model checkpoint: ", model_path)

    if args.wav_file is None:
        # list wav files given a folder
        print("Searching for random kws wav file...")
        label = CLASSES[2:]
        label = np.random.choice(label)
        path = os.path.join(args.path, "SpeechCommands/speech_commands_v0.02/")
        path = os.path.join(path, label)
        wav_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.wav')]
        # select random wav file
        wav_file = np.random.choice(wav_files)
    else:
        wav_file = args.wav_file
        label = args.wav_file.split("/")[-1].split(".")[0]

    waveform, sample_rate = torchaudio.load(wav_file)
    transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                              n_fft=args.n_fft,
                                                              win_length=args.win_length,
                                                              hop_length=args.hop_length,
                                                              n_mels=args.n_mels,
                                                              power=2.0)

    
    mel = ToTensor()(librosa.power_to_db(transform(waveform).squeeze().numpy(), ref=np.max))
    mel = mel.unsqueeze(0)
    scripted_module = torch.jit.load(model_path)
    pred = torch.argmax(scripted_module(mel), dim=1)
    print(f"Ground Truth: {label}, Prediction: {idx_to_class[pred.item()]}")