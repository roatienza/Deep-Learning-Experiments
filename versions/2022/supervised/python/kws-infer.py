'''
Runs an inference on a single audio file.
Assumption is data file and checkpoint are in the same args.path

Usage:
    python3 kws-infer.py --wav-file test_audio/right.wav --model-path resnet18-kws-best-acc.pt

To use microphone input, run:
    python3 kws-infer.py --gui

    On RPi 4:
    python3 kws-infer.py --rpi --gui

Dependencies:
    pip3 install pysimplegui
    pip3 install sounddevice 
    pip3 install librosa

    sudo apt-get install libasound2-dev libportaudio2 

Inference time:
    0.03 sec Quad Core Intel i7 2.3GHz
    0.09 sec on RPi 4
'''


import torch
import argparse
import torchaudio
import os
import numpy as np
import librosa
import sounddevice as sd
import time
from torchvision.transforms import ToTensor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/speech_commands/")
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--win-length", type=int, default=None)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--wav-file", type=str, default=None)
    parser.add_argument("--model-path", type=str, default="resnet18-kws-best-acc.pt")
    parser.add_argument("--gui", default=False, action="store_true")
    parser.add_argument("--rpi", default=False, action="store_true")
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

    model_path = args.model_path if args.model_path else os.path.join(
        args.path, "checkpoints", "resnet18-kws-best-acc.pt")
    print("Loading model checkpoint: ", model_path)
    scripted_module = torch.jit.load(model_path)

    if args.gui:
        import PySimpleGUI as sg
        sample_rate = 16000
        sd.default.samplerate = sample_rate
        sd.default.channels = 1
        sg.theme('DarkAmber')
  
    elif args.wav_file is None:
        # list wav files given a folder
        print("Searching for random kws wav file...")
        label = CLASSES[2:]
        label = np.random.choice(label)
        path = os.path.join(args.path, "SpeechCommands/speech_commands_v0.02/")
        path = os.path.join(path, label)
        wav_files = [os.path.join(path, f)
                     for f in os.listdir(path) if f.endswith('.wav')]
        # select random wav file
        wav_file = np.random.choice(wav_files)
    else:
        wav_file = args.wav_file
        label = args.wav_file.split("/")[-1].split(".")[0]

    if not args.gui:
        waveform, sample_rate = torchaudio.load(wav_file)

    transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                     n_fft=args.n_fft,
                                                     win_length=args.win_length,
                                                     hop_length=args.hop_length,
                                                     n_mels=args.n_mels,
                                                     power=2.0)
    if not args.gui:
        mel = ToTensor()(librosa.power_to_db(transform(waveform).squeeze().numpy(), ref=np.max))
        mel = mel.unsqueeze(0)

        pred = torch.argmax(scripted_module(mel), dim=1)
        print(f"Ground Truth: {label}, Prediction: {idx_to_class[pred.item()]}")
        exit(0)

    layout = [ 
        [sg.Text('Detecting...', justification='center', size=(10, 1), font=("Helvetica", 100), key='-OUTPUT-')],
        [sg.Text('', size=(10, 1), font=("Helvetica", 24), key='-TIME-')],
    ]
    window = sg.Window('KWS Inference', layout, finalize=True)
    total_runtime = 0
    n_loops = 0
    while True:
        event, values = window.read(100)
        if event == sg.WIN_CLOSED:
            break
        
        waveform = sd.rec(sample_rate).squeeze()
        
        sd.wait()
        if waveform.max() > 1.0:
            continue
        start_time = time.time()
        if args.rpi:
            waveform = torch.FloatTensor(waveform.tolist())
            mel = np.array(transform(waveform).squeeze().tolist())
            mel = librosa.power_to_db(mel, ref=np.max).tolist()
            
            mel = torch.FloatTensor(mel)
            mel = mel.unsqueeze(0)

        else:
            waveform = torch.from_numpy(waveform).unsqueeze(0)
            mel = ToTensor()(librosa.power_to_db(transform(waveform).squeeze().numpy(), ref=np.max))
        mel = mel.unsqueeze(0)
        pred = scripted_module(mel)
        max_prob =  pred.max()
        elapsed_time = time.time() - start_time
        total_runtime += elapsed_time
        n_loops += 1
        ave_pred_time = total_runtime / n_loops
        if max_prob > 2.0:
            pred = torch.argmax(pred, dim=1)
            human_label = f"{idx_to_class[pred.item()]}"
            window['-OUTPUT-'].update(human_label)
        else:
            window['-OUTPUT-'].update("...")
        
        window['-TIME-'].update(f"{ave_pred_time:.2f} sec")


    window.close()

            

