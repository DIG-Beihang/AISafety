import sys, os
sys.path.append('../')
import torch
from Models.utils import load_decoder, load_model
from Models.pytorch_model import PyTorchAudioModel
from EvalBox.Attack.AudioAttack import *
import torchaudio
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # I/O parameters
    parser.add_argument('--input_wav', type=str, help='input wav. file')
    parser.add_argument('--output_wav', type=str, default='adv.wav', help='output adversarial wav. file')
    parser.add_argument('--model_path', type=str, default='librispeech_pretrained_v3.pth.tar', help='model pth path; please use absolute path')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--target_sentence', type=str, default="HELLO", help='Please use uppercase')

    # plot parameters
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_model(model_path=args.model_path)
    decoder = load_decoder(labels=model.labels)
    audio_model = PyTorchAudioModel(model, decoder, device)

    sound, sample_rate = torchaudio.load(args.input_wav)
    sound = sound.to(device)
    target_sentence = args.target_sentence.upper()
    print(audio_model(sound, decode=True))
    # attacker = FGSMAttacker(model=audio_model, device=args.device)
    # attacker = PGDAttacker(model=audio_model, device=args.device)
    # attacker = CWAttacker(model=audio_model, device=args.device)
    attacker = ImperceptibleCWAttacker(model=audio_model, device=args.device)
    # attacker = GeneticAttacker(model=audio_model, device=args.device)
    adv = attacker.generate(sound, target_sentence)
    print(audio_model(adv, decode=True))
    print((adv - sound).abs().max())
    torchaudio.save(args.output_wav, adv.cpu(), sample_rate=sample_rate)