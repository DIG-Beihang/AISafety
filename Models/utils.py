import sys
sys.path.append('../')
import torch
from utils.deepspeech.inference_config import TranscribeConfig
from utils.deepspeech.decoder import BeamCTCDecoder, GreedyDecoder
from utils.deepspeech.model import DeepSpeech
from utils.deepspeech.enums import DecoderType

def load_model(model_path):
    info = torch.load(model_path, 'cpu')
    hyper_parameters = info['hyper_parameters']
    labels = hyper_parameters['labels']
    precision = hyper_parameters['precision']
    model = DeepSpeech(labels, precision)
    model.load_state_dict(info['state_dict'])
    model.eval()
    return model


def load_decoder(labels):
    cfg = TranscribeConfig
    lm = cfg.lm
    if lm.decoder_type == DecoderType.beam:
        decoder = BeamCTCDecoder(labels=labels,
                                 lm_path=lm.lm_path,
                                 alpha=lm.alpha,
                                 beta=lm.beta,
                                 cutoff_top_n=lm.cutoff_top_n,
                                 cutoff_prob=lm.cutoff_prob,
                                 beam_width=lm.beam_width,
                                 num_processes=lm.lm_workers,
                                 blank_index=labels.index('_'))
    else:
        decoder = GreedyDecoder(labels=labels,
                                blank_index=labels.index('_'))
    return decoder
