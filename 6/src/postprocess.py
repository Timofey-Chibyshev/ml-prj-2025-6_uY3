import numpy as np
import torch
from iam_sentences_crnn.alphabet import greedy_ctc_decoder

def postprocess_model_output(output):
    nsamples = output.size(0)
    res = np.empty((nsamples,), dtype='<U128')
    for (idx, sample) in enumerate(output):
        _, indices = torch.max(sample, 1)

        sentence = greedy_ctc_decoder(indices)
        sentence = sentence.replace("|", " ")
        res[idx] = sentence

    return res
