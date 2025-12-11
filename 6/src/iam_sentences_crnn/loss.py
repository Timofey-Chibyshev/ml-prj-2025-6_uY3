from torch import nn
import torch
from .alphabet import alphabet, greedy_ctc_decoder, ints_to_chars
from .preprocess_dataset import target_chunks
import numpy as np

import Levenshtein as lev

class MyCTCLoss(nn.Module):
    def __init__(self): 
        super(MyCTCLoss, self).__init__()  
    
    def forward(self, inputs, targets):
        batch_size = inputs.shape[0]
        input_lengths = target_chunks * torch.ones(batch_size, dtype=torch.long)
        target_lengths = torch.ones(batch_size, dtype=torch.long)
        for (idx, val) in enumerate(targets):
            ctc_blank_place_matches = (val == len(alphabet)).nonzero()
            match_idx = ctc_blank_place_matches[0] if ctc_blank_place_matches.size(0) > 0 else targets.shape[1]
            target_lengths[idx] = match_idx

        targets_flat_len = torch.sum(target_lengths)
        targets_flat = torch.zeros(targets_flat_len, dtype=torch.long)
        cur_idx = 0
        for (idx, val) in enumerate(target_lengths):
            targets_flat[cur_idx:(cur_idx + val)] = targets[idx][:val]
            cur_idx += val

        inputsT = torch.transpose(inputs, 0, 1)
        loss = nn.CTCLoss(blank=len(alphabet))
        return loss(inputsT, targets_flat, input_lengths, target_lengths)
    
class MyCERAccuracy(nn.Module):
    def __init__(self): 
        super(MyCERAccuracy, self).__init__()  
    
    def forward(self, inputs, targets):
        nsamples = inputs.size(0)
        cers = np.zeros((nsamples,), dtype=np.float64)
        for (idx, sample) in enumerate(inputs):
            target = targets[idx]
            target = ints_to_chars(target).replace('+','')

            _, indices = torch.max(sample, 1)
            sentence = greedy_ctc_decoder(indices)

            cers[idx] = lev.distance(target, sentence) / len(target)
        
        return (1.0 - np.mean(cers)) * 100.0
