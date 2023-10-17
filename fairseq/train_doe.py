#!/usr/bin/env python3 -u

import math
import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F

index_file = open('/apdcephfs/share_916081/apheliosgao/results/knn-index/law/knn_index_4_0.8_10.txt', 'r')
index_lines = index_file.readlines()
target_file = open('/apdcephfs/share_916081/apheliosgao/results/knn-index/law/targets.txt', 'r')
target_lines = target_file.readlines()
targets = []
for target in target_lines:
    targets.append(map(int, target.split()))

targets = np.array(targets)
targets = torch.from_numpy(targets)
print (targets.shape)

vals_all = np.memmap('/apdcephfs/share_916081/apheliosgao/datastores/postnorm/law/vals.npy', dtype=np.int, mode='r')
keys_all = np.memmap('/apdcephfs/share_916081/apheliosgao/datastores/postnorm/law/keys.npy', dtype=np.float16, mode='r', shape=(vals_all.shape[0], 1024))
hiddens = np.memmap('/apdcephfs/share_916081/apheliosgao/results/knn-index/law/hidden_states.npy', dtype=np.float16, mode='r', shape=(vals_all.shape[0], 1024))

keys_all = torch.from_numpy(keys_all)
vals_all = torch.from_numpy(vals_all)
print (keys_all.shape)
print (vals_all.shape)


def cross_entropy(lprobs, target, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        
    else:
        nll_loss = nll_loss.squeeze(-1)
        
    if reduce:
        nll_loss = nll_loss.sum()
    return nll_loss

def train(keys, vals, init_model):
    # bsz = args.batch_size
    model = nn.Linear(1024, 42024, bias=False, device=torch.device('cuda:0'))
    model.weight = init_model.weight.clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    optimizer.zero_grad()
    lprobs1 = model(keys)
    lprobs = F.log_softmax(lprobs1, dim=-1, dtype=torch.float32)
    loss = cross_entropy(
        lprobs,
        vals,
    )
    loss.backward()
    optimizer.step()

    return model

PATH = "/apdcephfs/share_916081/apheliosgao/models/wmt19/wmt19.de-en.ffn8192.pt"

def main():
    embed_tokens = nn.Linear(1024, 42024, bias=False, device=torch.device('cuda:0'))
    model_dict = torch.load(PATH)['state_dict']
    new_dict = {k: v for k, v in model_dict.items() if k == 'embed_tokens.weight'}
    print(new_dict.keys)
    embed_tokens.load_state_dict(new_dict)
    for i, index_line in enumerate(index_lines):
        index = list(map(int, index_line.split()))
        # print (index)
        index = torch.tensor(index, device=torch.device('cuda:0'))
        keys = keys_all[index].to(torch.device('cuda:0'))
        vals = vals_all[index].to(torch.device('cuda:0'))
        h = torch.tensor(hiddens[i], device=torch.device('cuda:0'))
        doe_model = train(keys, vals, embed_tokens)
        probs1 = doe_model(h)
        probs = F.softmax(probs1, dim=-1, dtype=torch.float32)
        print ("step")
    
    return

main()
        
'''
self.embed_tokens.weight = self.embed_tokens.weight
'''