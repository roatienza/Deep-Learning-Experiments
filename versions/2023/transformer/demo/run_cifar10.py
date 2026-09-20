"""Minimal pipeline check for the patched transformer_cifar10.ipynb.
Runs the notebook cells, then a reduced training run (2 blocks, embed 32,
2 epochs, batch 64), and verifies the best checkpoint + test eval.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

import matplotlib
matplotlib.use('Agg')

nb = json.load(open('transformer_cifar10.ipynb'))
g = {}
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    if src.startswith('if __name__'):
        continue
    exec(compile(src, f'<cell {i}>', 'exec'), g)

import argparse
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


def small_get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--embed_dim', type=int, default=32)
    p.add_argument('--num_heads', type=int, default=4)
    p.add_argument('--patch_num', type=int, default=8)
    p.add_argument('--kernel_size', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--max-epochs', type=int, default=2)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--accelerator', default='gpu', type=str)
    p.add_argument('--devices', type=int, default=1)
    p.add_argument('--dataset', default='cifar10', type=str)
    p.add_argument('--num_workers', type=int, default=2)
    return p.parse_args('')


g['get_args'] = small_get_args
src15 = ''.join(nb['cells'][15]['source'])
exec(compile(src15, '<cell 15>', 'exec'), g)

# Verify checkpoint was saved.
ckpt_dir = os.path.join(HERE, 'lightning_logs')
found = []
if os.path.isdir(ckpt_dir):
    for root, dirs, files in os.walk(ckpt_dir):
        for f in files:
            if f.endswith('.ckpt'):
                found.append(os.path.join(root, f))
print('CHECKPOINTS:', found)
print('CIFAR10 PIPELINE OK' if found else 'CIFAR10 PIPELINE MISSING CKPT')
