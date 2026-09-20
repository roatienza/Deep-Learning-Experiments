"""Make the CIFAR10 DDP run robust:

1. Drop AMP (precision 32) — the fp16 path was the suspected cause of the
   rank divergence that deadlocked the NCCL all-reduce.
2. Fewer dataloader workers (8 per rank) — 32 workers x 2 ranks was saturating
   the CPU and stalling the DataLoader, which is what pushed one rank 30 min
   behind the other.
3. NCCL debug logging so any future hang is diagnosable.
"""
import json

path = '/home/rowel/sandbox/Deep-Learning-Experiments/versions/2023/transformer/demo/transformer_cifar10.ipynb'
nb = json.load(open(path))

# Cell 13: default workers 8
src = ''.join(nb['cells'][13]['source'])
old = "parser.add_argument('--num_workers', default=4, type=int, metavar='N')"
assert old in src
src = src.replace(old, "parser.add_argument('--num_workers', default=8, type=int, metavar='N')")
nb['cells'][13]['source'] = src.splitlines(keepends=True)

# Cell 15: precision 32, NCCL debug env
src = ''.join(nb['cells'][15]['source'])
old = """    # Use the GPU when available, otherwise CPU.
    accelerator = args.accelerator if torch.cuda.is_available() else 'cpu'
    precision = 16 if accelerator == 'gpu' else 32
    print(f"Running on: {accelerator} (precision {precision})")"""
assert old in src
new = """    # Use the GPU when available, otherwise CPU.
    accelerator = args.accelerator if torch.cuda.is_available() else 'cpu'
    precision = 32
    print(f"Running on: {accelerator} (precision {precision})")

    # NCCL debug logging: if the 2-rank run ever hangs, the log shows which
    # collective each rank was waiting on.
    os.environ.setdefault("NCCL_DEBUG", "WARN")"""
src = src.replace(old, new)
nb['cells'][15]['source'] = src.splitlines(keepends=True)

json.dump(nb, open(path, 'w'), indent=1)
print('patched precision/workers/nccl')
