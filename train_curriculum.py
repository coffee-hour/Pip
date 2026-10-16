#!/usr/bin/env python3
"""Two-stage curriculum for Pip.

Stage 1: TinyStories (input.txt, from download_data.py) up to --stage1_steps (default 3000).
Stage 2: assistant dialogue (data/assistant.txt, from make_assistant_data.py) up to --stage2_steps (default 6000).

Usage: python train_curriculum.py
Re-running continues where it stopped. A fresh start (ignoring any old checkpoint) happens only
when checkpoints/curriculum.json does not exist yet.

Tokenizer: byte-level BPE trained once on the stage 1 text and stored in the checkpoint. Every byte
has an ID, so assistant text never hits an unknown token, and stage 2 keeps the same vocabulary.

Stop: SIGTERM or Ctrl-C is forwarded to train.py, which saves and exits. Re-run to continue.
"""
import argparse
import json
import os
import signal
import subprocess
import sys

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
STAGE1_DATA = os.path.join(ROOT, 'input.txt')
STAGE2_DATA = os.path.join(ROOT, 'data', 'assistant.txt')

CHILD = {'proc': None}


def forward_stop(signum, frame):
    proc = CHILD['proc']
    if proc is not None and proc.poll() is None:
        proc.send_signal(signum)
    print('stop requested: waiting for train.py to save...', flush=True)


def ckpt_step(path):
    if not os.path.exists(path):
        return 0
    return int(torch.load(path, map_location='cpu')['step'])


def run(cmd):
    CHILD['proc'] = subprocess.Popen(cmd, cwd=ROOT)
    code = CHILD['proc'].wait()
    CHILD['proc'] = None
    return code


def main():
    ap = argparse.ArgumentParser(description='Pip two-stage curriculum runner')
    ap.add_argument('--ckpt', default=os.path.join('checkpoints', 'pip.pt'))
    ap.add_argument('--stage1_steps', type=int, default=3000)
    ap.add_argument('--stage2_steps', type=int, default=6000)
    ap.add_argument('--batch_size', type=int, default=8)
    args = ap.parse_args()

    if not os.path.exists(STAGE1_DATA):
        raise SystemExit('input.txt is missing. Run: python download_data.py')
    if not os.path.exists(STAGE2_DATA):
        print('building assistant data...', flush=True)
        if run([sys.executable, os.path.join(ROOT, 'make_assistant_data.py')]) != 0:
            raise SystemExit('could not build assistant data')

    ckpt = os.path.join(ROOT, args.ckpt)
    state_path = os.path.join(os.path.dirname(ckpt), 'curriculum.json')
    fresh = not os.path.exists(state_path)
    if fresh:
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump({'started': True}, f)
        print('new curriculum: stage 1 starts from scratch (any old checkpoint is ignored)', flush=True)

    signal.signal(signal.SIGTERM, forward_stop)
    signal.signal(signal.SIGINT, forward_stop)

    base = [sys.executable, os.path.join(ROOT, 'train.py'), '--out', ckpt,
            '--batch_size', str(args.batch_size)]

    step = 0 if fresh else ckpt_step(ckpt)
    if step < args.stage1_steps:
        print('stage 1 (TinyStories): steps ' + str(step) + ' to ' + str(args.stage1_steps), flush=True)
        cmd = base + ['--data', STAGE1_DATA, '--steps', str(args.stage1_steps)]
        if fresh:
            cmd.append('--fresh')
        run(cmd)
        step = ckpt_step(ckpt)
        if step < args.stage1_steps:
            print('stage 1 stopped at step ' + str(step) + '. Re-run python train_curriculum.py to continue.', flush=True)
            return
        print('stage 1 complete at step ' + str(step) + '.', flush=True)

    if step < args.stage2_steps:
        print('stage 2 (assistant dialogue): steps ' + str(step) + ' to ' + str(args.stage2_steps), flush=True)
        cmd = base + ['--data', STAGE2_DATA, '--resume', ckpt, '--steps', str(args.stage2_steps)]
        run(cmd)
        step = ckpt_step(ckpt)
        if step < args.stage2_steps:
            print('stage 2 stopped at step ' + str(step) + '. Re-run python train_curriculum.py to continue.', flush=True)
            return

    print('curriculum complete at step ' + str(step) + '. Checkpoint: ' + ckpt, flush=True)


if __name__ == '__main__':
    main()
