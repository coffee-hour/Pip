#!/usr/bin/env python3
"""Curriculum for Pip-Medium (about 26.5M parameters): a fixed step plan of CPU training.

Stage 1: TinyStories (input.txt, from download_data.py) up to --stage1_steps (default 3000).
Stage 2: conversation and personality (data/conversation.txt) up to step 5000.
Stage 3: tools and reasoning (data/tools_reasoning.txt) up to step 7500.
Stage 4: companion polish, mixing mood, sprite, tools and chat (data/companion_polish.txt) up to step 10000.

The stage 2 to 4 datasets come from make_skill_data.py and are built automatically if missing.
Pass --rebuild_data to regenerate them. The generator is deterministic, so this changes nothing.

Usage: python train_curriculum.py
       python train_curriculum.py --train_args "--bf16 --threads 4"   (extra flags for train.py)

CLEAN START FOR PIP-MEDIUM: an existing checkpoint whose architecture does not match Pip-Medium
(8 layers, 8 heads, width 512, context 512), such as the old 9.7M Pip, is NOT loaded. It is renamed
to <name>.legacy.pt so nothing is lost, the old curriculum.json is removed, and training starts from
scratch with a freshly trained tokenizer. A checkpoint that already matches is adopted as stage 1
progress, as before, so an interrupted Pip-Medium run resumes where it stopped.

Schedule: each stage calls train.py with --steps set to that stage's end step. The cosine learning
rate decays toward that end step, so it restarts from wherever it is at the stage boundary.

Stop: SIGTERM or Ctrl-C is forwarded to train.py, which saves and exits. Re-run to continue.
"""
import argparse
import json
import os
import shlex
import signal
import subprocess
import sys

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
STAGE1_DATA = os.path.join(ROOT, 'input.txt')
DATA_DIR = os.path.join(ROOT, 'data')

MEDIUM = {'n_layer': 8, 'n_head': 8, 'n_embd': 512, 'block_size': 512}

# (name, dataset file under data/, end step)
SKILL_STAGES = [
    ('conversation', 'conversation.txt', 5000),
    ('tools_reasoning', 'tools_reasoning.txt', 7500),
    ('companion_polish', 'companion_polish.txt', 10000),
]
SKILL_FILES = [f for _, f, _ in SKILL_STAGES]

CHILD = {'proc': None}


def forward_stop(signum, frame):
    proc = CHILD['proc']
    if proc is not None and proc.poll() is None:
        proc.send_signal(signum)
    print('stop requested: waiting for train.py to save...', flush=True)


def load_meta(path):
    return torch.load(path, map_location='cpu')


def ckpt_step(path):
    if not os.path.exists(path):
        return 0
    return int(load_meta(path)['step'])


def is_medium(path):
    """True when the checkpoint architecture matches Pip-Medium exactly."""
    cfg = load_meta(path).get('config', {})
    return all(cfg.get(k) == v for k, v in MEDIUM.items())


def run(cmd):
    CHILD['proc'] = subprocess.Popen(cmd, cwd=ROOT)
    code = CHILD['proc'].wait()
    CHILD['proc'] = None
    return code


def build_skill_data(rebuild):
    missing = [f for f in SKILL_FILES if not os.path.exists(os.path.join(DATA_DIR, f))]
    if not missing and not rebuild:
        return True
    print('building skill datasets (make_skill_data.py)...', flush=True)
    return run([sys.executable, os.path.join(ROOT, 'make_skill_data.py')]) == 0


def main():
    ap = argparse.ArgumentParser(description='Pip-Medium skill curriculum runner')
    ap.add_argument('--ckpt', default=os.path.join('checkpoints', 'pip.pt'))
    ap.add_argument('--stage1_steps', type=int, default=3000)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--rebuild_data', action='store_true', help='regenerate the skill datasets first')
    ap.add_argument('--train_args', default='', help='extra flags passed to train.py, e.g. "--bf16 --threads 4"')
    ap.add_argument('--stage2_steps', type=int, default=None,
                    help='no longer used: the skill stages have fixed end steps (5000, 7500, 10000)')
    args = ap.parse_args()
    if args.stage2_steps is not None:
        print('note: --stage2_steps is ignored. Stage end steps are fixed in SKILL_STAGES.', flush=True)
    extra = shlex.split(args.train_args)

    ckpt = os.path.join(ROOT, args.ckpt)
    state_path = os.path.join(os.path.dirname(ckpt), 'curriculum.json')
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)

    fresh = False
    if os.path.exists(ckpt) and not is_medium(ckpt):
        legacy = ckpt[:-3] + '.legacy.pt' if ckpt.endswith('.pt') else ckpt + '.legacy'
        os.replace(ckpt, legacy)
        print('found a checkpoint that is not Pip-Medium: moved to ' + legacy +
              '. Starting Pip-Medium from scratch.', flush=True)
        if os.path.exists(state_path):
            os.remove(state_path)

    if not os.path.exists(state_path):
        if os.path.exists(ckpt):
            print('found Pip-Medium checkpoint ' + ckpt + ' at step ' + str(ckpt_step(ckpt)) +
                  ': adopting it as stage 1 progress (not restarting)', flush=True)
        else:
            fresh = True
            print('new curriculum: Pip-Medium stage 1 starts from scratch', flush=True)
        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump({'started': True, 'model': MEDIUM}, f)

    if not os.path.exists(STAGE1_DATA):
        raise SystemExit('input.txt is missing. Run: python download_data.py')
    if not build_skill_data(args.rebuild_data):
        raise SystemExit('could not build skill datasets')

    signal.signal(signal.SIGTERM, forward_stop)
    signal.signal(signal.SIGINT, forward_stop)

    base = [sys.executable, os.path.join(ROOT, 'train.py'), '--out', ckpt,
            '--batch_size', str(args.batch_size)] + extra

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
    else:
        print('stage 1 already complete at step ' + str(step) + ': skipping to skill stages', flush=True)

    for name, fname, end_step in SKILL_STAGES:
        step = ckpt_step(ckpt)
        if step >= end_step:
            print('stage ' + name + ' already complete at step ' + str(step) + ': skipping', flush=True)
            continue
        print('stage ' + name + ' (' + fname + '): steps ' + str(step) + ' to ' + str(end_step), flush=True)
        cmd = base + ['--data', os.path.join(DATA_DIR, fname), '--resume', ckpt, '--steps', str(end_step)]
        run(cmd)
        step = ckpt_step(ckpt)
        if step < end_step:
            print('stage ' + name + ' stopped at step ' + str(step) + '. Re-run python train_curriculum.py to continue.', flush=True)
            return
        print('stage ' + name + ' complete at step ' + str(step) + '.', flush=True)

    print('curriculum complete at step ' + str(ckpt_step(ckpt)) + '. Checkpoint: ' + ckpt, flush=True)


if __name__ == '__main__':
    main()
