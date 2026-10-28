# Pip Satellite Sub-networks

An optional layer of small, rule-based satellites around the main PipLM (model.py, about 1.11B parameters). The main model code is unchanged. Everything here is pure Python and PyTorch, with no extra dependencies.

## Architecture

    user text
       |
    router        (brainstem)      classify: chat, math, code, memory
    thalamus      (sensory)        valence, arousal, threat
    endocrine     (hypothalamus)   dopamine, serotonin, cortisol, oxytocin with decay
    hippocampus   (memory)         recall top episodes by cosine similarity
    specialists   (math, code)     exact arithmetic or syntax check, never eval()
       |
    PipLM generate()  with hormone-shaped temperature and top-k
       |
    critic        (prefrontal)     voice and authenticity score, one cooler retry
       |
    hippocampus.store, then dream.py offline consolidation

## Modules

- router.py: keyword routing into math, code, memory, or chat.
- thalamus.py: lexicon-based stimulus with valence in [-1, 1], arousal and threat in [0, 1].
- endocrine.py: four levels in [0, 1], each relaxing to baseline with a half-life in turns. Dopamine decays in about 2 turns, cortisol in 5, oxytocin in 8, serotonin in 12. Sampling: temperature in [0.4, 1.2], top-k at least 10.
- hippocampus.py: episodes stored as JSON with 512-dim hashed bag-of-words vectors (zlib.crc32, stable across runs).
- specialists.py: safe arithmetic via ast, and a Python syntax check that parses but never runs code.
- critic.py: scores repetition, length, role-marker leaks, and assistant boilerplate. Passing threshold is 0.6.
- dream.py: removes near-duplicate episodes (cosine at or above 0.95) and keeps the most salient up to a budget.
- orchestrator.py: wires the modules to PipLM using generate() from generate.py.

## Usage

    from generate import load_model
    from subnets.orchestrator import Orchestrator

    model, tok = load_model("checkpoints/pip.pt")
    pip = Orchestrator(model, tok, memory_path="checkpoints/episodes.json")
    print(pip.respond("hello there"))

Consolidate memory between sessions:

    python -m subnets.dream checkpoints/episodes.json --keep 500

## Notes

- Episode files live in checkpoints/ and are not committed (same rule as checkpoints).
- The critic and router are heuristics. Treat them as a first pass, not a guarantee.
- The hormone model is a control signal for sampling, not a claim about biology.
