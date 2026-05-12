"""FinDPO reproduction package — shared utilities used by scripts/*.

Public submodules:
    seeding      : deterministic seeding (R2)
    paths        : output directory structure (R10)
    env_info     : git / package / hardware capture (R1, R6)
    data         : dataset load, split, SHA256 manifest (R5)
    labels       : label mapping + prompt formatting (single source of truth)
    tracking     : dual W&B + JSONL logger (R4, R9)
    sanity       : R8 sanity tests
"""

__version__ = "0.1.0"
