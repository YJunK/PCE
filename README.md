# PCE / Topological Segmentation Workspace

This workspace contains the initial prototype for a synthetic 2D topological segmentation study.

The project lives under the `topological_segmentation/` directory and is intentionally kept small, modular, and mathematically transparent.

## Overview

The current prototype focuses on:

- synthetic binary disk generation,
- controlled topology perturbations,
- Dice/BCE metrics,
- Betti number sanity checks,
- cubical persistent homology for 2D toy data,
- toy experiments that compare overlap metrics and topology.

The project is designed so that future extensions can add:

- 3D volumetric data,
- cortical mesh representations,
- thickness-based filtration,
- more advanced topological losses,
- neural-network training stages.

## Quick start

```bash
cd /home/yejuny2/Desktop/code/PCE
python -m pip install -r topological_segmentation/requirements.txt
PYTHONPATH=topological_segmentation/src pytest -q topological_segmentation/tests
```
