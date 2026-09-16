# Topological Segmentation Research Project

This repository implements a compact research prototype for testing whether persistent homology can capture topological information that overlap-based metrics such as Dice and BCE miss.

## Research question

The central idea is:

> Similar Dice/BCE scores do not necessarily imply similar topology.

The current project focuses on a synthetic 2D setting in which the topology is explicitly controlled so that a user can compare overlap metrics and topological summaries.

## Current implementation scope

We currently implement:

- synthetic 2D masks,
- deterministic perturbations,
- filtration logic,
- cubical persistent homology for toy images,
- Betti-number sanity checks,
- Dice and BCE functions,
- a few toy experiments.

We intentionally do not implement yet:

- PCE,
- neural networks,
- U-Net,
- volumetric segmentation,
- cortical mesh analysis,
- training pipeline.

## Filtration convention

We use:

```python
filtration = 1.0 - probability_map
```

This is intentional because higher segmentation probability corresponds to more likely foreground, while the standard sublevel-set persistent homology convention interprets smaller function values as entering the filtration first.

## Installation

```bash
cd /home/yejuny2/Desktop/code/PCE
python -m pip install -r topological_segmentation/requirements.txt
```

## Running tests

```bash
cd /home/yejuny2/Desktop/code/PCE
PYTHONPATH=topological_segmentation/src pytest -q topological_segmentation/tests
```

## Toy experiments

```bash
cd /home/yejuny2/Desktop/code/PCE
PYTHONPATH=topological_segmentation/src python topological_segmentation/experiments/toy_2d/01_basic_topology.py
PYTHONPATH=topological_segmentation/src python topological_segmentation/experiments/toy_2d/02_hole_sensitivity.py
PYTHONPATH=topological_segmentation/src python topological_segmentation/experiments/toy_2d/03_matched_dice.py
```
