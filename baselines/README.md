# Baselines

This directory contains lightweight wrappers/scaffolds for baselines discussed in the CURATE paper.

- `superfiltering_ifd.py`: a minimal IFD-style scaffold. Replace `simple_ifd_score` with the exact IFD definition used in your experiments.
- LESS: we recommend using the official LESS implementation and converting its selected examples to the same JSONL format consumed by CURATE. Document the command, checkpoint, and selection ratio used in your experiments here.
