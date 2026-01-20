# Joint Transmit and Pinching Beamforming for Pinching Antenna Systems (PASS): Optimization-Based or Learning-Based?
This is PyTorch code implementation for ML algorithm in paper "Joint Transmit and Pinching Beamforming for Pinching Antenna Systems (PASS): Optimization-Based or Learning-Based?", 
accepted by IEEE Transactions on Wireless Communications (TWC) (Preprint is available at [https://arxiv.org/abs/2502.08637](https://arxiv.org/abs/2502.08637))

This repository contains the reproducible training demo of the proposed **KKT-guided dual-learning (KDL) Transformer**, which provides a novel machine learning (ML) empowered **joint transmit beamforming and pinching beamforming** 
method. 

## Quick Start

- To reproduce the results of KDL-Transformer, run [KDL_Transformer.py](KDL_Transformer.py)
> [Note] The dual variables $\boldsymbol{\lambda}$ are projected onto the range $(1, 10^{12})$.
This range can be adjusted to improve learning stability and overall performance.

## Environment
- Python >= 3.8, torch >= 1.10

## Key Idea of KDL
KDL is a novel learning-to-optimize (L2O) paradigm that combines the strengths of model-driven and data-driven approaches.
- KDL trains an ML model to predict **dual variables**. For optimization variables with **closed-form KKT solutions**, the corresponding primal variables can be reconstructed (e.g., the transmit beamforming matrix in this paper).
- The optimization variables **without** closed-form KKT solutions are predicted jointly with the dual variables in a data-driven manner.
- Advantages of KDL
  - Compared to purely black-box L2O methods, KDL demonstrates significant gains.
  - KDL achieves a faster response than iterative mathematical optimization algorithms at inference time. 
  - For highly oscillatory and strongly coupled optimization problems with multiple local optima, KDL can better avoid undesirable local solutions.

## Reference
If you find this code useful for your research, please consider citing 
> X. Xu, X. Mu, Y. Liu, and A. Nallanathan, ``Joint Transmit and Pinching Beamforming for Pinching Antenna Systems (PASS): Optimization-Based or Learning-Based?'', *IEEE Trans. Wireless Commun.*, accepted, 2026. 
