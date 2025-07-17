# R-AdaZO: Refining Adaptive Zeroth-Order Optimization at Ease

Welcome to the R-AdaZO project page! This is the official implementation for our paper [Refining Adaptive Zeroth-Order Optimization at Ease](https://openreview.net/forum?id=NpIIbrg361), published at ICML 2025.

> Recently, zeroth-order (ZO) optimization plays an essential role in scenarios where gradient information is inaccessible or unaffordable, such as black-box systems and resource-constrained environments. While existing adaptive methods such as ZO-AdaMM have shown promise, they are fundamentally limited by their underutilization of moment information during optimization, usually resulting in underperforming convergence. To overcome these limitations, this paper introduces Refined Adaptive Zeroth-Order Optimization (R-AdaZO). Specifically, we first show the untapped variance reduction effect of first moment estimate on ZO gradient estimation, which improves the accuracy and stability of ZO updates. We then refine the second moment estimate based on these variance-reduced gradient estimates to better capture the geometry of the optimization landscape, enabling a more effective scaling of ZO updates. We present rigorous theoretical analysis to show (a) the first analysis to the variance reduction of first moment estimate in ZO optimization, (b) the improved second moment estimates with a more accurate approximation of its variance-free ideal, (c) the first variance-aware convergence framework for adaptive ZO methods, which may be of independent interest, and (d) the faster convergence of R-AdaZO than existing baselines like ZO-AdaMM. Our extensive experiments, including synthetic problems, black-box adversarial attack, and memory-efficient fine-tuning of large language models (LLMs), further verify the superior convergence of R-AdaZO, indicating that R-AdaZO offers an improved solution for real-world ZO optimization challenges.

## Installation

Since R-AdaZO is a standalone Python file, you can simply copy it to your project.

```bash
cp radazo.py your_project_directory/
```

## Usage

R-AdaZO can be used similarly to other optimizers in PyTorch. Here's a simple example:

```python
import torch
from radazo import R_AdaZO # Assuming razo_optimizer.py is in the current path

# Define a model or parameters to optimize
param = torch.randn(10, requires_grad=True)

# Define a closure function to compute the loss value
def closure():
    # Example: a simple quadratic function
    return torch.sum(param ** 2)

# Initialize the R-AdaZO optimizer
optimizer = R_AdaZO([param], lr=0.001, betas=(0.9, 0.99), n_samples=10)

# Optimization loop
num_steps = 100
for step in range(num_steps):
    optimizer._reset_seeds() # Reset perturbation seeds at the beginning of each step
    optimizer.zero_grad()    # Zero out gradients (standard practice for PyTorch optimizers, even if not directly used by ZO)
    loss = closure()         # Compute loss
    # loss.backward()        # Backpropagation is not needed for zeroth-order optimization
    optimizer.step(closure)  # Perform optimization step, passing the closure function
    
    if (step + 1) % 10 == 0:
        print(f"Step {step + 1}, Loss: {loss.item():.4f}")

print(f"Optimized parameters: {param.data}")
```

## Citation

If you use R-AdaZO in your research, please cite our paper:

```bibtex
@inproceedings{shu2025refining,
  title={Refining Adaptive Zeroth-Order Optimization at Ease},
  author={Shu, Yao and Zhang, Qixin and He, Kun and Dai, Zhongxiang},
  booktitle={Proc. ICML},
  year={2025}
}
```
