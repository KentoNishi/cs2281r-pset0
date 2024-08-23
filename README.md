# cs2281r-pset0

## Implementation Deviations

### Parallelized Multi-Head Self-Attention

In the original tutorial, the multi-head self-attention mechanism is implemented in a concatenated list comprehension loop. This is quite inefficient, so I parallelized the computation using tensor operations. The code is based on my own fork of Andrej Karpathy's nanoGPT repository ([KentoNishi/generic-nanogpt](https://github.com/KentoNishi/generic-nanogpt)) which I created in 2023.
