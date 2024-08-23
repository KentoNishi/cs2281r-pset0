# cs2281r-pset0

## Implementation Deviations

### Parallelized Multi-Head Self-Attention

In the original tutorial, the multi-head self-attention mechanism is implemented in a concatenated list comprehension loop. This is quite inefficient, so I parallelized the computation using tensor operations. The code is based on my own fork of Andrej Karpathy's nanoGPT repository ([KentoNishi/generic-nanogpt](https://github.com/KentoNishi/generic-nanogpt)) which I created in 2023.

My `MultiHeadSelfAttention` class can be found [here](https://github.com/KentoNishi/cs2281r-pset0/blob/master/gpt.py#L94).

### Weight Tying between Token Embedding and LM Head

Weight tying is a common technique in transformer models to reduce the number of parameters. The original tutorial does not implement weight tying to keep the code simple, but I added it to my implementation.

The additional line of code can be found [here](https://github.com/KentoNishi/cs2281r-pset0/blob/master/gpt.py#L188).
