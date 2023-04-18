
T5's Simplified Relative Positional Encoding
https://arxiv.org/abs/1910.10683
* Simplified relative positional endcoding based on learned bias

One Write-Head Is All You Need
https://arxiv.org/abs/1911.02150
* When using MHA, you can reuse a single key/value with each head having its own query which should cut down on parameters

Do Transformers Need Deep Long-Range Memory?
https://arxiv.org/pdf/2007.03356.pdf
* Shallow layers perform worse with longer memories
* Up to 1/6 reduction in memory lengths
* Pairs well with adaptive attention span, which lets this be learned

Compressive transformers for long-range sequence modelling
https://arxiv.org/pdf/1911.05507.pdf
* Recurrent memory by applying lossy compression to old memories which are then concatenated

* [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913)
* [Augmenting Self-attention with Persistent Memory](https://arxiv.org/pdf/1907.01470.pdf)
  - Proves FF networks are equivalent to attention with static memory
* [Attention Approximates Sparse Distributed Memory](https://arxiv.org/abs/2111.05498)
  - Theoretical basis for why FF might be both attention and memory
* [Memorizing Transformers](https://arxiv.org/abs/2203.08913)
  - kNN memory, paper uses it as an alternative to recurrency
* [Neural Turing Machines](https://arxiv.org/abs/1410.5401)

### Layerwise feedback
* [Addressing Some Limitations of Transformers with Feedback Memory](https://arxiv.org/abs/2002.09402)
  - Using output of upper layers for lower (modified: per layer pair, no pooling)
* [Memory transformers](https://arxiv.org/abs/2006.11527)
  - Concatenating memory to input tokens (modified: no memory controller)

## Research

### Misc improvements
* [Cramming: Training a Language Model on a Single GPU in One Day](https://arxiv.org/abs/2212.14034)
  - Removing bias has negligible effect on loss and reduces parameters
* [Transformers without Tears](https://arxiv.org/abs/1910.05895)
  - Scaled L2 normalization leads to faster convergence than LayerNorm
* [Towards Better Few-Shot and Finetuning Performance with Forgetful Causal Language Models](https://arxiv.org/abs/2210.13432)
  - Masking prior tokens at random ala BERT-type models leads to better generalization
* [Query-Key Normalization for Transformers](https://arxiv.org/abs/2010.04245)
  - L2 normalization along head dimension of query and key matrix with learnable scaling
  - Prevents attention operation from overflowing and removes need for numerical stability prior to softmax - both are problems for Transformers
* [Primer: Searching for Efficient Transformers for Language Modeling](https://arxiv.org/abs/2109.08668)
  - Squared ReLU performs better than GELU
  - GLU + GELU performs even better according to x-transformers but that adds parameters
  - Also cheaper to implement

### Techniques
* [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
  - Adding rotary embedding to every layer can improve learning
* [A Length-Extrapolatable Transformer](https://arxiv.org/abs/2212.10554v1)
  - Rotary positional embeddings don't generalize input length well
  - Modifies it so it does
* [Rethinking Positional Encoding in Language Pre-training](https://arxiv.org/abs/2006.15595)
  - TUPE positional embeddings learned separately rather than additive
  
* [Burst-dependent synaptic plasticity can coordinate learning in hierarchical circuits](https://pubmed.ncbi.nlm.nih.gov/33986551/)
  - Biological plausibility of layerwise feedback