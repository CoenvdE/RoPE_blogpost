# Rotary positional embeddings (RoPE)

*Draft — expand sections as you write.*

Rotary positional embeddings (RoPE), introduced in [*RoFormer: Enhanced Transformer with Rotary Position Embedding*](https://arxiv.org/abs/2104.09864), encode token positions by rotating pairs of dimensions in query and key vectors instead of adding absolute position vectors.

## Why rotate?

Attention compares queries and keys. RoPE applies a position-dependent rotation to \(q\) and \(k\) so that their inner product depends on **relative** index \(m - n\) in a structured way. Intuitively, frequency choices along embedding pairs create phase shifts that repeat when positions differ by fixed offsets—much like how sinusoidal absolute encodings relate phases across positions, but here the rotation is baked into how \(q\) and \(k\) interact.

## Sketch

Split the hidden dimension into pairs \((x_{2i}, x_{2i+1})\). For position \(m\), multiply each pair by a 2D rotation matrix with angle \(m \theta_i\). Apply the same idea separately to \(q\) at position \(m\) and \(k\) at position \(n\) using their respective indices. The dot product \(q_m^\top k_n\) then picks up terms that depend on \(m - n\) rather than on \(m\) and \(n\) individually.

## Practical notes

- Frequencies \(\theta_i\) are usually chosen in geometric progression (similar spirit to sinusoidal PE).
- RoPE extends naturally to **relative** structure when you generalize the rotation argument (e.g. multidimensional positions).
- Many efficient attention kernels assume RoPE-friendly layouts; keep layer norms and head splitting consistent with your framework.

## References

1. Su *et al.*, RoFormer (RoPE), arXiv:2104.09864.
2. Llama / modern LLM stacks — RoPE is the default positional mechanism in many open-weight models.
