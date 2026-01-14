# Chapter 06a: Advanced Memory Dynamics (Supplement)

**Prerequisites**: [Chapter 06: MemoryUpdate](06-memory-update.md)  
**Status**: Practical guide to principled memory management  
**Reading time**: ~45 minutes

---

## Overview

In [Chapter 06](06-memory-update.md), we learned Algorithm 1: MemoryUpdate from the original paper. It uses a **hard threshold** $\tau$ for experience association:

$$\text{Associate particles } i \text{ and } j \text{ if } k(z_i, z_j) > \tau$$

**The problem**: This threshold is:
- **Brittle**: Sensitive to $\tau$ choice
- **Global**: Same $\tau$ everywhere (doesn't adapt to density)
- **Manual**: Requires tuning, not data-driven

This chapter presents **two practical improvements** that address these limitations:

1. **Top-k Adaptive Neighbors**: Density-aware, no global threshold
2. **Surprise-Gated Consolidation**: Data-driven, biologically inspired

Both are straightforward to implement and provide immediate benefits over the baseline.

---

## Table of Contents

1. [Why Go Beyond Hard Thresholds?](#motivation)
2. [Method 1: Top-k Adaptive Neighbors](#top-k)
3. [Method 2: Surprise-Gated Consolidation](#surprise-gating)
4. [Combining Both Methods](#combining)
5. [When to Use Which?](#decision-guide)
6. [Implementation Notes](#implementation)
7. [Further Reading](#further-reading)

---

<a name="motivation"></a>
## 1. Why Go Beyond Hard Thresholds?

### The Hard Threshold Problem

**Algorithm 1 uses**:
```python
for i in range(len(memory)):
    if kernel(z_new, z_i) > tau:  # Hard threshold
        memory[i].w += lambda_prop * kernel(z_new, z_i) * w_new
```

**Issues**:

**Issue 1: Density Variation**

In dense regions (many particles):
- Many particles exceed $\tau$ → lots of associations
- Computation expensive
- Redundant updates

In sparse regions (few particles):
- Few particles exceed $\tau$ → few associations
- Under-generalization
- Slow learning

**One $\tau$ can't handle both!**

---

**Issue 2: Sensitivity**

Small changes in $\tau$ cause large changes in behavior:
- $\tau$ too high: No associations, no generalization
- $\tau$ too low: Everything associates, no selectivity
- "Just right" $\tau$ is different for each environment

**Manual tuning is tedious and fragile.**

---

**Issue 3: Not Data-Driven**

$\tau$ doesn't consider:
- Prediction error (is this experience surprising?)
- Particle importance (is this a critical memory?)
- Learning stage (early vs. late training)

**We can do better with adaptive, data-driven criteria.**

---

### What Makes a Good Alternative?

**Desirable properties**:
1. **Adaptive**: Adjusts to local density automatically
2. **Data-driven**: Uses TD-error, novelty, or other signals
3. **Simple**: Easy to implement and understand
4. **Efficient**: No significant computational overhead
5. **Stable**: Robust to hyperparameter choices

The two methods below satisfy these criteria.

---

<a name="top-k"></a>
## 2. Method 1: Top-k Adaptive Neighbors

### The Idea

**Instead of global threshold $\tau$:**

> Associate each particle with its **k nearest neighbors** (by kernel similarity)

**Per-particle threshold**: $\tau_i$ = similarity to $k$-th nearest neighbor

**Effect**:
- Dense regions: High $\tau_i$ (many close neighbors)
- Sparse regions: Low $\tau_i$ (few close neighbors)
- **Self-normalizing** across density variations

---

### Algorithm

**For new experience** $(z_{\text{new}}, w_{\text{new}})$:

1. Compute similarities to all existing particles:
   $$a_i = k(z_{\text{new}}, z_i) \quad \text{for } i = 1, \ldots, N$$

2. Sort similarities in descending order

3. Select **top-k** particles (k nearest neighbors)

4. Update only these k particles:
   $$w_i \leftarrow w_i + \lambda_{\text{prop}} \cdot a_i \cdot w_{\text{new}} \quad \text{for } i \in \text{top-k}$$

5. Add new particle: $(z_{\text{new}}, w_{\text{new}})$ to memory

---

### Code Example

```python
import numpy as np

def memory_update_topk(memory, z_new, w_new, kernel, k=5, lambda_prop=0.5):
    """
    MemoryUpdate with top-k adaptive neighbors.
    
    Args:
        memory: List of (z_i, w_i) tuples
        z_new: New augmented state (s, theta)
        w_new: New weight (typically reward or TD target)
        kernel: Kernel function k(z, z')
        k: Number of neighbors to update
        lambda_prop: Propagation strength
    
    Returns:
        Updated memory
    """
    if len(memory) == 0:
        # First particle
        return [(z_new, w_new)]
    
    # Compute similarities to all existing particles
    similarities = []
    for i, (z_i, w_i) in enumerate(memory):
        sim = kernel(z_new, z_i)
        similarities.append((i, sim))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Select top-k neighbors
    top_k_indices = [idx for idx, sim in similarities[:k]]
    
    # Update top-k neighbors
    for i in top_k_indices:
        z_i, w_i = memory[i]
        sim = similarities[i][1]  # Get similarity
        memory[i] = (z_i, w_i + lambda_prop * sim * w_new)
    
    # Add new particle
    memory.append((z_new, w_new))
    
    return memory
```

---

### Example: 1D Navigation

**Setup**:
- State space: $s \in [0, 10]$
- Action space: $\theta \in \{-1, +1\}$ (left/right)
- Kernel: RBF with bandwidth $\sigma = 1.0$

**Scenario**: Agent explores, creating particles at:
- Dense region: $s = \{5.0, 5.1, 5.2, 5.3, 5.4\}$ (5 particles)
- Sparse region: $s = \{1.0, 9.0\}$ (2 particles)

**New experience**: $s = 5.25$, $\theta = +1$, $r = 1.0$

---

**Hard Threshold** ($\tau = 0.5$):

Particles that exceed $\tau$:
- All 5 particles in dense region (wasteful!)
- 0 particles in sparse region

**Top-k** ($k = 3$):

Top-3 neighbors (by similarity):
1. $s = 5.2$ (closest)
2. $s = 5.3$ (second closest)
3. $s = 5.1$ (third closest)

**Only these 3 are updated** — efficient and appropriate!

---

### Advantages

✅ **Density-adaptive**: Automatically adjusts to local structure

✅ **No tuning**: $k$ is more intuitive than $\tau$ (number of neighbors)

✅ **Efficient**: Fixed computational cost ($O(N \log k)$ if using heap)

✅ **Stable**: Small changes in $k$ don't drastically change behavior

---

### Choosing $k$

**Rule of thumb**:
- $k = 5$: Good default for most environments
- $k = 3$: More selective (sparse updates)
- $k = 10$: More diffuse (broad generalization)

**Heuristic**: $k \approx$ number of neighbors within typical kernel bandwidth

**Practical**: Try $k \in \{3, 5, 10\}$, pick based on performance

---

<a name="surprise-gating"></a>
## 3. Method 2: Surprise-Gated Consolidation

### The Idea

**Key insight from neuroscience**: 
- **Surprising experiences** (high prediction error) → stored distinctly
- **Predictable experiences** (low prediction error) → consolidated into existing memories

**In GRL terms**:

$$\text{Surprise}(z_t) = |Q^+(z_t) - y_t|$$

where $y_t = r_t + \gamma \max_a Q^+(s_{t+1}, a)$ is the TD target.

---

### Algorithm

**For new experience** $(z_t, y_t)$:

1. Query current field: $\hat{Q} = Q^+(z_t)$

2. Compute surprise: $\text{surprise} = |\hat{Q} - y_t|$

3. **Decision**:
   
   **If surprise $> \tau_{\text{surprise}}$**: (Novel/unexpected)
   - **Create new particle**: $(z_t, y_t)$
   - Do NOT update neighbors
   
   **If surprise $\leq \tau_{\text{surprise}}$**: (Familiar/predictable)
   - **Update neighbors** via association
   - Do NOT create new particle (consolidate into existing)

---

### Why This Works

**High surprise** (large TD-error):
- Current memory is **wrong** about this experience
- Storing distinctly **preserves** this information for learning
- Avoids corrupting existing memories

**Low surprise** (small TD-error):
- Current memory is **already accurate**
- Consolidating **compresses** redundant information
- Saves memory space

**Result**: Memory grows **only when needed** (for surprising events)

---

### Code Example

```python
def memory_update_surprise_gated(memory, z_new, r_t, s_next, 
                                  kernel, gamma=0.99, 
                                  tau_surprise=0.5, lambda_prop=0.5):
    """
    MemoryUpdate with surprise-gated consolidation.
    
    Args:
        memory: List of (z_i, w_i) tuples
        z_new: New augmented state (s, theta)
        r_t: Reward
        s_next: Next state
        kernel: Kernel function
        gamma: Discount factor
        tau_surprise: Surprise threshold
        lambda_prop: Propagation strength
    
    Returns:
        Updated memory
    """
    # Compute TD target
    Q_next = max([query_field(memory, (s_next, a), kernel) 
                  for a in action_space])
    y_t = r_t + gamma * Q_next
    
    # Query current estimate
    Q_current = query_field(memory, z_new, kernel)
    
    # Compute surprise
    surprise = abs(Q_current - y_t)
    
    if surprise > tau_surprise:
        # High surprise: Store distinctly
        memory.append((z_new, y_t))
        print(f"High surprise ({surprise:.2f}): New particle created")
    else:
        # Low surprise: Consolidate into neighbors
        for i, (z_i, w_i) in enumerate(memory):
            sim = kernel(z_new, z_i)
            if sim > 0.1:  # Small threshold for efficiency
                memory[i] = (z_i, w_i + lambda_prop * sim * y_t)
        print(f"Low surprise ({surprise:.2f}): Consolidated")
    
    return memory


def query_field(memory, z, kernel):
    """Query the reinforcement field at z."""
    return sum(w_i * kernel(z, z_i) for z_i, w_i in memory)
```

---

### Example: GridWorld Navigation

**Setup**:
- 5×5 GridWorld, goal at (4, 4), reward = +10
- Agent has seen goal region before (memory exists)

**Scenario 1**: Agent at (3, 4), moves right, reaches goal

- Current estimate: $Q^+ \approx 9.5$ (accurate)
- Actual: $r + \gamma Q^+ = 10 + 0 = 10$
- Surprise: $|9.5 - 10| = 0.5$ (low)
- **Action**: Consolidate into existing memories

**Scenario 2**: Agent discovers shortcut (new optimal path)

- Current estimate: $Q^+ \approx 2.0$ (outdated)
- Actual: $r + \gamma Q^+ = 0 + 0.99 \times 9.5 = 9.4$ (much better!)
- Surprise: $|2.0 - 9.4| = 7.4$ (high!)
- **Action**: Create new particle (preserve discovery)

---

### Advantages

✅ **Data-driven**: Uses TD-error, not manual threshold

✅ **Memory-efficient**: Only grows when necessary

✅ **Biologically plausible**: Mirrors human memory consolidation

✅ **Exploration-friendly**: High-surprise regions naturally explored more

---

### Choosing $\tau_{\text{surprise}}$

**Rule of thumb**:
- $\tau_{\text{surprise}} = 0.5 \times \text{reward scale}$
- If rewards are $\in [0, 10]$: try $\tau_{\text{surprise}} = 0.5$
- If rewards are $\in [-1, +1]$: try $\tau_{\text{surprise}} = 0.1$

**Heuristic**: Start high (store most things), decrease over time (consolidate more)

**Adaptive**: Use running average of TD-errors to set $\tau_{\text{surprise}}$ dynamically

---

<a name="combining"></a>
## 4. Combining Both Methods

**Best of both worlds**: Use **surprise-gating** for formation, **top-k** for propagation.

### Hybrid Algorithm

```python
def memory_update_hybrid(memory, z_new, r_t, s_next, kernel,
                          k=5, gamma=0.99, tau_surprise=0.5, 
                          lambda_prop=0.5):
    """
    Hybrid MemoryUpdate: surprise-gating + top-k.
    
    Formation: Surprise-gated (create new particle if surprise > tau)
    Propagation: Top-k neighbors (update k nearest)
    """
    # Compute TD target
    Q_next = max([query_field(memory, (s_next, a), kernel) 
                  for a in action_space])
    y_t = r_t + gamma * Q_next
    
    # Query current estimate
    Q_current = query_field(memory, z_new, kernel)
    surprise = abs(Q_current - y_t)
    
    # === FORMATION (surprise-gated) ===
    should_create_new = (surprise > tau_surprise) or (len(memory) == 0)
    
    # === PROPAGATION (top-k) ===
    if len(memory) > 0:
        # Compute similarities
        similarities = [(i, kernel(z_new, z_i)) 
                        for i, (z_i, w_i) in enumerate(memory)]
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Update top-k neighbors
        for i, sim in similarities[:k]:
            z_i, w_i = memory[i]
            memory[i] = (z_i, w_i + lambda_prop * sim * y_t)
    
    # === ADD NEW (if needed) ===
    if should_create_new:
        memory.append((z_new, y_t))
    
    return memory
```

---

### Why This Combination Works

**Surprise-gating** (formation):
- Controls **when** to add particles
- Memory grows only for important experiences
- Bounded memory growth

**Top-k** (propagation):
- Controls **how** to update neighbors
- Efficient, density-adaptive
- Stable generalization

**Together**:
- **Selective formation** + **efficient propagation**
- **Bounded memory** + **good generalization**
- Works across diverse environments

---

<a name="decision-guide"></a>
## 5. When to Use Which?

### Method Comparison

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Hard Threshold** (baseline) | Simple, interpretable | Brittle, not adaptive | Initial prototypes, uniform density |
| **Top-k Adaptive** | Density-adaptive, stable | Still grows unbounded | Variable density environments |
| **Surprise-Gating** | Memory-efficient, data-driven | Requires TD-error | Online learning, lifelong tasks |
| **Hybrid (Top-k + Surprise)** | Best of both | Slightly more complex | Production systems, resource-constrained |

---

### Decision Tree

```
Are you resource-constrained (memory/compute)?
├─ YES: Use Surprise-Gating or Hybrid
└─ NO: 
    │
    Is your environment density-varying?
    ├─ YES: Use Top-k or Hybrid
    └─ NO: Hard threshold is fine (keep it simple)
```

---

### Recommendations by Use Case

**Prototyping / Research**:
- Start with **Hard Threshold** (baseline)
- If memory grows too large → add **Surprise-Gating**
- If performance varies by region → add **Top-k**

**Production / Robotics**:
- Use **Hybrid** (surprise + top-k)
- Bounded memory, stable performance
- Tune $k$ and $\tau_{\text{surprise}}$ on validation set

**Lifelong Learning**:
- Use **Surprise-Gating** (required for bounded memory)
- Consider adding **Decay** (old particles fade)
- Periodically **Prune** low-importance particles

---

<a name="implementation"></a>
## 6. Implementation Notes

### Computational Complexity

| Method | Per-Update Cost | Memory Cost |
|--------|-----------------|-------------|
| **Hard Threshold** | $O(N)$ | $O(N)$ (unbounded) |
| **Top-k** | $O(N + k \log k)$ | $O(N)$ (unbounded) |
| **Surprise-Gating** | $O(N)$ | $O(N_{\text{surprise}})$ (bounded!) |
| **Hybrid** | $O(N + k \log k)$ | $O(N_{\text{surprise}})$ (bounded) |

**Note**: All methods are $O(N)$ in existing memory size. Top-k adds small $O(k \log k)$ overhead for sorting.

---

### Optimization Tricks

**1. Efficient Top-k with Heap**

Instead of sorting all similarities:

```python
import heapq

def get_topk_neighbors(similarities, k):
    """Get top-k largest similarities efficiently."""
    # Use max-heap (negate values for min-heap)
    return heapq.nlargest(k, similarities, key=lambda x: x[1])
```

Reduces sorting from $O(N \log N)$ to $O(N + k \log k)$.

---

**2. Adaptive $\tau_{\text{surprise}}$**

Use running statistics:

```python
class AdaptiveSurpriseThreshold:
    def __init__(self, initial_tau=0.5, percentile=75):
        self.tau = initial_tau
        self.td_errors = []
        self.percentile = percentile
    
    def update(self, td_error):
        """Update threshold based on recent TD-errors."""
        self.td_errors.append(abs(td_error))
        if len(self.td_errors) > 100:
            self.td_errors.pop(0)  # Keep last 100
        
        # Set tau to 75th percentile of recent TD-errors
        self.tau = np.percentile(self.td_errors, self.percentile)
    
    def should_create_new(self, surprise):
        return surprise > self.tau
```

**Effect**: $\tau_{\text{surprise}}$ adapts to environment scale automatically.

---

**3. Periodic Pruning**

Prevent unbounded growth (even with surprise-gating):

```python
def prune_memory(memory, max_size=1000, kernel=None):
    """Remove least important particles if memory exceeds max_size."""
    if len(memory) <= max_size:
        return memory
    
    # Compute importance scores (e.g., weight magnitude)
    importance = [(i, abs(w_i)) for i, (z_i, w_i) in enumerate(memory)]
    importance.sort(key=lambda x: x[1], reverse=True)
    
    # Keep top max_size particles
    keep_indices = set(i for i, score in importance[:max_size])
    memory = [p for i, p in enumerate(memory) if i in keep_indices]
    
    return memory
```

**When to prune**: After every episode or every N steps.

---

### Hyperparameter Guidelines

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| $k$ (top-k) | 5 | 3-10 | Higher → more generalization |
| $\tau_{\text{surprise}}$ | 0.5 | 0.1-1.0 | Scale with reward magnitude |
| $\lambda_{\text{prop}}$ | 0.5 | 0.1-1.0 | Same as baseline |
| $\gamma$ (discount) | 0.99 | 0.9-0.999 | Task-dependent |

**Meta-learning**: Learn these from distribution of tasks (see Chapter 08 for details).

---

<a name="further-reading"></a>
## 7. Further Reading

### Within This Tutorial Series

**Part I Tutorials:**
- **[Chapter 05: Particle Memory](05-particle-memory.md)** — Foundations of particle representation
- **[Chapter 06: MemoryUpdate](06-memory-update.md)** — Algorithm 1 baseline

**Quantum-Inspired Extensions** (for full theory):
- **[Chapter 07: Learning Beyond GP](../quantum_inspired/07-learning-the-field-beyond-gp.md)** — Alternative learning mechanisms (online SGD, sparse methods, MoE, neural networks)
- **[Chapter 08: Memory Dynamics](../quantum_inspired/08-memory-dynamics-formation-consolidation-retrieval.md)** — Formation, consolidation, retrieval operators with full theoretical treatment

---

### Additional Methods Not Covered Here

For even more advanced approaches, see Chapter 08:

**Soft Association** (no threshold):
- Temperature-controlled: $\alpha_{ij} = \text{softmax}(\gamma k(z_i, z_j))$
- Differentiable, smooth

**MDL Consolidation**:
- Minimize: $\text{TD-error}(Q^+) + \lambda |\Omega|$
- Principled compression via information theory
- More complex to implement

**Memory Type Tags**:
- Factual (never forget)
- Experiential (normal decay)
- Working (fast decay)

---

### Related Literature

**Adaptive Memory in RL:**
- Schaul et al. (2015). "Prioritized Experience Replay." *ICLR*.
- Isele & Cosgun (2018). "Selective Experience Replay for Lifelong Learning." *AAAI*.

**Neuroscience of Memory:**
- McClelland et al. (1995). "Why There Are Complementary Learning Systems in the Hippocampus and Neocortex."
- Dudai (2004). "The Neurobiology of Consolidations, Or, How Stable Is the Engram?" *Annual Review of Psychology*.

**Information-Theoretic Learning:**
- Rissanen (1978). "Modeling by Shortest Data Description." *Automatica*.

---

## Summary

### Key Takeaways

1. **Hard thresholds are brittle** — density-adaptive and data-driven alternatives are better

2. **Top-k Adaptive Neighbors** — Simple, practical improvement
   - No global threshold
   - Self-normalizing across density
   - $k = 5$ is a good default

3. **Surprise-Gated Consolidation** — Data-driven memory growth
   - High TD-error → new particle
   - Low TD-error → consolidate
   - Bounded memory growth

4. **Hybrid approach is best** for production — combines both benefits

5. **Implementation is straightforward** — code examples provided, minimal overhead

---

### Next Steps

**Immediate**:
- [ ] Replace hard threshold in your MemoryUpdate with top-k
- [ ] Measure memory growth and performance
- [ ] Compare to baseline (Algorithm 1)

**Advanced**:
- [ ] Implement surprise-gating
- [ ] Add adaptive $\tau_{\text{surprise}}$
- [ ] Periodic pruning for long-running agents

**Theoretical Depth**:
- [ ] Read [Chapter 08](../quantum_inspired/08-memory-dynamics-formation-consolidation-retrieval.md) for operator formalism
- [ ] Explore MDL consolidation
- [ ] Study memory type differentiation

---

**Last Updated**: January 14, 2026  
**Next**: [Chapter 07: RF-SARSA Algorithm](07-rf-sarsa.md) (planned)
