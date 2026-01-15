# Chapter 8: Memory Dynamics — Formation, Consolidation, and Retrieval

## Motivation

In [Chapter 6](06-agent-state-and-belief-evolution.md), we established that:
- The agent's **state** is the reinforcement field $Q^+ \in \mathcal{H}_k$
- **MemoryUpdate** is the belief evolution operator

In [Chapter 7](07-learning-the-field-beyond-gp.md), we explored **how to learn** the field (GP, ridge, online SGD, sparse, deep nets, MoE).

Now we address **memory dynamics over time:**
1. **Formation:** How is new experience written to memory?
2. **Consolidation:** What should be retained vs. forgotten?
3. **Retrieval:** How is memory accessed for decision-making?

**Why this matters:** Current RL and LLM agents suffer from:
- **Drift:** Long-term memory contaminated by transient information
- **Repetition:** Same mistakes repeated (poor consolidation)
- **Forgetting:** Constraints/facts lost (no principled retention criteria)

GRL can address these by treating memory dynamics as **operators with learnable criteria**, not ad hoc heuristics.

---

## 1. The Three-Layer Memory Stack

### Inspired by Recent Memory Research

Recent work on AI agent memory (Cao et al. 2024, "Memory in the Age of AI Agents") identifies:
- **Forms:** What memory is made of (representation)
- **Functions:** What memory is for (roles)
- **Dynamics:** How memory evolves (write, consolidate, retrieve)

---

### GRL's Memory Stack

**Layer 1: Latent/Internal = The RKHS State**

The "true" agent memory is the function $Q^+ \in \mathcal{H}_k$, represented by particles:

$$\Omega = \{(z_i, w_i)\}_{i=1}^N$$

**This is the belief state.**

---

**Layer 2: External = Persistent Particle Store**

Engineering layer: particle database/graph/tree for:
- Scalable retrieval
- Compression/pruning
- Hierarchical organization

**Semantically:** Stores basis elements of a field, not just "documents" (not RAG!).

---

**Layer 3: Token-Level = Derived Narrative Buffer**

For LLM integration: synthesize "what matters" from particle state into text:
- Top concepts
- Active constraints
- Recent surprises

**Explicitly downstream, not source of truth.**

---

### Key Distinction

> **GRL's primary memory is latent functional memory (the field).**  
> **Token memory is an interface artifact.**

This prevents the "memory is RAG" confusion that plagues current LLM agents.

---

## 2. Memory Functions: What Memory Is For

### Three Memory Roles

**Factual Memory (Stable Constraints)**

**What:** Things that should not drift
- Physical laws
- Task constraints
- Safety rules

**In GRL:** 
- High-persistence anchor particles
- Hard constraints in kernel (ignore irrelevant dimensions)
- Repeller regions in action field

**Example:** "Never use Tool X with PII" → persistent negative weight in action subspace.

---

**Experiential Memory (What Happened + How It Felt)**

**What:** Episode traces with value
- $(s, a, r)$ transitions
- Success/failure outcomes
- Temporal context

**In GRL:**
- This is native particle memory
- Particles = experience evidence
- Weights = fitness/energy
- Kernel overlap = generalization

---

**Working Memory (Task-Local, Short-Horizon)**

**What:** Temporary context for current decision
- Sub-goal state
- Recent observations
- Current plan step

**In GRL:** Temporary overlay field

$$Q^{\text{work}}_t = Q^+_t + \Delta_t$$

where $\Delta_t$ is a fast-decaying particle set or low-rank concept activation.

**Why separate?** Prevents working memory from polluting long-term belief (addresses drift!).

---

## 3. Memory Dynamics: The Three Operators

### Decomposition of MemoryUpdate

MemoryUpdate is actually **three sub-operators:**

$$Q^+_{t+1} = \underbrace{\mathcal{C}}_{\text{consolidate}} \circ \underbrace{\mathcal{P}}_{\text{propagate}} \circ \underbrace{\mathcal{E}}_{\text{inject}}(Q^+_t; \text{experience}_t)$$

Let's formalize each.

---

## 4. Formation (Write): Operator $\mathcal{E}$

### What Formation Does

**Inject new evidence into memory:**

**Input:** $(Q^+_t, (s_t, a_t, r_t))$

**Output:** $Q^+_t$ with new particle or updated weights

---

### Option A: Add New Particle

**Simplest:**

$$\Omega_{t+1} = \Omega_t \cup \{(z_t, w_t)\}$$

where:

- $z_t = (s_t, a_t)$ (augmented state)
- $w_t = r_t$ or TD target $y_t = r_t + \gamma \max_{a'} Q^+_t(s_{t+1}, a')$

**Effect:**

$$Q^+_{t+1}(z) = Q^+_t(z) + w_t k(z_t, z)$$

**Pure growth:** memory size increases by 1.

---

### Option B: Update Existing Weights

**If particle $z_t$ is "close" to existing particles:**

Find neighbors: $\mathcal{N}(z_t) = \{i : k(z_i, z_t) > \epsilon\}$

Update their weights:

$$w_i \leftarrow w_i + \alpha_i \cdot w_t$$

where $\alpha_i = k(z_i, z_t)$ (association strength).

**Effect:** Spread evidence to neighbors via kernel overlap.

---

### Option C: Tag Memory Type

**Distinguish factual/experiential/working:**

**Factual:** High persistence flag
- Decay rate: $\lambda_{\text{factual}} \approx 0$ (never forget)
- Prune priority: low

**Experiential:** Normal persistence
- Decay rate: $\lambda_{\text{exp}} = 0.01$ (slow decay)
- Prune priority: based on predictive value

**Working:** Fast decay
- Decay rate: $\lambda_{\text{work}} = 0.5$ (forget quickly)
- Prune priority: high (after task episode)

---

### Formation Criteria

**When to create new particle vs. update existing?**

**Novelty criterion:**

$$\text{novelty}(z_t) = 1 - \max_i k(z_i, z_t)$$

- If novelty $> \tau_{\text{novel}}$: create new particle
- Else: update neighbors

**Surprise criterion:**

$$\text{surprise}(z_t) = |Q^+_t(z_t) - y_t|$$

- High surprise: store distinctly (new particle)
- Low surprise: consolidate into neighbors

**This is psychologically plausible!** Human memory:
- Novel experiences → encoded distinctly
- Familiar experiences → integrated into schemas

---

## 5. Consolidation (Compress): Operator $\mathcal{C}$

### The Consolidation Problem

**Memory grows unbounded** without consolidation:
- Every experience → new particle
- $N$ increases indefinitely
- Computation/memory: $O(N)$

**Consolidation:** Merge, prune, compress while preserving predictive power.

---

### The Hard Threshold Problem

**Original GRL (Algorithm 1):**

Associate particles if $k(z_i, z_j) > \tau$

**Problems:**
- $\tau$ is a **hard hyperparameter** (not learned)
- Brittle: sensitive to $\tau$ choice
- Doesn't adapt to local density

**We need something better!**

---

### Alternative 1: Soft Association (No Threshold)

**Replace hard threshold with soft weights:**

$$\alpha_{ij} = \frac{\exp(\gamma \, k(z_i, z_j))}{\sum_{j'} \exp(\gamma \, k(z_i, z_{j'}))}$$

**Properties:**
- No $\tau$!
- Smooth: differentiable
- Temperature $\gamma$ controls spread (learnable)

**Effect:** Soft neighborhood graph, not binary adjacency.

---

### Alternative 2: Adaptive Threshold (Top-k Neighbors)

**Per-particle threshold:**

$$\tau_i = \text{quantile}_q \{k(z_i, z_j)\}_{j \neq i}$$

**Choose $\tau_i$ so each particle has $\approx k$ neighbors.**

**Properties:**
- Self-normalizing across regions
- Dense regions: higher $\tau_i$
- Sparse regions: lower $\tau_i$

**Very "memory-like":** Association density adapts to local structure.

---

### Alternative 3: Information-Theoretic Consolidation (MDL)

**Objective:** Minimize description length

$$\min_{\Omega'} \underbrace{\text{TD-error}(Q^+(\Omega'))}_{\text{accuracy}} + \lambda |\Omega'|$$

**Interpretation:** 
- Keep particles that reduce prediction error
- Prune particles that don't contribute

**Merge criterion:** Merge $(z_i, w_i)$ and $(z_j, w_j)$ if it reduces objective.

---

#### Practical Implementation

**Greedy merging:**

1. For each pair $(i, j)$ with $k(z_i, z_j) > \epsilon_{\min}$:
   - Compute merged particle: $z' = (w_i z_i + w_j z_j)/(w_i + w_j)$, $w' = w_i + w_j$
   - Evaluate: $\Delta \text{error} = \text{TD-error after merge} - \text{TD-error before}$
   - Evaluate: $\Delta \text{size} = -1$ (one fewer particle)

2. Merge pair with best trade-off: $\Delta \text{error} + \lambda \Delta \text{size}$

3. Repeat until no beneficial merges remain

**This is principled:** Consolidation is optimization, not heuristic!

---

### Alternative 4: Surprise-Gated Consolidation

**Idea:** How human memory consolidates

**Rule:**
- **High prediction error** → store distinctly (don't merge)
- **Low prediction error** → consolidate (merge with neighbors)

**Formally:**

$$\text{merge-probability}(i, j) \propto k(z_i, z_j) \cdot \exp(-\beta \cdot \text{TD-error}_i)$$

**Properties:**
- Surprising experiences preserved (for learning)
- Predictable experiences compressed (save space)
- $\beta$ controls sensitivity (learnable)

---

### Alternative 5: Nonparametric Clustering (DP Mixtures)

**Treat consolidation as clustering:**

Use Dirichlet Process mixture or Chinese Restaurant Process:
- Prior penalizes too many clusters
- But allows growth when needed

**Association = cluster assignment**

**Properties:**
- No fixed $k$ (clusters)
- Automatic complexity control
- Bayesian: uncertainty-aware

**For GRL:** Each cluster is a "concept" (see Chapter 5!).

---

### Consolidation Summary

| Method | Pros | Cons | Complexity |
|--------|------|------|------------|
| **Soft association** | No threshold, smooth | Still need $\gamma$ | Low |
| **Top-k neighbors** | Density-adaptive, simple | Fixed $k$ | Low |
| **MDL** | Principled, objective-driven | Computationally expensive | Medium |
| **Surprise-gated** | Psychologically plausible | Requires TD-error | Medium |
| **Clustering** | Automatic, Bayesian | Complex inference | High |

**Recommended:** Start with top-k (simple), move to MDL (principled) or surprise-gated (adaptive).

---

## 6. Retrieval (Read): Operator $\mathcal{R}$

### What Retrieval Does

**Query the memory for decision-making:**

**Input:** Query point $z = (s, a)$

**Output:** Field value $Q^+(z)$ and/or related context

---

### Retrieval Modes

**Mode 1: Point Query (Standard)**

$$Q^+(z) = \sum_{i=1}^N w_i k(z_i, z)$$

**Use:** Standard action selection.

---

**Mode 2: Projection Query (Chapter 4)**

**State field (fixed action):**

$$Q^+(s, a_{\text{fixed}}) \text{ for varying } s$$

**Action field (fixed state):**

$$Q^+(s_{\text{fixed}}, a) \text{ for varying } a$$

**Use:** Visualize landscapes, precondition learning.

---

**Mode 3: Concept Projection Query (Chapter 5)**

**Project onto concept subspace $\mathcal{C}_m$:**

$$Q^+_m = P_{\mathcal{C}_m} Q^+$$

**Concept activation:**

$$\text{activation}_m(z) = \|P_{\mathcal{C}_m} k(z, \cdot)\|^2$$

**Use:** Abstract reasoning, hierarchical planning, transfer learning.

---

**Mode 4: Neighborhood Retrieval**

**Find particles similar to $z$:**

$$\mathcal{N}(z) = \{i : k(z_i, z) > \epsilon\}$$

**Use:**
- Explain prediction (which particles contributed?)
- Case-based reasoning
- Memory inspection/debugging

---

### Retrieval Abstraction Levels

**GRL supports multi-scale retrieval:**

| Level | Granularity | Query |
|-------|-------------|-------|
| **Particle** | Fine | $Q^+(z) = \sum_i w_i k(z_i, z)$ |
| **Neighborhood** | Local | $\mathcal{N}(z) = \{i : k(z_i, z) > \epsilon\}$ |
| **Concept** | Coarse | $P_{\mathcal{C}_m} Q^+$ |
| **Global** | Abstract | $Q^+$ itself (full field) |

**Key insight:** Different retrieval protocols serve different purposes — fine-grained control uses particles, abstract reasoning uses concepts.

---

## 7. The Complete Memory Dynamics Pipeline

### Unified Framework

**Operator composition:**

$$Q^+_{t+1} = \mathcal{C}_{\lambda} \circ \mathcal{P}_{\text{soft}} \circ \mathcal{E}_{\text{surprise}}(Q^+_t; (s_t, a_t, r_t))$$

---

### Step-by-Step Algorithm

```python
def memory_dynamics_update(Q_plus, experience, config):
    """
    Complete memory dynamics: formation, propagation, consolidation.
    
    Args:
        Q_plus: Current field (particle set)
        experience: (s_t, a_t, r_t, s_{t+1})
        config: {epsilon, gamma, lambda_mdl, decay_rates, ...}
    
    Returns:
        Q_plus_new: Updated field
    """
    s_t, a_t, r_t, s_next = experience
    z_t = augment(s_t, a_t)
    
    # === FORMATION ===
    # Compute novelty and surprise
    novelty = 1 - max(kernel(z_t, z_i) for z_i in Q_plus.particles)
    y_t = r_t + gamma * max_a(Q_plus.query(s_next, a))
    surprise = abs(Q_plus.query(z_t) - y_t)
    
    if novelty > config.tau_novel or surprise > config.tau_surprise:
        # High novelty/surprise: create new particle
        Q_plus.add_particle(z_t, w_t=y_t, memory_type='experiential')
    else:
        # Low novelty/surprise: update neighbors
        neighbors = Q_plus.neighbors(z_t, epsilon=config.epsilon)
        for i in neighbors:
            alpha_i = kernel(Q_plus.particles[i].z, z_t)
            Q_plus.particles[i].w += alpha_i * y_t
    
    # === PROPAGATION (soft association) ===
    for i in range(len(Q_plus.particles)):
        # Compute soft association weights
        alphas = [softmax_kernel(z_i, z_j, gamma=config.gamma) 
                  for z_j in Q_plus.particles]
        # Spread influence (optional, for coherence)
        Q_plus.particles[i].w = sum(alphas[j] * Q_plus.particles[j].w 
                                    for j in range(len(Q_plus.particles)))
    
    # === CONSOLIDATION ===
    # Option A: MDL-based merging
    Q_plus = mdl_merge(Q_plus, lambda_mdl=config.lambda_mdl)
    
    # Option B: Pruning low-weight particles
    Q_plus.prune(threshold=config.prune_threshold)
    
    # Option C: Decay working memory
    for particle in Q_plus.particles:
        if particle.memory_type == 'working':
            particle.w *= (1 - config.decay_work)
    
    return Q_plus
```

---

### Learnable vs. Fixed Parameters

| Parameter | Type | Notes |
|-----------|------|-------|
| $\epsilon$ (novelty threshold) | Can learn | Adaptive per region |
| $\gamma$ (temperature) | **Should learn** | Controls association spread |
| $\lambda_{\text{MDL}}$ (complexity) | Can learn | Trade accuracy/sparsity |
| Decay rates | Can learn | Per memory type |
| Kernel bandwidth | **Should learn** | Generalization scale |

**Modern approach:** Meta-learn these on a distribution of tasks.

---

## 8. Addressing Agent Drift

### The Drift Problem

**Current LLM agents:**
- Long-term memory contaminated by transient context
- Constraints forgotten after few steps
- Mistakes repeated (no consolidation)

**Root cause:** No separation between working/long-term memory.

---

### GRL Solution

**1. Separate Memory Types (Formation)**

- Factual: persistent, high priority
- Experiential: normal decay
- Working: fast decay

**2. Consolidation Criteria (Not Random)**

- Merge low-surprise experiences (compress)
- Preserve high-surprise experiences (learn)

**3. Retrieval at Right Abstraction**

- Use concepts for abstract reasoning
- Use particles for fine-grained control

---

### Why This Works

**Drift prevention:**

$$Q^+_{\text{total}} = \underbrace{Q^+_{\text{long-term}}}_{\text{stable}} + \underbrace{\Delta_{\text{work}}}_{\text{decays fast}}$$

Working memory $\Delta_{\text{work}}$ doesn't contaminate $Q^+_{\text{long-term}}$ because it decays quickly.

**Constraint preservation:**

Factual memory has $\lambda_{\text{decay}} \approx 0$, so constraints never forgotten.

**Mistake avoidance:**

Consolidation based on TD-error: high-error experiences retained for learning.

---

## 9. Connection to Biological Memory

### Human Memory Stages

**Short-term (working) memory:**
- Capacity: $\sim$7 items
- Duration: seconds to minutes
- Function: active task context

**Long-term memory:**
- Capacity: unlimited
- Duration: lifetime
- Function: knowledge, skills, episodes

**Consolidation:**
- Sleep-dependent
- Surprise-modulated (emotional salience)
- Semantic compression (gist extraction)

---

### GRL Parallels

| Human | GRL | Mechanism |
|-------|-----|-----------|
| **Working memory** | $\Delta_{\text{work}}$ | Fast-decay particles |
| **Long-term memory** | $Q^+_{\text{stable}}$ | Persistent particles |
| **Consolidation** | $\mathcal{C}$ | Merge, prune, compress |
| **Surprise modulation** | Surprise-gated formation | High TD-error → distinct storage |
| **Semantic compression** | Concept formation | Spectral clustering (Chapter 5) |

**GRL provides computational mechanisms for these phenomena!**

---

## 10. Practical Implementation Notes

### For GRL v0 (Baseline)

**Simplest viable memory dynamics:**

1. **Formation:** Add new particle if novelty $> \epsilon$
2. **Consolidation:** Top-k neighbor graph + periodic pruning
3. **Retrieval:** Standard kernel query

**Complexity:** $O(N)$ per update, $O(N)$ per query

---

### For Scalable GRL

**Add:**

1. **Sparse inducing points** ($M \ll N$)
2. **Hierarchical storage** (tree structure)
3. **Lazy consolidation** (only when memory budget exceeded)

**Complexity:** $O(M)$ per update, $O(\log M)$ per query

---

### For Research Extensions

**Explore:**

1. **Meta-learning consolidation criteria**
2. **Amplitude-based memory** (complex weights for phase)
3. **Hierarchical consolidation** (concepts at multiple scales)

---

## Summary

### Key Insights

1. **Memory has three dynamics:** formation, consolidation, retrieval

2. **Each is an operator:** $\mathcal{E}$, $\mathcal{C}$, $\mathcal{R}$

3. **Hard thresholds are brittle** → use adaptive/learned criteria

4. **Memory types matter** → factual/experiential/working have different dynamics

5. **Consolidation is optimization** → MDL, surprise-gating, not ad hoc

6. **Retrieval has abstraction levels** → particle, neighborhood, concept, global

7. **Drift is preventable** → separate working from long-term memory

---

### Key Equations

**Complete update:**

$$Q^+_{t+1} = \mathcal{C} \circ \mathcal{P} \circ \mathcal{E}(Q^+_t; \text{experience}_t)$$

**Soft association:**

$$\alpha_{ij} = \frac{\exp(\gamma \, k(z_i, z_j))}{\sum_{j'} \exp(\gamma \, k(z_i, z_{j'}))}$$

**MDL consolidation:**

$$\min_{\Omega'} \text{TD-error}(Q^+(\Omega')) + \lambda |\Omega'|$$

**Working + long-term:**

$$Q^+_{\text{total}} = Q^+_{\text{stable}} + \Delta_{\text{work}}$$

**Surprise-gated formation:**

$$\text{store-distinct if} \quad |Q^+(z_t) - y_t| > \tau_{\text{surprise}}$$

---

### Principled Memory Management

**Key Principles for Memory Update:**

Replace hard threshold $\tau$ with adaptive criteria:
- **Soft association**: Temperature-controlled ($\gamma$)
- **Top-k adaptive neighbors**: Density-aware
- **MDL-based consolidation**: Optimization-driven
- **Surprise-gating**: Psychologically plausible

**Retention Strategy:**

**What to Retain:**
- High surprise (large TD-error) — valuable for learning
- High novelty (far from existing particles) — new information
- Factual constraints (tagged) — critical knowledge

**What to Forget (merge/prune):**
- Low surprise (predictable) — redundant information
- Redundant (close to neighbors) — can be compressed
- Working memory (after episode) — task-specific, temporary

**Implementation:** MDL consolidation or surprise-gated formation provide principled, data-driven criteria rather than fixed hyperparameters.

---

## Further Reading

### Within This Series

- **[Chapter 5](05-concept-projections-and-measurements.md):** Concept Subspaces (hierarchical retrieval)
- **[Chapter 6](06-agent-state-and-belief-evolution.md):** State Evolution Framework
- **[Chapter 7](07-learning-the-field-beyond-gp.md):** Learning Mechanisms

### GRL Tutorials

- **[Tutorial 5](../tutorials/05-particle-memory.md):** Particle Memory Basics
- **[Tutorial 6](../tutorials/06-memory-update.md):** MemoryUpdate Algorithm

### Related Literature

**Agent Memory:**
- Cao et al. (2024). "Memory in the Age of AI Agents." arXiv:2512.13564.

**Memory Consolidation:**
- McClelland et al. (1995). "Why There Are Complementary Learning Systems."

**Information-Theoretic Learning:**
- Rissanen (1978). "Modeling by Shortest Data Description." *Automatica*.

**Surprise-Modulated Memory:**
- Schultz & Dickinson (2000). "Neuronal Coding of Prediction Errors." *Ann. Rev. Neurosci.*

**Nonparametric Clustering:**
- Rasmussen (1999). "The Infinite Gaussian Mixture Model." *NIPS*.

---

**Last Updated:** January 14, 2026
