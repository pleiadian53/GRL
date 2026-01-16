# Chapter 5: Concept Subspaces, Projections, and Measurement Theory

## Motivation

In [Chapter 4](04-action-and-state-fields.md), we saw how one state $Q^+$ gives rise to multiple coordinate representations by projecting onto different subspaces (action slices, state slices).

But all those projections were **pointwise**—we projected onto individual basis elements $k(z, \cdot)$.

**Natural question:** Can we project onto **multi-dimensional subspaces** discovered by spectral analysis?

**Answer:** Yes! And this gives us a rigorous framework for **concepts** in reinforcement learning.

This chapter develops:

- **Concepts as invariant subspaces** (not clusters)
- **Projection operators** for concept activation
- **Measurement theory** connecting to quantum mechanics
- **Hierarchical composition** via nested subspaces
- **Practical algorithms** for concept-driven learning

**This formalizes Part II (Emergent Structure & Spectral Abstraction) of the GRL tutorial paper.**

---

## 1. From Spectral Clustering to Concept Subspaces

### The Problem with Clusters

**Traditional clustering** (k-means, hierarchical, spectral) produces:

- **Discrete assignments:** Point $x$ belongs to cluster $k$
- **Hard boundaries:** Sharp transitions between clusters
- **No smooth interpolation:** Can't blend concepts

**This doesn't match how concepts work in cognition or RL!**

---

### The GRL Approach: Functional Clustering

**Section V of the original paper** introduces spectral clustering in RKHS:

1. Compute kernel matrix $K_{ij} = k(z_i, z_j)$
2. Eigendecomposition: $K = \Phi \Lambda \Phi^T$
3. Cluster eigenvectors by similarity
4. Eigenmodes = "concepts"

**But what does this mean mathematically?**

---

### Concepts as Subspaces

**Key insight:** Each cluster of eigenvectors defines a **subspace** in RKHS.

**Formal definition:**

Let $\{\phi_{k,1}, \phi_{k,2}, \ldots, \phi_{k,m_k}\}$ be eigenvectors in cluster $k$.

**Concept $k$ is the subspace:**

$$\mathcal{C}_k = \text{span}\{\phi_{k,1}, \phi_{k,2}, \ldots, \phi_{k,m_k}\} \subset \mathcal{H}_k$$

**Properties:**

- $\mathcal{C}_k$ is a linear subspace of the RKHS
- Dimension: $\dim(\mathcal{C}_k) = m_k$
- Orthogonal decomposition: $\mathcal{H}_k = \bigoplus_k \mathcal{C}_k \oplus \mathcal{C}_{\perp}$

---

### Why Subspaces, Not Clusters?

**Subspaces give you:**

1. **Smooth activation:** Degree of membership, not binary
2. **Compositionality:** Combine multiple concepts
3. **Interpolation:** Blend between concepts
4. **Hierarchy:** Nested subspaces = hierarchical concepts
5. **Operators:** Well-defined projection and measurement

**Clusters only give you:** Hard assignments.

---

## 2. Projection Operators

### Definition

For concept subspace $\mathcal{C}_k$ with orthonormal basis $\{\phi_{k,i}\}_{i=1}^{m_k}$:

**Projection operator:**

$$P_k: \mathcal{H}_k \to \mathcal{C}_k$$

$$P_k f = \sum_{i=1}^{m_k} \langle f, \phi_{k,i} \rangle_{\mathcal{H}_k} \phi_{k,i}$$

**For the reinforcement field:**

$$P_k Q^+ = \sum_{i=1}^{m_k} \langle Q^+, \phi_{k,i} \rangle_{\mathcal{H}_k} \phi_{k,i}$$

---

### Properties

**1. Idempotence:**

$$P_k^2 = P_k$$

(Projecting twice = projecting once)

**2. Orthogonality:**

$$P_k P_\ell = 0 \quad \text{for } k \neq \ell$$

(Different concepts are orthogonal)

**3. Completeness:**

$$\sum_k P_k + P_{\perp} = I$$

(Concepts span the full space)

**4. Self-adjoint:**

$$\langle P_k f, g \rangle = \langle f, P_k g \rangle$$

(Symmetric inner product)

**These are exactly the properties of quantum mechanical projection operators!**

---

### Visual Intuition

In 3D, projecting onto a plane:

```
         Q⁺ (state)
          •
         /|\
        / | \
       /  |  \
      /   |   \  ← projection
     /____|____\ 
    plane (concept subspace)
```

$P_k Q^+$ is the "shadow" of $Q^+$ on the concept subspace.

---

### Computational Form

**Given:**

- Reinforcement field: $Q^+ = \sum_i w_i k(z_i, \cdot)$
- Concept basis: $\{\phi_{k,1}, \ldots, \phi_{k,m_k}\}$

**Compute projection:**

$$P_k Q^+ = \sum_{j=1}^{m_k} \underbrace{\left(\sum_i w_i \langle k(z_i, \cdot), \phi_{k,j} \rangle\right)}_{\text{coefficient } c_{k,j}} \phi_{k,j}$$

**In matrix form:** $\mathbf{c}_k = \mathbf{K} \mathbf{w}$ where:

- $K_{ji} = \langle k(z_i, \cdot), \phi_{k,j} \rangle$
- $\mathbf{w} = [w_1, \ldots, w_N]^T$

---

## 3. Concept Activation: Observables

### The Measurement Question

Given field $Q^+$ and concept $k$, we want to measure:

> "How strongly does the current field activate this concept?"

**This is exactly the quantum measurement problem!**

---

### Concept Activation Observable

**Definition:**

$$A_k = \|P_k Q^+\|_{\mathcal{H}_k}^2$$

**Expanded form:**

$$A_k = \langle P_k Q^+, P_k Q^+ \rangle_{\mathcal{H}_k} = \sum_{i=1}^{m_k} |\langle Q^+, \phi_{k,i} \rangle|^2$$

**Interpretation:** Sum of squared projections onto concept basis vectors.

---

### Properties

**1. Non-negativity:**

$$A_k \geq 0$$

**2. Boundedness:**

$$\sum_k A_k \leq \|Q^+\|^2$$

(Total activation bounded by field strength)

**3. Normalized activation:**

$$\tilde{A}_k = \frac{A_k}{\sum_\ell A_\ell + \|P_{\perp} Q^+\|^2}$$

gives a probability-like distribution over concepts.

**4. Continuity:**

$A_k$ varies smoothly as $Q^+$ evolves—no discrete jumps!

---

### Connection to Quantum Mechanics

| Quantum Mechanics | GRL Concepts |
|-------------------|--------------|
| Observable $\hat{O}$ | Projection operator $P_k$ |
| Eigenspace $\mathcal{H}_\lambda$ | Concept subspace $\mathcal{C}_k$ |
| State $\|\psi\rangle$ | Field $Q^+$ |
| Measurement outcome | Concept activation $A_k$ |
| Born rule: $p = \|\langle \lambda \| \psi \rangle\|^2$ | Activation: $A_k = \|P_k Q^+\|^2$ |

**This is not an analogy—it's the same mathematical structure!**

---

## 4. Concept-Conditioned Representations

### Projected Field Values

**Standard field evaluation:**

$$Q^+(z) = \langle Q^+, k(z, \cdot) \rangle_{\mathcal{H}_k}$$

**Concept-conditioned evaluation:**

$$Q^+_k(z) = \langle P_k Q^+, k(z, \cdot) \rangle_{\mathcal{H}_k}$$

**Interpretation:** Value at $z$ according to concept $k$ only.

---

### Concept-Conditioned Policy

**Standard policy:**

$$\pi(a|s) \propto \exp(\beta Q^+(s, a))$$

**Concept-conditioned policy:**

$$\pi_k(a|s) \propto \exp(\beta Q^+_k(s, a))$$

**Use case:** Different concepts induce different policies
- Concept "explore" → high entropy policy
- Concept "exploit" → peaked policy
- Concept "avoid" → negative values

---

### Action Wavefunction per Concept

From [Chapter 4](04-action-and-state-fields.md), we had action wavefunction $\psi_s(a) = Q^+(s, a)$.

**Concept-specific action wavefunction:**

$$\psi_{s,k}(a) = Q^+_k(s, a)$$

**Interpretation:** Action landscape at state $s$ according to concept $k$.

**Visual:**

```
Full field:     ψ_s(a)
                  ***

                 *****

Concept 1:      ψ_{s,1}(a)
                  *
                 ***
                (sharp peak)

Concept 2:      ψ_{s,2}(a)
                *****

               *******
              (broad support)
```

Different concepts emphasize different action modes!

---

## 5. Hierarchical Composition

### Nested Subspaces

**Key idea:** Concepts can be hierarchical via **nested subspaces**.

**Example:**

$$\mathcal{C}_1 \supset \mathcal{C}_{1,1} \supset \mathcal{C}_{1,1,1}$$

**Interpretation:**

- $\mathcal{C}_1$: "Locomotion" (coarse)
- $\mathcal{C}_{1,1}$: "Forward motion" (medium)
- $\mathcal{C}_{1,1,1}$: "Running" (fine)

---

### Hierarchical Projection

**Level 1 (coarse):**

$$Q^+_{\text{coarse}} = P_1 Q^+$$

**Level 2 (medium):**

$$Q^+_{\text{medium}} = P_{1,1} (P_1 Q^+)$$

**Level 3 (fine):**

$$Q^+_{\text{fine}} = P_{1,1,1} (P_{1,1} (P_1 Q^+))$$

**Note:** Because of nesting, $P_{1,1,1} P_{1,1} P_1 = P_{1,1,1}$.

---

### Activation Hierarchy

**Coarse activation:**

$$A_1 = \|P_1 Q^+\|^2$$

**Medium activation (given coarse):**

$$A_{1,1|1} = \frac{\|P_{1,1} Q^+\|^2}{\|P_1 Q^+\|^2}$$

**Fine activation (given medium):**

$$A_{1,1,1|1,1} = \frac{\|P_{1,1,1} Q^+\|^2}{\|P_{1,1} Q^+\|^2}$$

**Interpretation:** Conditional activations down the hierarchy.

---

### Compositional Activation Tree

```
                Q⁺
                 |
        ┌────────┼────────┐
        │        │        │
       P₁       P₂       P₃
       0.6      0.3      0.1
        |
    ┌───┼───┐
    │   │   │
  P₁,₁ P₁,₂ P₁,₃
  0.4  0.3  0.3
    |
  ┌─┼─┐
P₁,₁,₁ P₁,₁,₂
 0.7    0.3
```

**Reading:** 

- 60% activation in concept 1
- Within concept 1, 40% in sub-concept 1.1
- Within 1.1, 70% in sub-sub-concept 1.1.1

**This is a continuous hierarchy, not a discrete tree!**

---

## 6. Spectral Discovery of Concepts

### Algorithm: Concept Subspace Extraction

**Input:** Kernel matrix $K \in \mathbb{R}^{N \times N}$, number of concepts $C$

**Step 1: Eigendecomposition**

$$K = \Phi \Lambda \Phi^T$$

where $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_N)$ with $\lambda_1 \geq \lambda_2 \geq \cdots$.

**Step 2: Select Top Eigenvectors**

Keep top $M$ eigenvectors: $\{\phi_1, \ldots, \phi_M\}$ (e.g., $M = 50$).

**Step 3: Cluster Eigenvectors**

Apply k-means (or other clustering) to eigenvector matrix $\Phi_M \in \mathbb{R}^{N \times M}$ to get $C$ clusters.

**Step 4: Define Concept Subspaces**

For cluster $k$ containing eigenvectors $\{\phi_{i_1}, \ldots, \phi_{i_{m_k}}\}$:

$$\mathcal{C}_k = \text{span}\{\phi_{i_1}, \ldots, \phi_{i_{m_k}}\}$$

**Output:** Concept subspaces $\{\mathcal{C}_1, \ldots, \mathcal{C}_C\}$

---

### Why This Works

**Intuition:** Eigenvectors of the kernel matrix capture **modes of variation** in the augmented space.

**Eigenvectors with similar profiles** → related functional patterns → same concept.

**Mathematical justification:**

- Kernel PCA identifies principal components
- Clustering groups related components
- Subspaces capture multi-dimensional concepts

---

### Adaptive Concept Discovery

**Instead of fixed $C$, use adaptive methods:**

**Gap heuristic:** Choose $C$ where eigenvalue gap is large:

$$\lambda_C \gg \lambda_{C+1}$$

**Information criterion:** Minimize BIC or AIC for cluster count.

**Stability:** Use consensus clustering across multiple runs.

---

## 7. Concept Dynamics and Evolution

### Temporal Concept Activation

As the agent learns, $Q^+(t)$ evolves, so concept activation changes:

$$A_k(t) = \|P_k Q^+(t)\|^2$$

**This gives interpretable learning curves:**

```
Activation
  ^
  | Concept 1 (exploration)
  | ────╮
  |     ╰─────────────────
  |
  | Concept 2 (exploitation)
  |       ╭─────────────
  | ──────╯
  +-----------------------> time
```

"Agent transitioned from exploratory to exploitative concept."

---

### Concept Transition Matrix

**Define concept dominance:** $c(t) = \arg\max_k A_k(t)$

**Transition matrix:** $T_{k\ell} = P(c(t+1) = \ell | c(t) = k)$

**This reveals concept dynamics without discrete states!**

---

### Concept Persistence

**Measure stability:** How long does a concept remain dominant?

$$\tau_k = \mathbb{E}[\text{time concept } k \text{ stays active}]$$

**Persistent concepts** = stable strategies  
**Transient concepts** = exploratory phases

---

## 8. Practical Algorithms

### Algorithm 1: Concept-Conditioned Policy

**Input:** State $s$, concept weights $\{\alpha_k\}$

**Step 1:** Compute concept-conditioned fields:

$$Q^+_k(s, a) = \langle P_k Q^+, k((s,a), \cdot) \rangle$$

**Step 2:** Weighted combination:

$$Q^+_{\text{mix}}(s, a) = \sum_k \alpha_k Q^+_k(s, a)$$

**Step 3:** Policy:

$$\pi(a|s) \propto \exp(\beta Q^+_{\text{mix}}(s, a))$$

**Use case:** Mix exploration and exploitation by adjusting $\alpha_k$.

---

### Algorithm 2: Hierarchical Action Selection

**Input:** State $s$, hierarchy depth $D$

**Level 1:** Select coarse concept:

$$c_1 = \arg\max_k A_k$$

**Level 2:** Select medium concept within $c_1$:

$$c_2 = \arg\max_{\ell \in c_1} A_{\ell|c_1}$$

**...**

**Level D:** Select action using fine concept:

$$a^* = \arg\max_a Q^+_{c_D}(s, a)$$

**Benefit:** Hierarchical decision-making with interpretable intermediate choices.

---

### Algorithm 3: Concept-Based Transfer

**Source task:**

1. Learn concept subspaces $\{\mathcal{C}_k^{\text{source}}\}$
2. Store projection operators $\{P_k^{\text{source}}\}$

**Target task:**

1. Initialize field: $Q^+_{\text{target}} = 0$
2. For each experience $(s, a, r)$:

   - Project particle onto source concepts:
     $$z_{\text{concept}} = \sum_k P_k^{\text{source}} k((s,a), \cdot)$$
   - Update field using projected basis

**Why this works:** Concepts capture abstract structure that transfers across tasks.

---

## 9. Connection to Existing Work

### Eigenoptions (Machado et al., 2017)

**Eigenoptions:** Use eigenvectors of state transition graph as options (skills).

**GRL concepts:** Use eigenvectors of kernel matrix as concept subspaces.

**Similarities:**

- Both use spectral methods
- Both identify "natural" structures

**Differences:**

- Eigenoptions: single eigenvector = one option (discrete)
- GRL concepts: subspace of eigenvectors = one concept (continuous)
- Eigenoptions: hard assignment
- GRL concepts: soft activation

**GRL generalizes eigenoptions to continuous, compositional representations.**

---

### Successor Features (Barreto et al., 2017)

**Successor features:** Represent value function as inner product $V(s) = \langle \psi(s), w \rangle$.

**GRL concepts:** Represent field as projection $Q^+ = \sum_k P_k Q^+$.

**Similarities:**

- Both use linear combinations
- Both enable transfer

**Differences:**

- Successor features: fixed basis (state features)
- GRL concepts: learned basis (eigenvectors)
- Successor features: flat structure
- GRL concepts: hierarchical structure

---

### Affordances (Gibson, 1979; Khetarpal et al., 2020)

**Affordance:** What actions are possible in a state?

**GRL state wavefunction** $\phi_a(s)$ (from Chapter 4) **is a learned affordance map!**

**Concept-conditioned affordances:**

$$\phi_{a,k}(s) = \langle P_k Q^+, k((s,a), \cdot) \rangle$$

shows affordances from perspective of concept $k$.

---

## 10. Implications for Part II of Tutorial Paper

### Current Status of Section V

**Original paper Section V:**

- Introduces spectral clustering idea
- Shows empirical results
- Demonstrates emergent concepts

**What's missing:**

- Formal definition of concepts (beyond "clusters")
- Operational semantics (what do you do with concepts?)
- Connection to learning algorithms

---

### What This Chapter Provides

**Formalization:**

- Concepts = subspaces $\mathcal{C}_k \subset \mathcal{H}_k$
- Activation = observable $A_k = \|P_k Q^+\|^2$
- Hierarchy = nested subspaces

**Operations:**

- Project field onto concepts: $P_k Q^+$
- Condition policy on concepts: $\pi_k(a|s)$
- Compose concepts hierarchically

**Algorithms:**

- Concept discovery (spectral + clustering)
- Concept-conditioned learning
- Hierarchical transfer

---

### Structure for Extended Section V

**Proposed outline:**

**V-A. Motivation**

- Why functional clustering?
- Limitations of discrete concepts

**V-B. Concept Subspaces**

- Definition via eigenspaces
- Projection operators
- This chapter's formalism

**V-C. Spectral Discovery**

- Algorithm
- Adaptive selection
- Stability analysis

**V-D. Concept Dynamics**

- Activation evolution
- Transition patterns
- Persistence measures

**V-E. Hierarchical Composition**

- Nested subspaces
- Multi-level activation
- Compositional policies

**V-F. Empirical Results**

- Concept discovery in benchmark tasks
- Activation curves
- Transfer experiments

**V-G. Connections**

- Eigenoptions, successor features, affordances
- Relation to hierarchical RL literature

---

## Summary

### Key Contributions of This Chapter

1. **Concepts as Subspaces**

   - Not clusters, but linear subspaces $\mathcal{C}_k \subset \mathcal{H}_k$
   - Enables smooth, compositional representations

2. **Projection Operators**

   - Formal definition: $P_k: \mathcal{H}_k \to \mathcal{C}_k$
   - Properties: idempotent, orthogonal, complete
   - Connection to QM measurement theory

3. **Concept Activation Observables**

   - Measure: $A_k = \|P_k Q^+\|^2$
   - Smooth evolution, no discrete jumps
   - Interpretable learning curves

4. **Hierarchical Composition**

   - Nested subspaces: $\mathcal{C}_1 \supset \mathcal{C}_{1,1} \supset \cdots$
   - Conditional activation at each level
   - Natural multi-scale representation

5. **Practical Algorithms**

   - Spectral discovery
   - Concept-conditioned policies
   - Hierarchical action selection
   - Transfer learning

---

### Key Equations

**Projection operator:**

$$P_k Q^+ = \sum_{i=1}^{m_k} \langle Q^+, \phi_{k,i} \rangle_{\mathcal{H}_k} \phi_{k,i}$$

**Concept activation:**

$$A_k = \|P_k Q^+\|_{\mathcal{H}_k}^2 = \sum_{i=1}^{m_k} |\langle Q^+, \phi_{k,i} \rangle|^2$$

**Concept-conditioned field:**

$$Q^+_k(z) = \langle P_k Q^+, k(z, \cdot) \rangle_{\mathcal{H}_k}$$

**Hierarchical activation:**

$$A_{k,\ell|k} = \frac{\|P_{k,\ell} Q^+\|^2}{\|P_k Q^+\|^2}$$

---

### What This Enables

**Theoretical:**

- Rigorous concept formalism
- Quantum measurement theory connection
- Hierarchical composition framework

**Practical:**

- Interpretable learning (activation curves)
- Hierarchical policies (multi-level decisions)
- Transfer learning (concept basis)
- Compositional strategies (concept mixing)

**For Part II:**

- Mathematical foundation for Section V
- Operational algorithms
- Clear connection to QM

---

## Further Reading

### Within This Series

- **[Chapter 1a](01a-wavefunction-interpretation.md):** State Vector vs. Wavefunction
- **[Chapter 2](02-rkhs-basis-and-amplitudes.md):** RKHS Basis and Amplitudes
- **[Chapter 4](04-action-and-state-fields.md):** Action and State Projections

### GRL Tutorials

- **[Tutorial Chapter 2](../tutorials/02-rkhs-foundations.md):** RKHS Foundations
- **[Tutorial Chapter 4](../tutorials/04-reinforcement-field.md):** Reinforcement Field
- **Part II (planned):** Emergent Structure & Spectral Abstraction

### Related Literature

**Spectral Methods in RL:**

- Machado et al. (2017). "A Laplacian Framework for Option Discovery in Reinforcement Learning." *ICML*.
- Mahadevan & Maggioni (2007). "Proto-value Functions: A Laplacian Framework." *ICML*.

**Hierarchical RL:**

- Sutton et al. (1999). "Between MDPs and semi-MDPs: A Framework for Temporal Abstraction."
- Bacon et al. (2017). "The Option-Critic Architecture." *AAAI*.

**Transfer Learning:**

- Barreto et al. (2017). "Successor Features for Transfer in Reinforcement Learning." *NIPS*.
- Taylor & Stone (2009). "Transfer Learning for Reinforcement Learning Domains: A Survey."

**Quantum Measurement Theory:**

- von Neumann, J. (1932). *Mathematical Foundations of Quantum Mechanics*.
- Peres, A. (1993). *Quantum Theory: Concepts and Methods*. Kluwer.

**Functional Data Analysis:**

- Ramsay & Silverman (2005). *Functional Data Analysis*. Springer.

---

## Next Steps

**For Research:**

1. Implement spectral concept discovery algorithm
2. Test concept-conditioned policies on benchmarks
3. Develop hierarchical composition framework
4. Apply to transfer learning problems

**For Tutorial Paper (Part II):**

1. Integrate this formalism into Section V
2. Add concept activation visualizations
3. Show hierarchical decomposition examples
4. Connect to experimental results

**For Extensions (Papers A/B/C):**

- **Paper B:** "Hierarchical Reinforcement Learning via Concept Subspace Projections"
- **Paper C:** "Transfer Learning with Functional Concept Bases"

---

**Last Updated:** January 14, 2026
