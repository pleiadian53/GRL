# GRL Research Roadmap

**Last Updated:** January 14, 2026  
**Purpose:** High-level plan for GRL research, documentation, and implementation

---

## Current Focus: GRL v0 (Baseline)

**Status:** 🔄 In Progress — Documentation & Formalization  
**Goal:** Complete tutorial paper and reference implementation of the original GRL framework

### Part I: Particle-Based Learning

**Status:** ✅ 8/10 chapters complete (00-07 written; 08-10 remain)

> **This file is the single source of truth for status.** `README.md`, `docs/index.md`, and `docs/README.md` link here rather than asserting progress themselves. They previously carried four different counts for this one series (6, 7, 8, and 9 of 10, with `README.md` claiming both 9 and 6). Do not reintroduce a count elsewhere.

| Chapter | Title | Status |
|---------|-------|--------|
| 00 | Overview | ✅ Complete |
| 01 | Core Concepts | ✅ Complete |
| 02 | RKHS Foundations | ✅ Complete |
| 03 | Energy and Fitness | ✅ Complete |
| 03a | Least Action Principle | ✅ Complete |
| 04 | Reinforcement Field | ✅ Complete |
| 04a | Riesz Representer | ✅ Complete |
| 05 | Particle Memory | ✅ Complete |
| 06 | MemoryUpdate Algorithm | ✅ Complete |
| 06a | Advanced Memory Dynamics | ✅ Complete |
| 07 | RF-SARSA Algorithm | ✅ Complete |
| 07a | Continuous Policy Inference | ✅ Complete |
| 07b | RF-Q-Learning and Convergence | ✅ Complete |
| 07c | Experience Replay and Particle Memory | ✅ Complete |
| **08** | **Soft State Transitions** | ⏳ Next |
| **09** | **POMDP Interpretation** | ⏳ Planned |
| **10** | **Practical Implementation** | ⏳ Planned |

**Priority:** Complete Chapters 08-10. The February 2026 target for 07-10 was met for 07 (with three supplements beyond plan) and missed for 08-10; a re-plan is due.

---

### Part II: Emergent Structure & Spectral Abstraction

**Status:** 🎯 Foundation laid, rewrite needed

**Original Section V topics:**

- Functional clustering in RKHS
- Spectral concept discovery
- Hierarchical policy organization
- Experience compression

**New formalization available:**

- [Chapter 5: Concept Projections and Measurements](GRL0/quantum_inspired/05-concept-projections-and-measurements.md) provides mathematical foundation

**Next Steps:**

- Rewrite Section V content using concept subspace formalism
- Add tutorial chapters on spectral clustering algorithms
- Implement concept discovery in code

**Priority:** Start after Part I complete (March 2026)

---

## Quantum-Inspired Extensions

**Status:** 🔬 Advanced topics — 11 chapters complete  
**Goal:** Explore mathematical connections to QM and novel probability formulations

### Completed Chapters

| Chapter | Title | Key Contribution |
|---------|-------|------------------|
| 01 | RKHS-Quantum Parallel | Structural mapping between RKHS and QM Hilbert spaces |
| 01a | Wavefunction Interpretation | State vector vs. wavefunction clarity |
| 02 | RKHS Basis and Amplitudes | Why GRL doesn't need Born rule normalization |
| 03 | Complex-Valued RKHS | Interference effects, phase semantics |
| 04 | Action and State Fields | Slicing $Q^+$ into projections |
| 05 | Concept Projections | Formal subspace theory (foundation for Part II) |
| 06 | Agent State & Belief Evolution | What "the state" is in GRL |
| 07 | Learning Beyond GP | Alternative learning mechanisms |
| 08 | Memory Dynamics | Formation, consolidation, retrieval |
| 09 | Path Integrals and Action Principles | Path-integral formulation and genuine quantum interference |
| 10 | Complex Kernels Tutorial | Worked patterns for complex-valued kernels |

### Future Directions from These Chapters

**1. Amplitude-Based Reinforcement Learning** 🔥 **High Priority**

**Motivation:** QM's probability amplitude formulation hasn't been applied to RL

**Proposal:**

- Complex-valued value functions: $Q^+(z) \in \mathbb{C}$
- Policy from Born rule: $\pi(a|s) \propto |Q^+(s,a)|^2$
- Phase semantics: temporal, directional, or contextual structure
- Interference-based TD updates

**Next Steps:**

- Expand Chapter 03 with detailed algorithms
- Design toy problems where phase helps
- Implement and benchmark

**Target:** Paper submission NeurIPS 2026 or ICML 2027

---

**2. Information-Theoretic Memory Consolidation** 🔥 **High Priority**

**Motivation:** Replace hard threshold $\tau$ in MemoryUpdate with principled criteria

**Proposal:**

- MDL objective: $\min_{\Omega'} \text{TD-error}(Q^+(\Omega')) + \lambda |\Omega'|$
- Surprise-gated consolidation: store distinct if high TD-error, merge if low
- Adaptive top-k neighbors (density-aware)

**Next Steps:**

- Implement MDL consolidation algorithm
- Compare: hard threshold vs. soft vs. top-k vs. MDL vs. surprise-gating
- Meta-learn consolidation parameters

**Target:** Paper submission ICML 2026 or NeurIPS 2027

---

**3. Mixture of Experts with Concept-Based Gating**

**Motivation:** Multiple local fields for scalability and modularity

**Proposal:**

- $Q^+(z) = \sum_m g_m(z) Q_m(z)$
- Gate by concept activation: $g_m(z) \propto \|P_{\mathcal{C}_m} k(z, \cdot)\|^2$
- Each expert specializes on a concept subspace

**Next Steps:**

- Connect to Part II concept discovery
- Implement hierarchical composition
- Benchmark on multi-task environments

**Target:** Part of larger hierarchical RL paper (2027)

---

**4. Hybrid Neural-Particle Architecture**

**Motivation:** Combine scalability of deep nets with fast adaptation of particles

**Proposal:**

- $Q^+(z) = Q_\theta(z) + \beta \sum_{i \in \text{recent}} w_i k(z_i, z)$
- Neural net: long-term memory (slow updates)
- Particle buffer: short-term memory (fast updates, bounded)

**Next Steps:**

- Implement distillation from buffer to network
- Large-scale continuous control experiments
- Compare to pure GP and pure neural baselines

**Target:** Practical algorithms paper (2027)

---

## GRL Extensions: Papers A, B, C

**Status:** Paper A ~70% complete, B & C planned  
**Goal:** Extend GRL with operator formalism, scalable algorithms, and applications

---

### Paper A: Theoretical Foundations (Operator Framework)

**Status:** 🟢 Draft Complete (~6,500 words), Figures in Progress  
**Progress:** 70% — Theory done, figures 43%, proofs 40%, experiments underway in ssl-lab

**Core Contribution:**

- Actions as parametric operators $\hat{O}_\theta$ (not just parameter vectors)
- Operator algebra: composition, Lie groups, hierarchical skills
- Generalized Bellman equation with energy regularization
- Unification: classical RL as special case

**Current State:**

✅ **Complete:**

- Unified draft with all sections
- 3 critical figures implemented (paradigm, operator families, unification)
- Proof outlines
- Figure generation framework

🔬 **Validation experiments — in progress, in the sibling project:**

The minimum viable experiment is running in **[ssl-lab](https://github.com/pleiadian53/ssl-lab)** on the Norman 2019 cell-perturbation dataset, and is written up as [Part 4: The Minimum Viable Experiment](action_operator/04-the-minimum-viable-experiment.md). The domain was chosen because activating a gene genuinely transforms a cell's whole expression state, because held-out two-gene combinations test operator *composition* directly, and because biology independently measures the departure from additivity (genetic interaction), giving the commutator an external referent.

What it has established:

- **The exponential map is buildable.** §4.4's $A_\theta = \exp(M_\theta)$ now exists in working code. It does not exist in this repository, where the Lie-algebra parameterization is theory only and `matrix_exp` appears nowhere.
- **The bracket has two readings**, and this is the interface between the projects: $[M_A, M_B]$ measures **epistasis** under simultaneous actions (only the swap-even part is identifiable) and **path-dependence** under sequential ones (the odd part becomes observable). ssl-lab validates the first; Paper A's contribution is the second.
- **A negative headline, reported as one.** The operator ties the flow baseline ($\Delta r$ 0.645 vs 0.648, not significant) and both lose to a conditional NB-VAE (0.766). A ceiling analysis shows the transition stage had at most **0.031** of headroom, so the domain could not have distinguished a good operator from a bad one. The tie is not evidence for the linear/Koopman premise, and Paper A should not cite it as such.
- **The compositional claim is refuted in the even half.** Round 4 tested whether $\lVert [M_A, M_B] \rVert_F$ predicts measured epistasis. It does not: four tests across both splits over all 111 pairs, the endpoint at $-0.070$ in-sample ($p = 0.75$) and $-0.262$ held-out ($p = 0.87$), and two post-hoc rescues that also failed, which strengthens the negative. The diagnosed cause is **identifiability**, not biology: each generator carries 65,536 parameters constrained only to match one marginal, so the bracket is free wherever the loss is silent. ssl-lab's operator thread is closed.
- **The near-identity prior is a claim about a coordinate system, and it did not survive the encoder.** Fitting the response wants $\lVert M_g \rVert_F \approx 12$ ($\Delta r$ 0.629); a bracket inside the regime where the theorem applies wants $\approx 1$ ($\Delta r$ 0.497). The operator cannot do both and pays 0.13 to buy validity. "Effects are small shifts on a large baseline" is true of expression coordinates and false in latent ones. **This lands directly on Paper A**: §4.4's exponential map and §4.2's least-action energy both assume near-identity without saying near-identity of *what space*. Measure it, do not assume it. The $\lVert M \rVert$ gate that caught this is the round's most portable artifact; it nearly reported a fake refutation without it.
- **Still untested: the odd half**, i.e. the raw bracket as path-dependence under sequential actions. That is Paper A's own claim, cell perturbation structurally cannot reach it ($T = 1$, no clock), and no experiment anywhere has tried it. See Part 4 §5.4 for which of the failure causes travel to a sequential domain and which do not.

⏳ **Remaining Work:**

- 4 additional figures (energy landscapes, composition, convergence, policy viz)
- Expand proofs (Appendices A)
- Operator catalog (Appendix B)
- Related work section
- Decide how Paper A reports the refuted even-half result. It is a genuine finding and belongs in the paper; the question is whether it lands here or in Paper B/C (see the tension noted above about whose job experiments are)
- If the bracket returns as a training signal, it needs a parameterization that constrains it (a shared low-rank basis confining brackets to $\mathrm{span}([B_i, B_j])$), pre-registered as a new experiment rather than bolted on as a rescue
- A sequential-action experiment, where the odd part of the bracket is observable — this is Paper A's own claim to test, and ssl-lab's $T = 1$ setting cannot reach it

**Timeline:**

- **January-February 2026:** Complete all figures, expand appendices
- **March 2026:** Run experiments, related work, final polish
- **April 2026:** Submit to NeurIPS/ICML 2026

**Timeline note (2026-07-16):** the dates above have slipped and are retained as the original plan of record. The experimental track moved to ssl-lab, which is where the operator work has actually progressed. A re-plan is due.

**Location:** `dev/papers/paper-a-theory/` · refactor plan: `dev/planning/action_operator/grl-4-ssl-lab-refactor-plan.md`

---

### Paper B: Algorithms & Implementation

**Status:** ⏳ Planned  
**Target:** ICML/NeurIPS 2026 (Applied Track)

**Planned Content:**

- Operator-Actor-Critic (OAC) algorithm
- Neural operator architectures (DeepONet, FNO integration)
- Training stability techniques
- Benchmark results (continuous control, physics tasks)
- Ablation studies

**Dependencies:**

- Requires Paper A theory finalized
- Requires implementation of `src/grl/operators/` and `src/grl/algorithms/`

**Timeline:**

- **February-March 2026:** Algorithm development
- **April-May 2026:** Benchmarking
- **June 2026:** Draft and submit

---

### Paper C: Empirical Applications

**Status:** ⏳ Planned  
**Target:** CoRL/IROS 2026 (Robotics/Applications)

**Planned Content:**

- Real-world robotic manipulation
- Fluid control problems
- PDE-governed systems
- Physics-based environments
- Interpretability analysis (energy landscapes, concept activation)

**Dependencies:**

- Requires Paper B algorithms
- Requires mature implementation

**Timeline:**

- **April-June 2026:** Application experiments
- **July 2026:** Draft and submit (CoRL deadline)

---

## Implementation Roadmap

**Status:** 🔄 Basic structure in place, algorithms pending  
**Goal:** Reference implementation of GRL v0, classical RL recovery, and modern applications

---

### Environment Simulation Package

**Vision:** Comprehensive environment package supporting:

1. **Validation**: Classical RL baselines (CartPole, Pendulum, MuJoCo)
2. **Strategy**: Modern applications (RLHF, prompt optimization, molecule design)
3. **Innovation**: GRL-native domains (physics simulation, field control)

**Package Structure:**

```
src/grl/envs/
├── validation/              # Tier 1: Reproduce classical RL
│   ├── nav2d.py            # 2D navigation (Priority 7) ⭐⭐
│   ├── cartpole.py         # DQN validation
│   ├── pendulum.py         # SAC validation
│   └── mujoco_envs.py      # Robotics (Ant, Humanoid)
│
├── strategic/               # Tier 2: Modern RL applications 🔥
│   ├── llm_finetuning.py   # RLHF for LLMs ⭐⭐⭐ HIGHEST PRIORITY
│   ├── prompt_opt.py       # Prompt optimization
│   ├── molecule_design.py  # Drug discovery
│   └── nas.py              # Neural Architecture Search
│
├── novel/                   # Tier 3: GRL-native applications
│   ├── physics_sim.py      # Force field control
│   ├── fluid_control.py    # PDE-governed systems
│   ├── image_editing.py    # Parametric transforms
│   └── multi_robot.py      # Multi-agent coordination
│
├── wrappers/                # Adapters for existing environments
│   ├── gym_wrapper.py      # OpenAI Gym → GRL
│   ├── gymnasium_wrapper.py# Gymnasium → GRL
│   ├── dm_control_wrapper.py # DeepMind Control → GRL
│   └── rlhf_wrapper.py     # TRL/transformers → GRL
│
└── scenarios/               # Predefined configurations
    ├── paper_scenarios.py   # Original paper scenarios
    ├── benchmark_suite.py   # Standard benchmarks
    └── tutorials.py         # Teaching examples
```

**Key Design Principles:**

- **Wrappers** enable GRL on **any existing RL environment**

- **Strategic environments** target commercially relevant problems
- **Scenarios** provide reproducible experiments

---

## Implementation Roadmap

---

### Phase 1: GRL v0 Baseline (Current)

**Target:** March 2026

**Modules:**

| Module | Status | Description |
|--------|--------|-------------|
| `grl/kernels/` | ✅ | Kernel functions (RBF, Matern, etc.) |
| `grl/particles/` | 🔄 | Particle memory management |
| `grl/fields/` | 🔄 | Reinforcement field computation |
| `grl/algorithms/memory_update.py` | ⏳ | MemoryUpdate algorithm |
| `grl/algorithms/rf_sarsa.py` | ⏳ | RF-SARSA algorithm |
| `grl/envs/` | 🔄 | Test environments |

**Priority Tasks:**

1. Complete MemoryUpdate implementation
2. Implement RF-SARSA
3. Add sparse GP variants
4. Create tutorial notebooks

---

### Phase 2: Scalable Learning (March-April 2026)

**Goal:** Implement alternative learning mechanisms from Chapter 07

**Tasks:**

- Kernel ridge regression
- Online SGD on weights
- Sparse methods (LASSO, inducing points)
- Hybrid (neural + particle)

---

### Phase 3: Memory Dynamics (April-May 2026)

**Goal:** Implement principled memory consolidation from Chapter 08

**Tasks:**

- Soft association (no hard threshold)
- Top-k adaptive neighbors
- MDL consolidation
- Surprise-gated formation
- Memory type tags (factual/experiential/working)

---

### Phase 4: Operator Framework (May-July 2026)

**Goal:** Implement Paper A operator formalism

**Tasks:**

- Operator base classes
- Operator families (affine, field, kernel, neural)
- Composition and algebra
- Operator-Actor-Critic (OAC)

---

### Phase 5: Applications (August 2026+)

**Goal:** Benchmarks and real-world demos

**Tasks:**

- Continuous control tasks
- Physics-based simulations
- Robotic manipulation (if hardware available)
- Transfer learning experiments

---

## Documentation Structure

**Goal:** Multi-level documentation for different audiences

---

### Public Documentation (`docs/`)

**Tutorial Papers:**

- `docs/GRL0/tutorials/` — Part I: Particle-Based Learning
- `docs/GRL0/quantum_inspired/` — Advanced topics (QM connections)
- `docs/GRL0/paper/` — Suggested edits for original paper
- `docs/GRL0/implementation/` — Implementation notes

**Future:**

- `docs/theory/` — Operator formalism theory (from Paper A)
- `docs/algorithms/` — Training algorithms (from Paper B)
- `docs/tutorials/` — Quick start guides

---

### Private Development (`dev/`)

**Current Work:**

- `dev/GRL0/` — Private notes for GRL v0 development
- `dev/papers/` — Paper drafts (A, B, C)
- `dev/GRL_extensions/` — Extension ideas
- `dev/references/` — Original paper, related papers

---

### Code Documentation

- README files in `src/grl/` subdirectories
- Docstrings in code (NumPy style)
- Tutorial notebooks in `notebooks/`
- API reference (Sphinx, future)

---

## Research Themes & Connections

### Theme 1: Functional Learning

**Across all work:**

- State as function $Q^+ \in \mathcal{H}_k$
- Operators on function spaces
- RKHS as mathematical foundation

**Papers:** GRL v0, Paper A, quantum-inspired extensions

---

### Theme 2: Particle-Based Inference

**Key insight:** Weighted particles as basis for belief state

**Papers:** GRL v0, memory dynamics (Chapter 08)

**Extensions:**

- Sparse approximations
- Hierarchical particles
- Nonparametric clustering

---

### Theme 3: Energy-Based Learning

**Key insight:** Energy function $E(z) = -Q^+(z)$ connects to physics

**Papers:** GRL v0 (Chapter 03), Paper A (least action principle)

**Extensions:**

- Hamilton-Jacobi-Bellman PDEs
- Conservative vector fields
- Lagrangian mechanics for policy

---

### Theme 4: Hierarchical Abstraction

**Key insight:** Concepts as subspaces in function space

**Papers:** GRL v0 Part II, concept projections (Chapter 05), MoE (Chapter 07)

**Extensions:**

- Multi-scale representations
- Transfer learning via shared basis
- Compositional behaviors

---

### Theme 5: Quantum-Inspired Probability

**Key insight:** Amplitude-based learning with phase and interference

**Papers:** Quantum-inspired chapters (01-04), potential standalone paper

**Extensions:**

- Complex RKHS for RL
- Born rule for action selection
- Spectral methods for concept discovery

---

## Strategic Applications: Demonstrating GRL's Generality

**Goal:** Show that GRL **subsumes** classical RL and applies to modern, commercially relevant problems.

---

### Application 1: Recovering Classical RL 🔥 **Critical for Adoption**

**Motivation:** Researchers trust frameworks that generalize what they already know.

**Objective:** Demonstrate that Q-learning, DQN, PPO, SAC, RLHF are **special cases** of GRL.

**Deliverables:**

- **Document:** [Recovering Classical RL from GRL](GRL0/recovering_classical_rl.md) ✅ Complete
- **Implementation:** Wrappers for Gym/Gymnasium environments
- **Validation:** Reproduce classical results (±5% performance)
  - Q-learning on GridWorld
  - DQN on CartPole
  - SAC on Pendulum
  - PPO on continuous control

**Timeline:** Q2 2026

**Impact:** 

- Convinces classical RL researchers GRL is not alien
- Provides clear migration path from classical to GRL
- Enables GRL to leverage existing benchmarks

---

### Application 2: RLHF for LLMs (Theoretical Connection + Future Direction)

**Status:** Theoretical connection established, implementation exploratory

**Why This Matters:**

- **Validation:** RLHF (ChatGPT, Claude, Llama) is most commercially important RL application
- **Familiarity:** Most ML researchers understand this problem
- **Generality:** If GRL generalizes RLHF theoretically, it validates framework's breadth

**Theoretical Formulation:**

- State: $s_t$ = (prompt, partial response)
- Action: $\theta_t$ = token ID (discrete action space)
- Augmented space: $(s_t, \theta_t)$
- Field: $Q^+(s_t, \theta_t)$ = expected reward for token $\theta_t$ in context $s_t$
- **Key insight:** Standard RLHF (PPO) is GRL with discrete actions + neural network approximation

**Documentation:**

- [Recovering Classical RL from GRL](GRL0/recovering_classical_rl.md) — Section 6 covers RLHF

**Potential Advantages** (Theoretical):

1. Off-policy learning (replay buffer of human feedback)
2. Kernel generalization (transfer across prompts)
3. Uncertainty quantification (exploration where uncertain)
4. Interpretability (energy landscapes)

**However:** These are speculative without empirical validation.

---

**Implementation Reality:**

**Challenges:**

- Infrastructure complexity (reward model, human feedback data)
- Computational cost (expensive even for GPT-2)
- Integration with existing tools (TRL, transformers, accelerate)
- Validation difficulty (matching PPO requires extensive tuning)

**Estimated Effort:** 6-12 months of focused work with GPU resources

**When to Pursue:**

- ✅ After GRL validated on simpler environments
- ✅ If collaborators or funding available
- ✅ If clear path to demonstrating advantages

**Realistic Alternative:**

- Write theoretical articles justifying the connection
- Toy RLHF-like problem (not real LLM) as proof-of-concept
- Wait for opportunities (industry collaboration, research grant)

---

### Application 3: Additional Modern RL Domains

**Prompt Optimization:**

- Parametric prompt generation (continuous in embedding space)
- GRL learns smooth prompt space
- Transfer across tasks

**Molecule Design:**

- Parametric molecular operators
- GRL discovers optimal molecules for drug properties
- Physics-informed kernels

**Neural Architecture Search:**

- Compositional architecture operators
- GRL explores architecture space efficiently
- Uncertainty-guided search

**Timeline:** Q4 2026+

---

## Potential Novel Contributions (Publishable)

### High-Priority Contributions

**1. Amplitude-Based Reinforcement Learning** 🔥 **Top Priority**

- **Novelty:** First RL with complex-valued value functions
- **Venue:** NeurIPS/ICML 2026-2027
- **Readiness:** 30% (theory done, needs implementation)

**2. Information-Theoretic Memory Consolidation**

- **Novelty:** MDL framework for experience replay
- **Venue:** ICML/NeurIPS 2026-2027
- **Readiness:** 40% (formulation clear, needs experiments)

**3. Operator-Based GRL (Paper A)**

- **Novelty:** Actions as operators, not symbols
- **Venue:** NeurIPS/ICML 2026
- **Readiness:** 70% (draft complete, figures/experiments needed)

---

### Medium-Priority Contributions

**4. Theoretical Articles: Modern RL as Special Cases of GRL**

- **Novelty:** Justify that RLHF, prompt optimization, NAS, molecule design are GRL special cases
- **Venue:** Blog posts, workshop papers, or sections in main papers
- **Readiness:** 50% ("Recovering Classical RL" document provides template)
- **Impact:** Demonstrates GRL's generality without requiring full implementations

**5. Concept Subspaces for Hierarchical RL**

- **Novelty:** Rigorous RKHS subspace formalism
- **Venue:** ICLR/AISTATS 2027
- **Readiness:** 50% (math done, algorithms needed)

**6. Surprise-Modulated Episodic Memory**

- **Novelty:** Bio-inspired consolidation
- **Venue:** CogSci/Neural Computation 2027
- **Readiness:** 60% (theory clear, needs validation)

**7. Hybrid Neural-Particle RL**

- **Novelty:** Combining deep learning with GP memory
- **Venue:** ICLR/ICML 2027
- **Readiness:** 30% (concept clear, full implementation needed)

---

### Strategic Applications (Future Possibilities, No Timeline)

**8. GRL for LLM Fine-tuning (RLHF)**

- **Novelty:** Application of functional RL to most commercially important RL problem
- **Venue:** ICLR/NeurIPS (if pursued)
- **Readiness:** 20% (theoretical connection clear, implementation requires major resources)
- **Status:** Exploratory — pursue only if collaborators/funding available
- **Alternative:** Write theoretical articles + toy proof-of-concept

**9. Other Strategic Applications**

- Prompt optimization, molecule design, neural architecture search
- **Status:** Theoretical connections to be documented
- **Implementation:** Pick 1-2 if resources available
- **Primary Value:** Demonstrate GRL generalizes modern RL methods

---

## Timeline Summary

### 2026 Q1 (January-March)

**Focus:** Complete GRL v0 documentation and baseline implementation

- ✅ Finish Part I tutorial chapters (07-10)
- ✅ Implement MemoryUpdate and RF-SARSA
- 🔄 Run first experiments
- 🔄 Complete Paper A figures and proofs

---

### 2026 Q2 (April-June)

**Focus:** Paper A submission, Classical RL recovery, scalable algorithms

- Submit Paper A (April deadline)
- **Implement wrappers for Gym/Gymnasium** (recover classical RL)
- **Reproduce DQN on CartPole** (validation)
- **Reproduce SAC on Pendulum** (validation)
- **Document: "Recovering Classical RL from GRL"** 🔥 Strategic
- Implement alternative learning mechanisms (Chapter 07)
- Implement memory dynamics (Chapter 08)
- Draft Paper B algorithms
- Start benchmark experiments

---

### 2026 Q3 (July-September)

**Focus:** Paper B submission, novel contributions, extensions

- Submit Paper B (June ICML or September NeurIPS)
- **Explore amplitude-based RL** (if promising after Part I complete)
- **Implement MDL consolidation** (principled memory dynamics)
- **Concept-based MoE** (mixture of experts via subspaces)
- Start operator framework implementation
- Run application experiments for Paper C
- **Write theoretical articles**: How RLHF/prompt-opt/NAS are special cases of GRL

---

### 2026 Q4 (October-December)

**Focus:** Paper C submission, novel contributions (amplitude/MDL)

- Submit Paper C (CoRL deadline ~July)
- Develop amplitude-based RL fully
- Implement MDL consolidation
- Draft standalone papers on extensions

---

### 2027+

**Focus:** Consolidate results, broader impact

- Package releases and documentation
- Workshop papers and tutorials
- Integration with popular RL libraries
- Real-world applications

---

## Success Metrics

### Short-Term (6 months)

- [ ] Complete GRL v0 tutorial paper (Parts I & II)
- [ ] Reference implementation working on 3+ environments
- [ ] Submit Paper A to top venue
- [ ] At least 10 GitHub stars

---

### Medium-Term (12 months)

- [ ] Paper A accepted or under review
- [ ] Papers B & C submitted
- [ ] 2-3 additional papers on extensions (amplitude, MDL, concepts)
- [ ] 50+ GitHub stars, some external users

---

### Long-Term (24+ months)

- [ ] 3+ papers published at top venues
- [ ] GRL adopted by other researchers
- [ ] Integration with popular libraries (Stable-Baselines3, RLlib)
- [ ] Tutorial at major conference (NeurIPS, ICML)
- [ ] Real-world deployment (robotics, control systems)

---

## Open Questions & Research Opportunities

### Theoretical Questions

1. **Sample complexity:** How does GRL compare to classical RL theoretically?
2. **Convergence rates:** Can we prove faster convergence in certain settings?
3. **Operator algebra:** What's the right group structure for operator composition?
4. **Phase semantics:** What should complex phase represent in amplitude-based RL?

---

### Algorithmic Questions

1. **Scalability:** Best way to handle millions of particles?
2. **Consolidation criterion:** MDL vs. surprise-gating vs. other?
3. **Mixture of experts:** How to partition concept subspaces automatically?
4. **Transfer learning:** Can concept basis enable zero-shot transfer?

---

### Application Questions

1. **Best domains:** Where does GRL shine vs. classical RL?
2. **Interpretability:** Can energy landscapes help explain decisions?
3. **Safety:** Can concept subspaces encode constraints?
4. **Multi-agent:** How to extend GRL to multi-agent settings?

---

## Resources & References

### Key Papers (Original Work)

- Chiu & Huber (2022). *Generalized Reinforcement Learning*. arXiv:2208.04822.

### Inspirations

**Kernel Methods:**

- Rasmussen & Williams (2006). *Gaussian Processes for Machine Learning*.

**Operator Learning:**

- Lu et al. (2021). *Learning Nonlinear Operators via DeepONet*. Nature Machine Intelligence.
- Li et al. (2021). *Fourier Neural Operator*. ICLR.

**Quantum-Inspired ML:**

- Cheng et al. (2018). *Quantum Generative Adversarial Learning*. PRL.
- Havlíček et al. (2019). *Supervised Learning with Quantum-Enhanced Feature Spaces*. Nature.

**Memory & Agent Systems:**

- Cao et al. (2024). *Memory in the Age of AI Agents*. arXiv:2512.13564.

---

## Contact & Collaboration

**Documentation:** [docs/](docs/)  
**Code:** [src/grl/](src/grl/)  
**Papers:** [dev/papers/](dev/papers/)  
**Issues:** GitHub Issues (coming soon)

---

**This roadmap is a living document and will be updated as research progresses.**

**Last Updated:** January 14, 2026
