# Field Series Roadmap

**Building GRL Understanding Systematically**

This document tracks the progression from foundational concepts to complete GRL algorithms.

---

## ✅ Completed: Foundation (Notebooks 0-3)

### Notebook 0: Introduction to Vector Fields
**Status**: Complete  
**Topics**: Real-world examples, basic intuition  
**Time**: ~10-15 minutes

### Notebook 1: Classical Vector Fields
**Status**: Complete  
**Topics**: 
- Vector field definition and visualization
- Gradient fields (connection to optimization)
- Rotational fields and curl
- Superposition of fields
- Trajectories following gradients

**Time**: ~20-25 minutes

### Notebook 1a: Vector Fields and ODEs
**Status**: Complete  
**Topics**:
- ODEs as following vector fields ($\dot{x} = F(x)$)
- Numerical solvers (Euler, RK4)
- Phase portraits and fixed points
- Gradient flow (optimization as ODE)
- Connection to flow matching (genai-lab)

**Time**: ~25-30 minutes

### Notebook 2: Functional Fields
**Status**: Complete  
**Topics**:
- Functions as infinite-dimensional vectors
- Kernel functions and similarity
- RKHS intuition
- Functional gradients
- Superposition in function space

**Time**: ~20-25 minutes

### Notebook 3: Reinforcement Fields
**Status**: Complete  
**Topics**:
- Augmented state-action space: $z = (s, \theta)$
- Particle memory: $\{(z_i, w_i)\}$
- Field emergence: $Q^+(z) = \sum_i w_i k(z, z_i)$
- **Basic policy inference**: $\theta^* = \arg\max_\theta Q^+(s, \theta)$ (discrete search)
- Obstacles via negative particles

**Time**: ~30 minutes

**Supplementary**:
- `03a_particle_coverage_effects.ipynb` — Visual proof of particle coverage effects
- `particle_vs_gradient_fields.md` — Theory comparison

---

## 🚧 In Progress / Planned: Learning Algorithms

### Notebook 4: Policy Inference (Planned)

**Goal**: Deep dive into how agents extract policies from the Q⁺ field

**Topics to Cover**:

1. **Greedy Policy** (already introduced in Notebook 3)
   - Discrete action search: $\theta^* = \arg\max_\theta Q^+(s, \theta)$
   - Computational considerations (number of angles)
   - Limitations of discrete search

2. **Gradient-Based Policy** (new)
   - Continuous optimization: $\nabla_\theta Q^+(s, \theta) = 0$
   - Gradient ascent on action space
   - Connection to policy gradient methods

3. **Boltzmann (Soft) Policy** (new)
   - Exploration via softmax: $\pi(\theta|s) \propto \exp(\beta Q^+(s, \theta))$
   - Temperature parameter $\beta$
   - Entropy regularization

4. **Action Landscapes** (expand from Notebook 3)
   - Visualizing $Q^+(s, \cdot)$ for fixed states
   - Multi-modal action distributions
   - Local vs. global optima

**Visualizations**:
- Polar plots of action landscapes at different temperatures
- Comparison: greedy vs. Boltzmann sampling
- Interactive sliders for temperature $\beta$

**Time Estimate**: ~25-30 minutes

**Prerequisites**: Notebook 3

---

### Notebook 5: Memory Update — Learning from Experience (Planned)

**Goal**: Understand how the field evolves as the agent learns

**Topics to Cover**:

1. **Single Particle Addition**
   - New experience: $(s, a, r)$
   - Creating particle: $(z_{new}, w_{new})$ where $z_{new} = (s, a)$
   - Weight assignment: $w_{new} = f(r, \gamma, ...)$

2. **Field Evolution**
   - Before/after comparison
   - Difference map: $\Delta Q^+ = Q^+_{after} - Q^+_{before}$
   - "Ripple" effect from new particle
   - Kernel lengthscale controls influence radius

3. **MemoryUpdate Algorithm**
   - Pseudocode walkthrough
   - When to add positive vs. negative particles
   - Memory management (capacity limits)

4. **Interactive Demonstration**
   - Click to add particles
   - See field update in real-time
   - Observe policy changes

**Visualizations**:
- Side-by-side: Q⁺ before/after adding particle
- Heatmap of $\Delta Q^+$
- Animated field evolution over multiple updates
- Policy vector field changes

**Code Examples**:
```python
def add_particle(particles, z_new, w_new):
    """Add a new particle to memory."""
    particles.append({'z': z_new, 'w': w_new})
    return particles

def compute_field_difference(X, Y, particles_before, particles_after):
    """Visualize how field changed."""
    Q_before = compute_Q_field(X, Y, particles_before)
    Q_after = compute_Q_field(X, Y, particles_after)
    return Q_after - Q_before
```

**Time Estimate**: ~30-35 minutes

**Prerequisites**: Notebooks 3, 4

---

### Notebook 6: RF-SARSA — Complete Learning Algorithm (Planned)

**Goal**: Implement and understand the full GRL learning algorithm

**Topics to Cover**:

1. **SARSA Recap**
   - Classical SARSA: $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma Q(s',a') - Q(s,a)]$
   - TD error and bootstrapping

2. **RF-SARSA Adaptation**
   - No explicit Q-table — field represents Q-function
   - TD error in RKHS: $\delta = r + \gamma Q^+(s', a') - Q^+(s, a)$
   - Particle weight from TD error: $w_{new} = \alpha \delta$

3. **Algorithm Walkthrough**
   - Initialize: empty particle memory
   - Episode loop:
     - Select action via policy (Boltzmann or greedy)
     - Execute, observe $(s', r)$
     - Compute TD error
     - Add particle if $|\delta| > \epsilon$
   - Field emerges from accumulated particles

4. **Convergence and Stability**
   - When does the field stabilize?
   - Memory growth over time
   - Particle pruning strategies

**Visualizations**:
- Episode-by-episode field evolution (animated)
- TD error over time
- Number of particles vs. episodes
- Final learned policy vs. optimal policy

**Code Examples**:
```python
def rf_sarsa_episode(env, particles, alpha, gamma, beta):
    """Run one episode of RF-SARSA."""
    s = env.reset()
    a = sample_boltzmann_policy(s, particles, beta)
    
    for t in range(max_steps):
        s_next, r, done = env.step(a)
        a_next = sample_boltzmann_policy(s_next, particles, beta)
        
        # TD error
        Q_sa = compute_Q_plus(s, a, particles)
        Q_next = compute_Q_plus(s_next, a_next, particles)
        delta = r + gamma * Q_next - Q_sa
        
        # Add particle if significant
        if abs(delta) > epsilon:
            z_new = (s, a)
            w_new = alpha * delta
            particles.append({'z': z_new, 'w': w_new})
        
        s, a = s_next, a_next
        if done: break
    
    return particles
```

**Experiments**:
- 2D navigation (from Notebook 3)
- Gridworld
- Mountain car (continuous actions)

**Time Estimate**: ~40-45 minutes

**Prerequisites**: Notebooks 3, 4, 5

---

## 🔮 Future Topics (Beyond Core Series)

### Advanced Topics (Potential Notebooks 7+)

1. **Kernel Design and Selection**
   - RBF vs. other kernels
   - Adaptive lengthscales
   - State-action factorization

2. **Scalability and Approximations**
   - Particle pruning
   - Sparse approximations
   - Nyström methods

3. **Multi-Task and Transfer Learning**
   - Shared particle memories
   - Task-specific fields
   - Meta-learning

4. **Theoretical Foundations**
   - Convergence proofs
   - Sample complexity
   - Relationship to kernel-based RL

5. **Comparison with Other Methods**
   - GRL vs. DQN
   - GRL vs. SAC
   - GRL vs. PPO
   - When to use GRL?

---

## Development Principles

**Systematic Progression**:
1. ✅ Build intuition (Notebooks 0-3)
2. 🚧 Understand components (Notebooks 4-5)
3. 🔮 Implement algorithms (Notebook 6)
4. 🔮 Explore advanced topics (Notebooks 7+)

**Each Notebook Should**:
- Build on previous concepts
- Include professional visualizations
- Provide working code examples
- Connect theory to practice
- Take 20-45 minutes to complete

**Pedagogical Goals**:
- Visual > Mathematical (when possible)
- Interactive > Static (when useful)
- Synthetic > Real (for clarity, then real for validation)
- Incremental > Comprehensive (build up systematically)

---

## Timeline and Priorities

### High Priority (Core Understanding)
- [ ] Notebook 4: Policy Inference
- [ ] Notebook 5: Memory Update
- [ ] Notebook 6: RF-SARSA

### Medium Priority (Practical Application)
- [ ] Integration with real RL environments
- [ ] Performance benchmarks
- [ ] Hyperparameter tuning guide

### Low Priority (Advanced Topics)
- [ ] Theoretical deep dives
- [ ] Comparison studies
- [ ] Extensions and variants

---

## Related Resources

**Within GRL Project**:
- Tutorial series: `docs/GRL0/tutorials/`
- Theory documents: `docs/theory/`
- Implementation: `src/` (when available)

**External Projects**:
- [genai-lab](https://github.com/pleiadian53/genai-lab) — Flow matching, diffusion models
- Original GRL paper: [arXiv:2208.04822](https://arxiv.org/abs/2208.04822)

---

**Last Updated**: January 15, 2026

**Status**: Foundation complete (Notebooks 0-3), planning next phase (Notebooks 4-6)
