# GRL Notebooks

Interactive Jupyter notebooks for exploring GRL concepts through visualization and experimentation.

---

## 📁 Notebook Collections

### [Field Series](field_series/) ⭐ Start Here

**Understanding Reinforcement Fields: From Classical Vectors to Functional Spaces**

A progressive 3-notebook series building intuition for GRL's core concept:

| # | Notebook | Status | Description |
|---|----------|--------|-------------|
| 0 | `00_intro_vector_fields.ipynb` | ✅ Complete | Gentle intro with real-world examples (optional) |
| 1 | `01_classical_vector_fields.ipynb` | ✅ Complete | Gradient fields, rotational fields, superposition, trajectories |
| 1a | `01a_vector_fields_and_odes.ipynb` | ✅ Complete | ODEs, numerical solvers (Euler/RK4), flow matching connection |
| 2 | `02_functional_fields.ipynb` | ✅ Complete | Functions as vectors, kernels, RKHS intuition |
| 3 | `03_reinforcement_fields.ipynb` | ✅ Complete | GRL's Q⁺ field, 2D navigation domain, policy inference |

**[→ Go to Field Series](field_series/)**

---

## 📚 Series Overview (Field Series)

The series consists of **3 notebooks** that build progressively:

```
┌─────────────────────────────────────┐
│  Notebook 1: Classical Vector Fields│
│  (Concrete, 2D arrows)              │
└──────────────┬──────────────────────┘
               │ You understand: arrows at points
               ↓
┌─────────────────────────────────────┐
│  Notebook 2: Functional Fields      │
│  (Abstract, functions as vectors)   │
└──────────────┬──────────────────────┘
               │ You understand: functions at points
               ↓
┌─────────────────────────────────────┐
│  Notebook 3: Reinforcement Fields   │
│  (GRL's Q+ field in RKHS)           │
└─────────────────────────────────────┘
               │ You understand: GRL's learning mechanism!
```

**Total time:** ~60-90 minutes  
**Prerequisites:** Basic calculus, linear algebra, Python  
**Goal:** Deep intuition for how GRL represents and learns policies

---

## 📓 Notebook 1: Classical Vector Fields

**Status:** 🔄 In Development  
**File:** `01_classical_vector_fields.ipynb`  
**Time:** ~15-20 minutes

### What You'll Learn

1. **Definition:** What is a vector field? (arrows at each point)
2. **Gradient fields:** Following "uphill" directions for optimization
3. **Rotational fields:** Circular flows (curl, vorticity)
4. **Superposition:** Combining multiple fields vectorially
5. **Trajectories:** Following a field to find extrema

### Key Concepts

- Vector field definition and visualization
- Potential functions and gradients
- Connection to optimization (gradient descent/ascent)
- Quiver plots, streamlines, contour maps

### Why It Matters

Vector fields provide the **concrete intuition** needed before moving to the abstract world of functional fields. Once you understand how arrows at each point create a field, you're ready to understand how **functions** at each point create GRL's reinforcement field!

### Visualizations

- ✅ Linear fields (radial patterns)
- ✅ Gradient fields (parabolic bowl, 3D surface)
- 🔄 Rotational fields (circular flow, vortex)
- 🔄 Combined fields (superposition)
- 🔄 Particle trajectories (following the field)

---

## 📓 Notebook 2: Functional Fields

**Status:** 📋 Planned  
**File:** `02_functional_fields.ipynb`  
**Time:** ~20-25 minutes

### What You'll Learn

1. **Functions as vectors:** Addition, scaling, inner products
2. **RKHS foundations:** What is a reproducing kernel Hilbert space?
3. **Explicit comparison:** Classical vectors vs. functional vectors
4. **Functional gradients:** Riesz representers and optimization in function space
5. **Kernel methods:** Similarity and generalization

### Key Concepts

- Infinite-dimensional vector spaces
- Inner products on functions: $\langle f, g \rangle = \int f(x) g(x) dx$
- Basis functions and expansions
- Kernels as generalized dot products
- RKHS: the mathematical foundation of GRL

### Why It Matters

This is the **conceptual bridge** from classical fields to GRL. Understanding that functions behave like vectors (they can be added, scaled, projected) is essential for grasping how GRL's reinforcement field operates.

### Special Section: "What IS a Functional Field?"

**Explicit Comparison:**

| Classical Vector Field | Functional Field |
|------------------------|------------------|
| Point → Arrow (2D/3D) | Point → Function |
| $\mathbf{v} = [v_x, v_y]$ | $f(\cdot) \in \mathcal{H}$ |
| Inner product: $\mathbf{v} \cdot \mathbf{w}$ | Inner product: $\langle f, g \rangle_{\mathcal{H}}$ |
| Gradient: $\nabla V$ | Functional gradient: $\nabla_f J[f]$ |

**Example:** At each point in augmented space $(s, a)$, instead of a 2D arrow, you have an **entire function** representing expected future rewards!

### Visualizations

- 📋 Functions as vectors (Gaussians, polynomials, combinations)
- 📋 Inner products and orthogonality
- 📋 Kernel functions (RBF, Matérn)
- 📋 Projection onto function subspaces
- 📋 Building Q⁺ from basis functions

---

## 📓 Notebook 3: Reinforcement Fields in GRL

**Status:** 📋 Planned  
**File:** `03_reinforcement_fields_grl.ipynb`  
**Time:** ~25-30 minutes

### What You'll Learn

1. **Augmented space:** Why $z = (s, a)$? State-action joint representation
2. **Particle memory:** Experience as weighted points in RKHS
3. **Field emergence:** How $Q^+(z) = \sum_i w_i k(z, z_i)$ creates the field
4. **Policy inference:** Reading the field to choose actions
5. **Memory update:** How new experiences reshape the field (learning!)

### Key Concepts

- Augmented state-action space
- Particle representation: $\{(z_i, w_i)\}_{i=1}^N$
- Kernel superposition: each particle creates a "bump"
- Q⁺ as an energy landscape
- Gradient-based action selection
- MemoryUpdate algorithm visualization

### Why It Matters

This notebook **brings it all together**! You'll see how:
- Classical intuition (vector fields) →
- Mathematical framework (functional fields) →
- **Actual GRL learning** (reinforcement fields)

By the end, you'll understand why GRL doesn't need explicit policy networks — the policy **emerges** from the field!

### Visualizations

- 📋 2D augmented space (state × action)
- 📋 Particle influence (single bump → complex field)
- 📋 Q⁺ landscape (3D energy surface)
- 📋 Policy inference (action landscape at fixed state)
- 📋 Before/after memory update (field reshaping)
- 📋 Boltzmann policy (exploration via temperature)

---

## 🚀 Getting Started

### Viewing Options

#### **Option 1: GitHub Pages (Best Rendering)** ⭐ **Recommended**

View rendered notebooks with proper math and plots:
- 📊 [Notebook 1: Classical Vector Fields](https://pleiadian53.github.io/GRL/notebooks/01_classical_vector_fields/)
- 📊 [Notebook 2: Functional Fields](https://pleiadian53.github.io/GRL/notebooks/02_functional_fields/) (Coming soon)
- 📊 [Notebook 3: Reinforcement Fields](https://pleiadian53.github.io/GRL/notebooks/03_reinforcement_fields_grl/) (Coming soon)

**Advantages:**
- ✅ Reliable rendering (no GitHub timeouts)
- ✅ Math properly displayed
- ✅ Plots and outputs preserved
- ✅ Mobile-friendly

#### **Option 2: Run Locally (Interactive)**

Clone and run in Jupyter:

```bash
# Clone repository
git clone https://github.com/pleiadian53/GRL.git
cd GRL/notebooks

# Create environment (recommended)
conda env create -f ../environment.yml
conda activate grl

# Or install dependencies
pip install numpy matplotlib seaborn jupyter

# Launch Jupyter
jupyter notebook
```

**Advantages:**
- ✅ Fully interactive
- ✅ Modify and experiment
- ✅ Add your own examples
- ✅ Optional interactive widgets (sliders, etc.)

#### **Option 3: GitHub.com (Quick View)**

Browse notebooks directly on GitHub:
- 📓 [View on GitHub](https://github.com/pleiadian53/GRL/tree/main/notebooks)

**Note:** GitHub's notebook renderer can be slow/unreliable for large notebooks. Use GitHub Pages for best experience.

---

## 📖 Learning Paths

### **Path 1: Visual Intuition (Notebooks Only)**

Just want to see it work? Go through the notebooks in order:
1. → Notebook 1 (classical fields)
2. → Notebook 2 (functional fields)
3. → Notebook 3 (GRL fields)

**Time:** ~60-90 minutes  
**Depth:** Intuitive understanding, ready to use GRL

---

### **Path 2: Deep Understanding (Notebooks + Tutorials)**

Want mathematical rigor? Combine notebooks with tutorials:

1. 📓 **Notebook 1** (classical fields) → 📚 No specific tutorial needed
2. 📓 **Notebook 2** (functional fields) → 📚 [Tutorial Ch 2: RKHS Foundations](https://pleiadian53.github.io/GRL/GRL0/tutorials/02-rkhs-foundations/)
3. 📓 **Notebook 3** (GRL fields) → 📚 [Tutorial Ch 4: Reinforcement Field](https://pleiadian53.github.io/GRL/GRL0/tutorials/04-reinforcement-field/)
4. 📓 **Notebook 3** (memory update) → 📚 [Tutorial Ch 6: MemoryUpdate](https://pleiadian53.github.io/GRL/GRL0/tutorials/06-memory-update/)

**Time:** ~3-4 hours  
**Depth:** Full mathematical understanding, ready to implement GRL

---

### **Path 3: Implementation-Focused (Notebooks + Code)**

Want to implement GRL? Use notebooks as context:

1. 📓 **All 3 notebooks** (build intuition)
2. 📚 [Implementation Guide](https://pleiadian53.github.io/GRL/GRL0/implementation/)
3. 💻 Study `src/grl/` codebase
4. 🧪 Run examples in `examples/`

**Time:** ~1-2 days  
**Depth:** Ready to build GRL applications

---

## 🎨 Visualization Gallery

### Coming Soon

Once all notebooks are complete, this section will showcase:
- Interactive widget demos (adjust parameters, see effects)
- Animated trajectories (particles following fields)
- 3D rotatable plots (energy landscapes)
- Comparison plots (classical RL vs. GRL)

---

## 🤝 Contributing

Found a bug? Have suggestions for additional visualizations?

1. **Open an issue:** [GitHub Issues](https://github.com/pleiadian53/GRL/issues)
2. **Suggest improvements:** What examples would help your understanding?
3. **Share your notebooks:** Built your own GRL demos? We'd love to feature them!

---

## 📚 Additional Resources

### Related Documentation

- **[GRL Tutorial Series](https://pleiadian53.github.io/GRL/GRL0/tutorials/)** — In-depth mathematical treatment
- **[Quantum-Inspired Extensions](https://pleiadian53.github.io/GRL/GRL0/quantum_inspired/)** — Advanced topics (amplitude fields, complex RKHS)
- **[Implementation Guide](https://pleiadian53.github.io/GRL/GRL0/implementation/)** — Technical specifications for coding GRL
- **[Recovering Classical RL](https://pleiadian53.github.io/GRL/GRL0/recovering_classical_rl/)** — How Q-learning, DQN, PPO emerge from GRL

### Original Paper

- **arXiv:** [Generalized Reinforcement Learning: Experience Particles, Action Operator, Reinforcement Field...](https://arxiv.org/abs/2208.04822)
- **Authors:** Po-Hsiang Chiu, Manfred Huber
- **Year:** 2022

---

## 📝 Development Status

| Notebook | Status | Cells | Visualizations | Interactive |
|----------|--------|-------|----------------|-------------|
| 01: Classical Vector Fields | 🔄 In Dev | 6/15 | 2/5 | 0/2 |
| 02: Functional Fields | 📋 Planned | 0/18 | 0/6 | 0/3 |
| 03: Reinforcement Fields | 📋 Planned | 0/20 | 0/7 | 0/3 |

**Legend:**
- ✅ Complete
- 🔄 In Development
- 📋 Planned
- ❌ Blocked/Postponed

---

## 💡 Tips for Learning

### **If you're new to vector fields:**
- Start with Notebook 1, take your time
- Try to predict what arrows will look like before running cells
- Experiment: change functions, see what happens!

### **If you're new to RKHS:**
- Read [Tutorial Ch 2](https://pleiadian53.github.io/GRL/GRL0/tutorials/02-rkhs-foundations/) alongside Notebook 2
- Don't worry about full mathematical rigor initially
- Focus on the **analogy**: functions ↔ vectors

### **If you're familiar with classical RL:**
- Jump to Notebook 3, refer back to 1-2 as needed
- Read [Recovering Classical RL](https://pleiadian53.github.io/GRL/GRL0/recovering_classical_rl/) to see connections
- Compare GRL's field-based approach to Q-networks you know

---

## ❓ FAQ

**Q: Can I skip Notebook 1 if I know vector fields?**  
A: Yes, but skim it! The GRL-specific interpretations (gradient = policy improvement) are valuable.

**Q: Is Notebook 2 mathematically rigorous?**  
A: Moderately. It's more rigorous than Notebook 1 but less than the tutorial series. Think "motivated introduction" rather than "proof-based course."

**Q: Do I need to run code locally or can I just read?**  
A: Just reading works! But running locally lets you experiment, which deepens understanding.

**Q: Are these notebooks tested?**  
A: Yes! All code is tested to run cleanly with the specified dependencies. If you encounter issues, please [open an issue](https://github.com/pleiadian53/GRL/issues).

**Q: Can I use these for teaching?**  
A: Absolutely! Licensed under MIT. Attribution appreciated. Let us know if you do — we'd love to hear about it!

---

## 🙏 Acknowledgments

**Notebook design principles inspired by:**
- Distill.pub (clarity through interactive visualization)
- 3Blue1Brown (progressive concept building)
- Jupyter Book community (reproducible computational narratives)

**Mathematical content based on:**
- Original GRL paper (Chiu & Huber, 2022)
- Kernel methods literature (Schölkopf, Smola)
- Functional analysis (Kreyszig, Rudin)

---

**Ready to start?** → [Notebook 1: Classical Vector Fields](https://pleiadian53.github.io/GRL/notebooks/01_classical_vector_fields/) 🚀

**Questions?** → [Open an issue](https://github.com/pleiadian53/GRL/issues) or [read the full tutorials](https://pleiadian53.github.io/GRL/GRL0/tutorials/)
