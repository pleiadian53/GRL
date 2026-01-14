# GRL-v0 Implementation Guide

**Purpose**: Technical specifications for implementing GRL-v0  
**Audience**: Developers ready to implement GRL  
**Prerequisites**: Familiarity with tutorial chapters

---

## Overview

This directory provides implementation specifications for GRL-v0. Each component is documented with:

- Theoretical foundation
- Interface design
- Implementation details
- Testing strategy

---

## Architecture Overview

GRL-v0 is organized into **four layers** spanning both Part I (Reinforcement Fields) and Part II (Emergent Structure):

```
┌─────────────────────────────────────────────────────────────────┐
│         Layer 4: Abstraction (Part II: Emergent Structure)      │
│  ┌─────────────────────────┐  ┌─────────────────────────────┐  │
│  │  Spectral Clustering    │  │   Concept Hierarchy         │  │
│  │  (Functional clusters)  │  │  (Multi-level abstraction)  │  │
│  └─────────────────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│         Layer 3: Inference (Part I: Reinforcement Fields)       │
│  ┌─────────────────────────┐  ┌─────────────────────────────┐  │
│  │    Policy Inference     │  │   Soft State Transitions    │  │
│  │  (Energy minimization)  │  │  (Distributed successors)   │  │
│  └─────────────────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│         Layer 2: Reinforcement (Part I: Reinforcement Fields)   │
│  ┌─────────────────────────┐  ┌─────────────────────────────┐  │
│  │      RF-SARSA           │  │      MemoryUpdate           │  │
│  │  (Two-layer TD system)  │  │  (Belief transition)        │  │
│  └─────────────────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│         Layer 1: Representation (Part I: Reinforcement Fields)  │
│  ┌─────────────────────────┐  ┌─────────────────────────────┐  │
│  │    Particle Memory      │  │     Kernel Functions        │  │
│  │   (Belief state Ω)      │  │     (RKHS geometry)         │  │
│  └─────────────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Part I (Layers 1-3)**: Particle-based learning, reinforcement fields, belief-state inference

**Part II (Layer 4)**: Emergent structure discovery, spectral concept formation, hierarchical control

*Based on:* [Section V of the original paper](../../references/GRL-v0.pdf) (Chiu & Huber, 2022)

---

## Implementation Specifications

### Core Infrastructure

| Spec | Component | Priority | Status |
|------|-----------|----------|--------|
| 00 | Implementation Overview | - | ⏳ Planned |
| 01 | Architecture Design | - | ⏳ Planned |

### Layer 1: Representation

| Spec | Component | Priority | Status |
|------|-----------|----------|--------|
| 02 | Particle Memory | ⭐ 1 | ⏳ Planned |
| 03 | Kernel Functions | ⭐ 2 | ⏳ Planned |

### Layer 2: Reinforcement

| Spec | Component | Priority | Status |
|------|-----------|----------|--------|
| 04 | MemoryUpdate Algorithm | ⭐ 3 | ⏳ Planned |
| 05 | RF-SARSA Algorithm | ⭐ 4 | ⏳ Planned |

### Layer 3: Inference

| Spec | Component | Priority | Status |
|------|-----------|----------|--------|
| 06 | Policy Inference | ⭐ 5 | ⏳ Planned |
| 07 | Soft State Transitions | ⭐ 6 | ⏳ Planned |

### Layer 4: Abstraction (Part II)

| Spec | Component | Priority | Status |
|------|-----------|----------|--------|
| 08 | Spectral Clustering | 🔬 1 | ⏳ Planned |
| 09 | Concept Discovery | 🔬 2 | ⏳ Planned |
| 10 | Concept Hierarchy | 🔬 3 | ⏳ Planned |
| 11 | Concept-Conditioned Policies | 🔬 4 | ⏳ Planned |

**Note:** Part II implementation begins after Part I is validated (see Priority 7 below)

### Demonstration Environment

| Spec | Component | Priority | Status |
|------|-----------|----------|--------|
| 12 | 2D Navigation Domain | ⭐⭐ 7 | ⏳ Planned |

**Note:** This is the primary environment for validating and demonstrating GRL-v0

### Supporting Components

| Spec | Component | Status |
|------|-----------|--------|
| 13 | Environment Interface | ⏳ Planned |
| 14 | Visualization Tools | ⏳ Planned |
| 15 | Testing Strategy | ⏳ Planned |
| 16 | Experiment Protocols | ⏳ Planned |

---

## Implementation Priorities

### Priority 1: Particle Memory ⭐

**Why first**: This IS the agent state. Everything else depends on it.

**Key Features**:
- Particle storage: `[(z_i, w_i)]`
- Energy queries: `E(z) = -Σ w_i k(z, z_i)`
- Association: Find similar particles
- Management: Add, merge, prune

### Priority 2: Kernel Functions

**Why second**: Defines geometry of augmented space.

**Key Features**:
- RBF kernel with ARD
- Augmented kernel: `k((s,θ), (s',θ'))`
- Gradient computation
- Hyperparameter adaptation

### Priority 3: MemoryUpdate (Algorithm 1)

**Why third**: The belief-state transition operator.

**Key Features**:
- Particle instantiation
- Kernel-based association
- Weight propagation
- Regularization

### Priority 4: RF-SARSA (Algorithm 2)

**Why fourth**: Provides reinforcement signals.

**Key Features**:
- Primitive SARSA layer
- Field GP layer
- Two-layer coupling
- ARD updates

### Priority 5: Policy Inference

**Why fifth**: How actions are selected.

**Key Features**:
- Energy-based selection
- Boltzmann sampling
- Greedy mode
- Gradient-based optimization (optional)

### Priority 6: Soft State Transitions

**Why sixth**: Emergent uncertainty from kernel overlap.

**Key Features**:
- Distributed successor states
- Transition probability from kernel
- Implicit POMDP interpretation
- Uncertainty quantification

### Priority 7: 2D Navigation Domain ⭐⭐ **Critical**

**Why seventh**: Primary validation and demonstration environment.

**Purpose**:
1. **Reproduce** the original paper (Figure 4, Section VI)
2. **Validate** all Part I components in a controlled setting
3. **Demonstrate** GRL capabilities professionally

**Key Features**:
- Continuous 2D state space
- Parametric movement actions (direction, magnitude)
- Obstacles, walls, and goals
- Energy landscape visualization
- Particle memory visualization
- Trajectory recording

**Deployment Goals**:
- **Reproducibility**: Match original paper results
- **Professionalism**: Publication-quality figures and demos
- **Accessibility**: 
  - Python API for programmatic use
  - Interactive web interface for exploration
  - Jupyter notebook tutorials
- **Extensibility**: Easy to add new scenarios

See: [2D Navigation Specification](#2d-navigation-domain-specification) below

---

## Part II Priorities (After Part I Validated)

### Priority 8: Spectral Clustering (Part II)

**Why first in Part II**: Foundation for concept discovery.

**Key Features**:
- Kernel matrix construction from particle memory
- Eigendecomposition
- Cluster identification
- Concept subspace projection (from quantum-inspired Chapter 05)

### Priority 9: Concept Discovery

**Why second**: Automated structure learning.

**Key Features**:
- Functional similarity metrics
- Automatic concept identification
- Concept naming/labeling
- Validation metrics

### Priority 10: Concept Hierarchy

**Why third**: Multi-level abstraction.

**Key Features**:
- Nested subspace structure
- Hierarchical composition
- Transfer across concepts
- Visualization

### Priority 11: Concept-Conditioned Policies

**Why fourth**: Use discovered structure.

**Key Features**:
- Policy per concept
- Concept-gated execution
- Hierarchical planning
- Abstract reasoning

---

## 2D Navigation Domain Specification

### Overview

The **2D Navigation Domain** is the primary environment for GRL-v0 validation and demonstration. Originally introduced in the paper (Figure 4, Section VI), we aim to:

1. **Reproduce** existing results with high fidelity
2. **Enhance** the domain to professional standards
3. **Deploy** as an accessible, interactive demonstration

---

### Domain Description

**State Space**: $\mathcal{S} = [0, L_x] \times [0, L_y]$ (continuous 2D position)

**Action Space**: $\mathcal{A} = \{(\theta, v) : \theta \in [0, 2\pi), v \in [0, v_{\max}]\}$
- $\theta$: Direction angle
- $v$: Speed magnitude

**Augmented Space**: $\mathcal{Z} = \mathcal{S} \times \mathcal{A}$ (4D continuous)

**Dynamics**:
$$s_{t+1} = s_t + v \cdot (\cos\theta, \sin\theta) \cdot \Delta t$$

**Obstacles**: Polygonal or circular regions (configurable)

**Goals**: Target positions with rewards

---

### Scenarios (From Original Paper)

**Scenario 1**: Simple goal-reaching
- Single goal, no obstacles
- Validate basic particle memory and policy inference

**Scenario 2**: Navigation with obstacles
- Multiple obstacles (replicating Figure 4)
- Demonstrate smooth navigation around barriers
- Show energy landscape and particle distribution

**Scenario 3**: Multi-goal task
- Multiple goals with different rewards
- Demonstrate action-state duality
- Show concept emergence (if Part II implemented)

---

### Reproduction Goals

**Figure 4 Recreation**:
- Exact environment setup from paper
- Energy landscape visualization
- Particle memory visualization
- Learned trajectory comparison

**Quantitative Metrics**:
- Success rate (reaching goal)
- Path efficiency (vs. optimal)
- Collision rate (with obstacles)
- Learning curves (episodes to convergence)

**Qualitative Assessment**:
- Smooth, natural trajectories
- Efficient obstacle avoidance
- Energy landscape interpretability

---

### Professional Enhancement

**Visual Quality**:
- Publication-ready figures (vector graphics)
- Interactive animations (mp4/gif)
- Real-time rendering (60 FPS)
- Multiple view modes:
  - Top-down environment view
  - Energy landscape heatmap
  - Particle distribution overlay
  - Trajectory history

**Code Quality**:
- Modular, extensible design
- Configuration files (YAML/JSON)
- Logging and metrics
- Reproducible random seeds

**Documentation**:
- API reference
- Tutorial notebooks
- Example scripts
- Performance benchmarks

---

### Deployment Plan

#### Phase 1: Core Implementation

**Components**:
- Environment class (`Nav2DEnv`)
- Rendering engine
- Action space handling
- Reward function

**Deliverables**:
- Python package installable via pip
- Basic visualization
- Unit tests

#### Phase 2: GRL Integration

**Components**:
- Particle memory integration
- MemoryUpdate in navigation loop
- RF-SARSA training
- Energy landscape computation

**Deliverables**:
- Training scripts
- Evaluation scripts
- Experiment configs

#### Phase 3: Professional Demo

**Components**:
- Interactive Jupyter notebooks
- Web-based interface (Flask/FastAPI + React)
- Video demonstrations
- Benchmark suite

**Deliverables**:
- Hosted web demo (e.g., Hugging Face Spaces, Streamlit)
- Tutorial video
- Blog post

---

### Web Interface Features

**Interactive Controls**:
- Place obstacles (drag-and-drop)
- Set goal positions
- Adjust GRL hyperparameters (kernel bandwidth, temperature)
- Start/stop/reset simulation

**Visualizations**:
- Real-time agent movement
- Energy landscape evolution
- Particle memory growth
- Learning curves

**Export**:
- Save trajectories
- Download figures
- Export particle memory

**Sharing**:
- Permalink to configurations
- Embed in documentation
- Public gallery of scenarios

---

### API Design

```python
from grl.envs import Nav2DEnv
from grl.agents import GRLAgent

# Create environment
env = Nav2DEnv(
    size=(10, 10),
    obstacles=[
        {"type": "circle", "center": (5, 5), "radius": 1.5},
        {"type": "polygon", "vertices": [(2, 2), (3, 2), (3, 3)]},
    ],
    goal=(9, 9),
    goal_reward=10.0,
)

# Create GRL agent
agent = GRLAgent(
    kernel="rbf",
    lengthscale=1.0,
    temperature=0.1,
)

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    done = False
    
    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state
    
    # Visualize
    if episode % 10 == 0:
        env.render(show_particles=True, show_energy=True)
        agent.save_memory(f"memory_ep{episode}.pkl")

# Evaluate
success_rate, avg_path_length = evaluate(agent, env, num_trials=100)
```

---

### Timeline

**Week 1-2**: Core environment implementation
**Week 3-4**: GRL integration and training
**Week 5-6**: Visualization and demos
**Week 7-8**: Web interface and deployment
**Week 9-10**: Documentation and tutorials

**Target**: Complete professional 2D navigation demo by **March 2026**

---

## Application Domains Beyond 2D Navigation

### Philosophy: GRL as a Generalization

**Key Insight**: Traditional RL is a **special case** of GRL when:
- Action space is discrete → GRL with fixed parametric mappings
- Action space is finite → GRL with finite operator set
- Q-learning → GRL with trivial augmentation (state only)

**Strategic Goal**: Demonstrate that GRL **subsumes** classical RL, including modern applications like:
- **RLHF for LLMs** (Reinforcement Learning from Human Feedback)
- **PPO/SAC for continuous control**
- **DQN for discrete actions**
- **Actor-critic methods**

This positions GRL not as "another RL algorithm" but as a **unifying framework** that recovers existing methods as special cases while enabling new capabilities.

---

### Priority Application Domains

#### **Tier 1: Validation Environments** (Demonstrate Correctness)

**Goal**: Show GRL recovers classical RL results

| Domain | Type | Classical Baseline | GRL Advantage | Status |
|--------|------|-------------------|---------------|--------|
| **2D Navigation** | Continuous control | N/A (novel) | Smooth generalization | ⏳ Priority 7 |
| **CartPole** | Discrete control | DQN | Continuous action variant | 📋 Planned |
| **Pendulum** | Continuous control | DDPG, SAC | Parametric torque | 📋 Planned |
| **MuJoCo Ant** | Robotics | PPO, SAC | Compositional gaits | 📋 Planned |

---

#### **Tier 2: Strategic Environments** (Demonstrate Generality)

**Goal**: Show GRL applies to modern RL problems, including LLMs

**Note:** These are **theoretical connections** with potential future implementations. Each would require significant engineering effort.

| Domain | Type | Why Important | Theoretical Connection | Implementation |
|--------|------|---------------|----------------------|----------------|
| **LLM Fine-tuning (RLHF)** | Discrete (tokens) | Massive industry relevance | Token selection as discrete action, PPO as special case | 🔬 Exploratory |
| **Prompt Optimization** | Discrete sequences | Growing field | Parametric prompt generation in embedding space | 🔬 Exploratory |
| **Molecule Design** | Graph generation | Drug discovery | Parametric molecule operators | 🔬 Exploratory |
| **Neural Architecture Search** | Discrete choices | AutoML | Compositional architecture operators | 🔬 Exploratory |

**Primary Value**: Demonstrating that GRL **theoretically generalizes** existing methods used in commercially relevant problems (RLHF, prompt tuning, etc.)

**Implementation Reality**: These are **massive undertakings** comparable to full research projects. They serve as:
- Motivation for why GRL matters
- Future directions if resources/collaborators available
- Examples in theoretical justification documents

---

#### **Tier 3: Novel Environments** (Demonstrate Unique Capabilities)

**Goal**: Show what GRL enables that classical RL cannot do easily

| Domain | Type | Novel Capability | Why GRL Shines | Status |
|--------|------|------------------|----------------|--------|
| **Physics Simulation** | Continuous fields | Apply force fields, not point forces | Operator actions on state space | 📋 Planned |
| **Fluid Control** | PDE-governed | Manipulate flow fields | Field operators, neural operators | 📋 Planned |
| **Image Editing** | High-dim continuous | Parametric transformations | Smooth action manifolds | 📋 Planned |
| **Multi-Robot Coordination** | Continuous, multi-agent | Compositional team behaviors | Operator algebra | 📋 Planned |

---

### Recovering Classical RL: A Bridge to Adoption

**Document**: `docs/GRL0/recovering_classical_rl.md` (to be created)

**Purpose**: Show step-by-step how classical RL algorithms emerge from GRL as special cases

**Contents**:

1. **Q-learning from GRL**
   - Discrete action space as fixed parametric mapping
   - Particle memory as replay buffer
   - TD update as special case of MemoryUpdate

2. **DQN from GRL**
   - Neural network Q-function as continuous approximation of particle field
   - Experience replay as particle subsampling
   - Target networks as delayed MemoryUpdate

3. **Policy Gradient (REINFORCE) from GRL**
   - Boltzmann policy from energy landscape
   - Score function gradient as field gradient
   - Baseline as energy normalization

4. **Actor-Critic (PPO, SAC) from GRL**
   - Actor = policy inference from field
   - Critic = reinforcement field itself
   - Entropy regularization as temperature parameter

5. **RLHF for LLMs from GRL**
   - Token selection as discrete action
   - Reward model as energy function
   - PPO update as special case of RF-SARSA

**Impact**: This document becomes the **key reference** for convincing classical RL researchers that GRL is not alien, but a natural generalization.

---

### LLM Fine-tuning as a GRL Application (Exploratory)

**Status**: Theoretical connection established, implementation exploratory

**Why This Is Interesting**: 
- **Relevance**: RLHF is used for ChatGPT, Claude, Llama, Gemini
- **Familiarity**: Most ML researchers understand this problem
- **Validation**: If GRL generalizes RLHF theoretically, it validates the framework's breadth

---

#### Theoretical Formulation

**State**: $s_t$ = (prompt, partial response up to token $t$)

**Action**: $a_t \in \mathcal{V}$ where $\mathcal{V}$ = vocabulary (discrete)

**GRL View**: 
- Augmented space: $(s_t, \theta_t)$ where $\theta_t$ represents token choice
- Particle memory: stores (prompt, response, reward) experiences
- Reinforcement field: $Q^+(s_t, \theta_t)$ over semantic embedding
- Policy inference: Sample from Boltzmann over $Q^+$

**Key Insight**: Standard RLHF (PPO) is GRL with:
- Discrete action space (tokens)
- Neural network approximation of field
- On-policy sampling

**This theoretical connection is documented in:** [Recovering Classical RL from GRL](../recovering_classical_rl.md)

---

#### Potential GRL Advantages (Theoretical)

- **Off-policy learning**: Particle memory could enable experience replay
- **Smooth generalization**: Nearby prompts might share value via kernel
- **Uncertainty**: Sparse particles could indicate high uncertainty
- **Interpretability**: Energy landscape over prompt space

**However**: These advantages are speculative without empirical validation.

---

#### Implementation Reality

**Challenges**:
1. **Infrastructure**: Requires reward model training, human feedback data, preference datasets
2. **Computational cost**: LLM fine-tuning is expensive (even GPT-2)
3. **Comparison difficulty**: Matching PPO requires careful hyperparameter tuning
4. **Integration**: Modern RLHF uses TRL, transformers, accelerate — non-trivial to integrate
5. **Validation**: Showing clear advantages requires extensive controlled experiments

**Estimated Effort**: 6-12 months of focused work with GPU resources

**When to Pursue**:
- ✅ After GRL validated on simpler environments (2D Nav, classical RL)
- ✅ If collaborators or funding available
- ✅ If clear path to demonstrating advantages
- ✅ If access to human feedback datasets

**Realistic First Step** (if pursued):
- Toy RLHF-like problem (small vocabulary, simple preference task)
- Not real LLM, but demonstrates GRL can handle discrete sequential choices
- Fast iteration, low compute cost

---

### Environment Simulation Package Structure

Given the scope of applications, we'll need a well-organized environment package:

```
src/grl/envs/
├── __init__.py
├── base_env.py                 # GRL environment interface
│
├── validation/                 # Tier 1: Classical RL baselines
│   ├── nav2d.py               # 2D navigation (Priority 7)
│   ├── cartpole.py            # Discrete control
│   ├── pendulum.py            # Continuous control
│   └── mujoco_envs.py         # Robotics (Ant, Humanoid)
│
├── strategic/                  # Tier 2: Modern RL applications
│   ├── llm_finetuning.py      # 🔥 RLHF for LLMs (High Priority)
│   ├── prompt_optimization.py  # Prompt tuning
│   ├── molecule_design.py      # Drug discovery
│   └── nas.py                  # Neural Architecture Search
│
├── novel/                      # Tier 3: GRL-native applications
│   ├── physics_sim.py          # Force field control
│   ├── fluid_control.py        # PDE-governed systems
│   ├── image_editing.py        # Parametric image transforms
│   └── multi_robot.py          # Multi-agent coordination
│
├── wrappers/                   # Adapters for existing environments
│   ├── gym_wrapper.py          # OpenAI Gym → GRL
│   ├── gymnasium_wrapper.py    # Gymnasium → GRL
│   ├── dm_control_wrapper.py   # DeepMind Control → GRL
│   └── rlhf_wrapper.py         # TRL/transformers → GRL
│
└── scenarios/                  # Predefined configurations
    ├── paper_scenarios.py      # Scenarios from original paper
    ├── benchmark_suite.py      # Standard benchmarks
    └── tutorials.py            # Teaching examples
```

**Key Design Principle**: 
- **Wrappers** allow GRL to be applied to **any existing RL environment**
- **Native environments** showcase GRL's unique capabilities
- **Scenarios** provide reproducible experiments

---

### Strategic Roadmap Update

**Phase 1 (Q1 2026)**: Foundation ⭐⭐⭐
- Complete Part I tutorial
- Implement core GRL components
- ✅ 2D Navigation validated

**Phase 2 (Q2 2026)**: Classical RL Recovery ⭐⭐
- Implement wrappers (Gym, Gymnasium)
- Reproduce DQN on CartPole
- Reproduce SAC on Pendulum
- **Document**: "Recovering Classical RL from GRL" ✅ Complete
- **Paper A** submission

**Phase 3 (Q3-Q4 2026)**: Novel Contributions ⭐
- Amplitude-based RL (if promising)
- MDL consolidation
- Concept-based mixture of experts
- **Papers B & C** submissions

**Future Directions** (No timeline):
- **Theoretical articles**: Justify how RLHF, prompt optimization, molecule design are special cases
- **Implementation**: If resources/collaborators available, pick 1-2 strategic applications
- **Novel applications**: Physics simulation, multi-robot coordination (GRL-native capabilities)

---

### Success Metrics

**Technical** (Achievable):
- [ ] 2D Navigation demo complete with professional web interface
- [ ] GRL recovers DQN/SAC results on classical benchmarks (±5% performance)
- [ ] Classical RL wrappers work with existing environments
- [ ] Documentation complete and accessible

**Research** (Achievable):
- [ ] Part I tutorial complete (Chapters 0-10)
- [ ] Part II foundation (concept subspaces formalized)
- [ ] "Recovering Classical RL" document demonstrates generality
- [ ] Paper A submitted (operator formalism)
- [ ] 1-2 papers on novel contributions (amplitude-based RL or MDL consolidation)

**Adoption** (Aspirational):
- [ ] GitHub stars: 100+ (realistic), 1000+ (stretch)
- [ ] External users beyond our lab
- [ ] Cited in other papers
- [ ] Conference workshop or tutorial (if invited)

**Strategic Applications** (Aspirational, No Timeline):
- [ ] Theoretical articles justify RLHF/prompt-opt as special cases
- [ ] If resources available: implement 1-2 strategic applications
- [ ] Industry partnerships (if opportunities arise)

---

## Code Structure

```
src/grl/
├── __init__.py
├── core/
│   ├── particle_memory.py          # Priority 1: Particle state
│   ├── kernels.py                  # Priority 2: RKHS geometry
│   └── soft_transitions.py         # Priority 6: Emergent uncertainty
├── algorithms/
│   ├── memory_update.py            # Priority 3: Belief transition
│   ├── rf_sarsa.py                 # Priority 4: TD learning
│   └── policy_inference.py         # Priority 5: Action selection
├── concepts/                        # Part II: Emergent Structure
│   ├── spectral_clustering.py      # Priority 8: Functional clustering
│   ├── concept_discovery.py        # Priority 9: Automated structure
│   ├── concept_hierarchy.py        # Priority 10: Multi-level abstraction
│   └── concept_policies.py         # Priority 11: Hierarchical control
├── envs/
│   ├── nav2d.py                    # Priority 7: 2D Navigation Domain
│   ├── scenarios.py                # Predefined scenarios (Figure 4)
│   └── base_env.py                 # Environment interface
├── agents/
│   ├── grl_agent.py                # Complete GRL agent
│   └── evaluation.py               # Agent evaluation tools
├── utils/
│   ├── config.py                   # Configuration management
│   ├── reproducibility.py          # Random seeds, determinism
│   └── metrics.py                  # Performance metrics
├── visualization/
│   ├── energy_landscape.py         # Energy field heatmaps
│   ├── particle_viz.py             # Particle memory plots
│   ├── trajectory_viz.py           # Agent trajectories
│   └── concept_viz.py              # Concept subspace plots (Part II)
└── web/                            # Web deployment (Priority 7)
    ├── api.py                      # FastAPI backend
    ├── static/                     # Frontend assets
    └── templates/                  # HTML templates
```

---

## Dependencies

### Core Dependencies

```
torch >= 2.0              # Neural operators, gradient computation
numpy >= 1.24             # Numerical operations
scipy >= 1.10             # Scientific computing, optimization
gpytorch >= 1.10          # Gaussian processes (optional)
scikit-learn >= 1.3       # Spectral clustering (Part II)
```

### Visualization

```
matplotlib >= 3.7         # Static plots
seaborn >= 0.12          # Statistical visualization
plotly >= 5.14           # Interactive plots
```

### Web Deployment (Priority 7)

```
fastapi >= 0.104         # Backend API
uvicorn >= 0.24          # ASGI server
pydantic >= 2.4          # Data validation
jinja2 >= 3.1            # Templating
```

### Development

```
pytest >= 7.4            # Testing
black >= 23.9            # Code formatting
mypy >= 1.6              # Type checking
sphinx >= 7.2            # Documentation
```

---

## Quality Standards

### Code Quality

- [ ] All public functions have docstrings (NumPy style)
- [ ] Type hints throughout (Python 3.10+)
- [ ] Unit test coverage > 80%
- [ ] No linting errors (black, mypy, flake8)
- [ ] Examples run without modification
- [ ] Math notation matches paper

### Part I Validation

- [ ] Reproduce original paper results (Figure 4)
- [ ] MemoryUpdate converges
- [ ] RF-SARSA learns effectively
- [ ] Energy landscapes are smooth
- [ ] Particle memory grows/prunes correctly

### Part II Validation (After Part I)

- [ ] Spectral clustering identifies meaningful concepts
- [ ] Concept hierarchy is interpretable
- [ ] Concept-conditioned policies improve performance
- [ ] Transfer learning across concepts works

### 2D Navigation Demo

- [ ] Web interface is responsive and intuitive
- [ ] Visualizations render at 60 FPS
- [ ] All scenarios from paper work
- [ ] Export/sharing functionality works
- [ ] Tutorial notebooks are clear and complete

---

## Summary

**GRL-v0 Implementation** spans:
- **Part I** (Layers 1-3): Particle-based reinforcement fields
- **Part II** (Layer 4): Emergent structure and concept discovery
- **Demonstration**: Professional 2D navigation domain with web deployment

**Priority Order**:
1. Part I foundations (Priorities 1-6)
2. 2D Navigation validation (Priority 7) ⭐⭐ **Critical milestone**
3. Part II extensions (Priorities 8-11)
4. Additional environments and experiments

**Target**: Complete 2D navigation demo by **March 2026**

**See also**: [Research Roadmap](../../ROADMAP.md) for broader research plan

---

**Last Updated**: January 14, 2026
