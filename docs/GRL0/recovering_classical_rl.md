# Recovering Classical RL from GRL

**Purpose**: Demonstrate that traditional RL algorithms are special cases of GRL  
**Audience**: Classical RL researchers, practitioners, skeptics  
**Goal**: Bridge the gap between familiar methods and the GRL framework

---

## Executive Summary

**Key Claim**: Generalized Reinforcement Learning (GRL) is not a replacement for classical RL—it's a **unifying framework** that recovers existing methods as special cases while enabling new capabilities.

**Why This Matters**:

- **Adoption**: Researchers trust frameworks that subsume what they already know
- **Validation**: If GRL recovers DQN/PPO/SAC, it must be correct
- **Innovation**: Once the connection is clear, extensions become natural

**What You'll Learn**:

1. How **Q-learning** emerges from GRL with discrete actions
2. How **DQN** is GRL with neural network approximation
3. How **Policy Gradients** (REINFORCE) follow from the energy landscape
4. How **Actor-Critic** (PPO, SAC) naturally arise in GRL
5. How **RLHF for LLMs** is GRL applied to language modeling

---

## Table of Contents

1. [The GRL→Classical RL Dictionary](#dictionary)
2. [Recovery 1: Q-Learning](#q-learning)
3. [Recovery 2: DQN (Deep Q-Network)](#dqn)
4. [Recovery 3: REINFORCE (Policy Gradient)](#reinforce)
5. [Recovery 4: Actor-Critic (PPO, SAC)](#actor-critic)
6. [Recovery 5: RLHF for LLMs](#rlhf)
7. [What GRL Adds Beyond Classical RL](#beyond)
8. [Implementation: From GRL to Classical](#implementation)

---

<a name="dictionary"></a>
## 1. The GRL→Classical RL Dictionary

| Classical RL Concept | GRL Equivalent | Notes |
|---------------------|----------------|-------|
| **State** $s$ | State $s$ | Same |
| **Discrete Action** $a \in \mathcal{A}$ | Fixed parametric mapping $\theta_a$ | One $\theta$ per discrete action |
| **Continuous Action** $a \in \mathbb{R}^d$ | Action parameters $\theta \in \mathbb{R}^d$ | Direct correspondence |
| **Q-function** $Q(s, a)$ | Reinforcement field $Q^+(s, \theta)$ | Evaluated at discrete $\theta_a$ |
| **Replay Buffer** $\mathcal{D}$ | Particle Memory $\Omega$ | Particles are weighted experiences |
| **Experience** $(s, a, r, s')$ | Particle $(z, w)$ where $z=(s,\theta)$, $w=r$ | Single transition |
| **TD Target** $y = r + \gamma \max_{a'} Q(s', a')$ | MemoryUpdate with TD target | Belief transition |
| **Policy** $\pi(a\|s)$ | Boltzmann over $Q^+(s, \cdot)$ | Temperature-controlled sampling |
| **Value Function** $V(s)$ | $\max_\theta Q^+(s, \theta)$ | Maximum over action parameters |
| **Exploration** $\epsilon$-greedy | Temperature $\beta$ in Boltzmann | Smooth instead of discrete |

**Key Insight**: Classical RL is GRL with:

- **Discrete or fixed action spaces**

- **Tabular or neural network approximation** of the field
- **Specific choices** of update rules

---

<a name="q-learning"></a>
## 2. Recovery 1: Q-Learning

### Classical Q-Learning

**Setup**:

- State space: $\mathcal{S}$
- Action space: $\mathcal{A} = \{a_1, \ldots, a_K\}$ (discrete, finite)
- Q-function: $Q(s, a)$ for each $(s, a)$ pair

**Update Rule**:
$$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$

**Policy**: $\epsilon$-greedy or softmax over $Q(s, \cdot)$

---

### GRL Version

**Setup**:

- State space: $\mathcal{S}$ (same)
- Action space: $\mathcal{A} = \{a_1, \ldots, a_K\}$ (discrete)
- **Map each discrete action to a parameter**: $\theta_1, \ldots, \theta_K$ (fixed)
- **Augmented space**: $\mathcal{Z} = \mathcal{S} \times \{\theta_1, \ldots, \theta_K\}$
- **Reinforcement field**: $Q^+(s, \theta_i)$ evaluated only at discrete points $\theta_i$

**Particle Memory**:

- Each experience $(s, a_i, r, s')$ creates particle $(z_i, w_i)$ where $z_i = (s, \theta_i)$, $w_i = r$

**MemoryUpdate**:

- Add particle $(z_t, w_t)$ where $z_t = (s_t, \theta_{a_t})$, $w_t = r_t$
- With **no kernel association** (set $k(z, z') = \delta(z, z')$), MemoryUpdate reduces to:
  $$Q^+(s, \theta_a) \leftarrow Q^+(s, \theta_a) + \alpha [y_t - Q^+(s, \theta_a)]$$
  where $y_t = r_t + \gamma \max_{a'} Q^+(s', \theta_{a'})$

**This is exactly Q-learning!**

---

### Key Takeaways

**Q-learning is GRL with**:

1. **Discrete action space** (finite $\{\theta_i\}$)
2. **Delta kernel** (no generalization between actions)
3. **Tabular representation** (store $Q$ for each state-action pair)

**What GRL adds**:

- Generalization via non-trivial kernels: $k((s, \theta_i), (s, \theta_j)) > 0$ for $i \neq j$
- Continuous interpolation: $Q^+(s, \theta)$ defined for all $\theta$, not just discrete actions
- Weighted particles: Experience importance via $w_i$

---

<a name="dqn"></a>
## 3. Recovery 2: DQN (Deep Q-Network)

### Classical DQN

**Setup**:

- Q-function approximated by neural network: $Q_\psi(s, a)$
- Experience replay buffer: $\mathcal{D} = \{(s_i, a_i, r_i, s_i')\}$
- Target network: $Q_{\psi^-}$ (delayed copy)

**Update Rule**:
$$\psi \leftarrow \psi - \eta \nabla_\psi \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} [(Q_\psi(s, a) - y)^2]$$
where $y = r + \gamma \max_{a'} Q_{\psi^-}(s', a')$

---

### GRL Version

**Setup**:

- **Field approximator**: Neural network $Q_\psi(s, \theta)$ approximates reinforcement field
- **Particle memory**: $\Omega = \{(z_i, w_i)\}$ where $z_i = (s_i, \theta_i)$
- **Kernel**: Implicit kernel induced by neural network architecture

**Update Rule**:
Sample particles from $\Omega$, compute TD targets:
$$\psi \leftarrow \psi - \eta \nabla_\psi \mathbb{E}_{(z,w) \sim \Omega} [(Q_\psi(z) - y)^2]$$
where $y = w + \gamma \max_{\theta'} Q_\psi(s', \theta')$

**Target network**: Optional, same as DQN

---

### Key Takeaways

**DQN is GRL with**:

1. **Neural network approximation** of the reinforcement field
2. **Experience replay** = particle memory sampling
3. **Discrete actions** (typically)

**What GRL adds**:

- **Explicit particle representation**: Particles are not just for replay, they define the field
- **Kernel interpretation**: Neural network as implicit kernel
- **Continuous action generalization**: $Q_\psi(s, \theta)$ for any $\theta$

---

<a name="reinforce"></a>
## 4. Recovery 3: REINFORCE (Policy Gradient)

### Classical REINFORCE

**Setup**:

- Policy: $\pi_\phi(a|s)$ (parameterized, e.g., neural network)
- Objective: $J(\phi) = \mathbb{E}_{\tau \sim \pi_\phi} [R(\tau)]$ (expected return)

**Update Rule** (score function gradient):
$$\nabla_\phi J(\phi) = \mathbb{E}_{\tau \sim \pi_\phi} [\sum_t \nabla_\phi \log \pi_\phi(a_t | s_t) \cdot G_t]$$

where $G_t = \sum_{t'=t}^T \gamma^{t'-t} r_{t'}$ (return from time $t$)

---

### GRL Version

**Setup**:

- **Reinforcement field**: $Q^+(s, \theta)$
- **Policy**: Boltzmann over field
  $$\pi(a | s) = \frac{\exp(\beta \, Q^+(s, \theta_a))}{\int \exp(\beta \, Q^+(s, \theta')) d\theta'}$$

**Gradient of Expected Return**:

The policy gradient in GRL is:
$$\nabla_\theta \mathbb{E}_{\pi} [R] = \mathbb{E}_{\pi} [\nabla_\theta Q^+(s, \theta) \cdot \text{advantage}]$$

If we parameterize $Q^+(s, \theta) = Q_\phi(s, \theta)$, then:
$$\nabla_\phi J(\phi) = \mathbb{E} [\nabla_\phi Q_\phi(s, \theta) \cdot G_t]$$

**Connection to REINFORCE**:

The score function gradient $\nabla_\phi \log \pi_\phi(a|s)$ in REINFORCE is equivalent to the field gradient $\nabla_\phi Q_\phi(s, \theta)$ when policy is Boltzmann.

**This recovers REINFORCE!**

---

### Key Takeaways

**REINFORCE is GRL with**:

1. **Boltzmann policy** derived from energy landscape
2. **Direct parameterization** of the field (or policy)
3. **Monte Carlo returns** ($G_t$) as targets

**What GRL adds**:

- **Energy interpretation**: $Q^+ = -E$ provides physics-inspired regularization
- **Particle-based updates**: No need for full gradient, use particle approximation
- **Smooth action selection**: Temperature $\beta$ controls exploration naturally

---

<a name="actor-critic"></a>
## 5. Recovery 4: Actor-Critic (PPO, SAC)

### Classical Actor-Critic

**Setup**:

- **Actor**: Policy $\pi_\phi(a|s)$
- **Critic**: Value function $V_\psi(s)$ or $Q_\psi(s, a)$

**Update**:

- **Critic**: TD learning
  $$\psi \leftarrow \psi - \eta \nabla_\psi [Q_\psi(s, a) - (r + \gamma V_\psi(s'))]^2$$
  
- **Actor**: Policy gradient with advantage
  $$\phi \leftarrow \phi + \eta \nabla_\phi \log \pi_\phi(a|s) \cdot A(s, a)$$
  where $A(s, a) = Q(s, a) - V(s)$ (advantage)

**Variants**:

- **PPO**: Clipped objective, KL penalty
- **SAC**: Entropy regularization, temperature tuning

---

### GRL Version

**Setup**:

- **Critic**: Reinforcement field $Q^+(s, \theta)$ (this is the "critic")
- **Actor**: Policy inferred from field via Boltzmann
  $$\pi(\theta | s) \propto \exp(\beta \, Q^+(s, \theta))$$

**Update**:

- **Field (Critic)**: RF-SARSA (two-layer TD system)
  - Primitive layer: TD learning for discrete transitions
  - GP layer: Smooth field over augmented space
  
- **Policy (Actor)**: Derived automatically from field
  - No separate actor parameters!
  - Policy gradient = field gradient

**Connection**:

- $Q^+(s, \theta)$ plays the role of both Q-function and value function
- Boltzmann policy automatically balances exploitation (high $Q^+$) and exploration (low $\beta$)
- Advantage: $A(s, \theta) = Q^+(s, \theta) - \max_{\theta'} Q^+(s, \theta')$

**This recovers Actor-Critic!**

---

### Key Takeaways

**Actor-Critic is GRL with**:

1. **Reinforcement field as critic**

2. **Boltzmann policy as actor** (no separate parameters)
3. **RF-SARSA as update rule**

**What GRL adds**:

- **Unified representation**: No need for separate actor and critic
- **Automatic exploration**: Temperature $\beta$ replaces entropy regularization
- **Particle-based**: Memory naturally handles off-policy data

**Special Cases**:

- **PPO**: GRL with clipped field updates, on-policy sampling
- **SAC**: GRL with entropy term in field (equivalent to temperature)

---

<a name="rlhf"></a>
## 6. Recovery 5: RLHF for LLMs

### Classical RLHF (Reinforcement Learning from Human Feedback)

**Setup** (e.g., for ChatGPT):

- **LLM**: $\pi_\phi(a_t | s_t)$ where $s_t$ = (prompt, response so far), $a_t$ = next token
- **Reward Model**: $r_\theta(s, a)$ learned from human preferences
- **Algorithm**: PPO or similar policy gradient method

**Update** (PPO):
$$\mathcal{L}(\phi) = \mathbb{E}_{(s,a) \sim \pi_\phi} [\min(r_\text{clip}, r_\text{KL})]$$

where:

- $r_\text{clip}$ = clipped advantage
- $r_\text{KL}$ = KL penalty from reference policy

---

### GRL Version

**Setup**:

- **State**: $s_t$ = (prompt, partial response)
- **Action**: $\theta_t$ = token ID or logit vector (discrete or continuous parameterization)
- **Augmented space**: $(s_t, \theta_t)$ in semantic embedding space
- **Reinforcement field**: $Q^+(s_t, \theta_t)$ = expected reward for generating token $\theta_t$ in context $s_t$

**Formulation**:

**Option 1: Discrete Tokens** (Classical RL recovery)
- $\theta_t \in \{1, \ldots, V\}$ (vocabulary size $V$)
- Field: $Q^+(s_t, \theta_t)$ for each token
- Policy: Softmax over field
  $$\pi(\theta_t | s_t) = \frac{\exp(\beta \, Q^+(s_t, \theta_t))}{\sum_{\theta'} \exp(\beta \, Q^+(s_t, \theta'))}$$

**Option 2: Continuous Parameterization** (GRL extension)
- $\theta_t \in \mathbb{R}^d$ = token embedding or logit vector
- Field: $Q^+(s_t, \theta_t)$ smooth over embedding space
- Policy: Sample from continuous distribution over $\theta$, map to nearest token

**Update**: RF-SARSA with human feedback as rewards
- Particle memory stores (prompt, response, reward) tuples
- Kernel generalizes across similar prompts/responses
- Field learns $Q^+(s, \theta)$ via TD learning

---

### Advantages of GRL for RLHF

**1. Off-Policy Learning**

- Particle memory enables replay
- Sample efficiency: learn from all past human feedback
- Classical RLHF (PPO) is on-policy only

**2. Smooth Generalization**

- Kernel similarity between prompts
- Transfer value across related contexts
- Fewer human labels needed

**3. Uncertainty Quantification**

- Sparse particles = high uncertainty
- Exploration naturally targets uncertain regions
- Safety: avoid high-stakes decisions with low confidence

**4. Interpretability**

- Energy landscape over prompt space
- Visualize which responses are preferred
- Particle inspection: "Why did you say that?"

---

### Implementation Path

**Phase 1: Discrete Tokens (Q1 2026)**

- Implement GRL on small model (GPT-2)
- Reproduce PPO results on standard RLHF benchmarks
- Show GRL recovers classical behavior

**Phase 2: Comparison (Q2 2026)**

- Compare sample efficiency: GRL vs. PPO
- Measure stability: fewer human labels needed?
- Quantify uncertainty: does GRL know what it doesn't know?

**Phase 3: Scale (Q3 2026)**

- Apply to larger model (LLaMA-7B or Mistral-7B)
- Test on real human feedback datasets
- Submit paper: "GRL for LLM Fine-tuning"

---

### Key Takeaways

**RLHF is GRL with**:

1. **Discrete action space** (tokens)
2. **On-policy updates** (PPO)
3. **Neural network approximation** of field

**What GRL adds for RLHF**:

- **Off-policy learning** (replay buffer of human feedback)
- **Kernel generalization** (transfer across prompts)
- **Uncertainty** (exploration where most uncertain)
- **Interpretability** (energy landscapes, particle inspection)

**Strategic Impact**: Demonstrating GRL on RLHF:

- Validates GRL on most commercially relevant RL problem
- Opens door to industry adoption (OpenAI, Anthropic, Meta)
- Natural bridge to scaling research

---

<a name="beyond"></a>
## 7. What GRL Adds Beyond Classical RL

While GRL recovers classical methods, it also enables capabilities that are difficult or impossible in standard RL:

### 1. Continuous Action Generalization

**Classical RL**: Discretize continuous actions or use neural networks
**GRL**: Smooth field $Q^+(s, \theta)$ over continuous $\theta$ via kernels

**Example**: Robot grasping
- Classical: Sample $N$ discrete grasp poses, learn Q-value for each
- GRL: Learn smooth field over continuous grasp space, interpolate between samples

---

### 2. Compositional Actions

**Classical RL**: Actions are atomic
**GRL**: Actions are operators that can be composed

**Example**: Multi-step manipulation
- Classical: Learn separate policies for "pick", "place", "push"
- GRL: Learn operators that compose: $\hat{O}_{\text{place}} \circ \hat{O}_{\text{pick}}$

---

### 3. Uncertainty Quantification

**Classical RL**: Uncertainty requires ensembles or Bayesian NNs
**GRL**: Particle sparsity directly indicates uncertainty

**Example**: Safe exploration
- Classical: Ensemble of Q-networks, high variance = uncertain
- GRL: Sparse particles = uncertain, avoid or explore based on risk

---

### 4. Energy-Based Regularization

**Classical RL**: Entropy regularization (SAC), KL penalties (PPO)
**GRL**: Energy function $E = -Q^+$ naturally regularizes via least-action principle

**Example**: Smooth, efficient policies
- Classical: Add entropy bonus to reward
- GRL: Energy naturally prefers smooth, low-energy paths (physics-inspired)

---

### 5. Particle-Based Interpretability

**Classical RL**: Black-box neural networks
**GRL**: Particles are interpretable experiences

**Example**: Debugging
- Classical: "Why did the policy fail?" → Inspect millions of weights
- GRL: "Why did the policy fail?" → Inspect nearby particles, visualize energy landscape

---

### 6. Hierarchical Abstraction (Part II)

**Classical RL**: Hierarchical RL requires careful design
**GRL**: Concepts emerge via spectral clustering

**Example**: Long-horizon tasks
- Classical: Manually define options/skills
- GRL: Spectral decomposition discovers concepts automatically

---

<a name="implementation"></a>
## 8. Implementation: From GRL to Classical

### Code Example: Q-Learning from GRL

```python
from grl.core import ParticleMemory, DeltaKernel
from grl.algorithms import MemoryUpdate

# Classical Q-learning setup
state_space = ["s1", "s2", "s3"]
action_space = ["a1", "a2"]

# GRL setup: Map discrete actions to fixed parameters
action_params = {
    "a1": torch.tensor([0.0]),  # θ_1
    "a2": torch.tensor([1.0]),  # θ_2
}

# Particle memory (GRL)
memory = ParticleMemory()

# Delta kernel (no generalization)
kernel = DeltaKernel()

# Experience: (s, a, r, s')
s, a, r, s_next = "s1", "a1", 1.0, "s2"

# Convert to GRL format
z = (s, action_params[a])
w = r

# MemoryUpdate (GRL) ≡ Q-learning update (Classical)
memory = memory_update(memory, z, w, kernel, alpha=0.1)

# Query field (GRL) ≡ Q(s, a) (Classical)
Q_sa = memory.query((s, action_params[a]))

# This is Q-learning!
```

---

### Code Example: DQN from GRL

```python
from grl.core import NeuralField
from grl.algorithms import FieldTDLearning

# Neural network approximates reinforcement field
field = NeuralField(state_dim=4, action_dim=2, hidden_dim=64)

# Experience replay (GRL particle memory)
memory = ParticleMemory()

# Training loop
for episode in range(num_episodes):
    for step in range(max_steps):
        # Sample from memory (experience replay)
        batch = memory.sample(batch_size=32)
        
        # Compute TD targets (same as DQN)
        td_targets = compute_td_targets(batch, field, gamma=0.99)
        
        # Update field (GRL) ≡ Update Q-network (DQN)
        loss = field.update(batch, td_targets)

# This is DQN!
```

---

## Conclusion

**GRL is a unifying framework** that:

1. ✅ **Recovers classical RL** (Q-learning, DQN, REINFORCE, PPO, SAC, RLHF)
2. ✅ **Extends to continuous actions** (smooth generalization via kernels)
3. ✅ **Enables composition** (operator algebra)
4. ✅ **Provides uncertainty** (particle sparsity)
5. ✅ **Interprets naturally** (energy landscapes, particles)
6. ✅ **Discovers structure** (spectral concepts in Part II)

**For practitioners**: GRL gives you what you already know (classical RL), plus new tools for continuous control, uncertainty, and interpretability.

**For researchers**: GRL provides a principled foundation for understanding why existing methods work and how to extend them.

**For industry**: GRL applies to modern problems (RLHF for LLMs) while offering advantages (off-policy, uncertainty, sample efficiency).

---

## Next Steps

**Reproduce Classical Results**:

- [ ] Implement Q-learning recovery on GridWorld
- [ ] Implement DQN recovery on CartPole
- [ ] Implement PPO recovery on continuous control (Pendulum)
- [ ] Implement RLHF recovery on small LLM (GPT-2)

**Document Connections**:

- [ ] Add "Classical RL Recovery" section to each tutorial chapter
- [ ] Create comparison tables (classical vs. GRL)
- [ ] Write blog post: "GRL: A Unifying Framework for RL"

**Validate**:

- [ ] Benchmark: GRL vs. DQN on Atari
- [ ] Benchmark: GRL vs. SAC on MuJoCo
- [ ] Benchmark: GRL vs. PPO on RLHF tasks

**Scale**:

- [ ] Apply GRL to LLaMA-7B fine-tuning
- [ ] Demonstrate advantages (sample efficiency, uncertainty)
- [ ] Submit paper: "GRL for LLM Fine-tuning"

---

**Last Updated**: January 14, 2026  
**See also**: [Implementation Roadmap](implementation/README.md) | [Research Roadmap](../ROADMAP.md)
