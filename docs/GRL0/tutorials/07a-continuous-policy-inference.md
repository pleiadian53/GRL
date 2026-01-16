# Chapter 07a: Beyond Discrete Actions — Continuous Policy Inference

**Purpose**: Address the discrete action bottleneck and explore fully continuous GRL formulations  
**Prerequisites**: Chapter 07 (RF-SARSA)  
**Key Concepts**: Continuous action spaces, Langevin sampling, actor-critic in RKHS, action discovery, learned embeddings

---

## Introduction

Chapter 7 presented RF-SARSA as the core learning algorithm for GRL. However, the algorithm as specified has a **critical limitation** that constrains its applicability:

**The Discrete Action Assumption**: RF-SARSA requires a finite set of primitive actions $\mathcal{A} = \{a^{(1)}, \ldots, a^{(n)}\}$ to query the field $Q^+(s, a^{(i)})$ during policy inference.

This creates two problems:

1. **Manual Action Design**: Requires hand-crafted mapping $f_{A^+}: a^{(i)} \mapsto x_a^{(i)}$ from primitive actions to parametric representation
2. **Scalability**: For high-dimensional action parameters $\theta \in \mathbb{R}^{d_a}$, enumerating all actions is intractable

**Example**: In the 2D navigation domain (original GRL paper):

- Primitive actions: move in 12 directions (like a clock: 0°, 30°, 60°, ..., 330°)
- Manual mapping: each direction → angle $\theta \in [0, 2\pi)$
- **Limitation**: What if optimal action is at angle $\pi/7 \approx 25.7°$ (not in the discrete set)?

**This chapter explores solutions** that eliminate discrete actions entirely, enabling **fully continuous** policy inference in parametric action spaces.

---

## 1. The Problem: Discrete Actions as a Bottleneck

### 1.1 Where Discrete Actions Enter RF-SARSA

In Algorithm 2 (Chapter 7), discrete actions appear in two places:

**Step 6: Field-Based Action Evaluation**
```
For each a^(i) ∈ A:
    Form z^(i) = (s, x_a^(i))
    Query Q+(z^(i)) via GPR
Select a via Boltzmann policy
```

**Step 10: Primitive SARSA Update**
```
δ = r + γ Q(s', a') - Q(s, a)
Q(s, a) ← Q(s, a) + α δ
```

Both steps assume $a \in \mathcal{A}$ is **discrete**.

---

### 1.2 Why This Is Limiting

**Problem 1: Manual Feature Engineering**

The mapping $f_{A^+}: a \mapsto x_a$ must be designed by hand:

- 2D navigation: direction → angle
- Robotic reaching: discrete waypoints → joint angles
- Continuous control: requires discretizing continuous action space

**This defeats the purpose** of parametric actions—you still need domain expertise!

**Problem 2: Curse of Dimensionality**

For high-dimensional actions $\theta \in \mathbb{R}^{d_a}$:

- Discretizing each dimension with $k$ values → $k^{d_a}$ primitive actions
- Example: $d_a = 10$, $k = 10$ → $10^{10}$ actions (intractable!)

**Problem 3: Suboptimal Actions**

Optimal action might lie **between** discrete choices:

- Discrete set: $\{30°, 45°, 60°\}$
- Optimal: $42°$ (not representable!)

---

### 1.3 The SARSA Constraint

Why does RF-SARSA use discrete actions? **Because SARSA does**.

**Original SARSA** (Rummery & Niranjan, 1994):
$$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma Q(s', a') - Q(s, a)]$$

This assumes $a, a'$ are **indexable** (discrete) so you can store $Q(s, a)$ in a table or lookup structure.

**For continuous actions** $\theta \in \mathbb{R}^{d_a}$, you can't index $Q(s, \theta)$ this way!

---

## 2. Solution 1: Continuous SARSA via Langevin Sampling

**Key insight**: We don't need primitive actions—we can sample directly from the continuous field $Q^+(s, \theta)$ using **gradient-based sampling**.

### 2.1 Langevin Dynamics Refresher (from Chapter 03a)

Recall from the least action principle that optimal actions follow gradient flow:

$$\theta_{t+1} = \theta_t + \epsilon \nabla_\theta Q^+(s, \theta_t) + \sqrt{2\epsilon\lambda} \, \xi_t$$

where:

- $\nabla_\theta Q^+(s, \theta)$ = gradient of field w.r.t. action parameters (via Riesz representer, Chapter 04a)
- $\lambda$ = temperature (exploration)
- $\xi_t \sim \mathcal{N}(0, I)$ = Gaussian noise

**This is Langevin Monte Carlo** sampling from the Boltzmann distribution $\pi(\theta | s) \propto \exp(Q^+(s, \theta) / \lambda)$.

**Advantage**: No discrete action set needed! Sample $\theta$ directly from the field.

---

### 2.2 Continuous RF-SARSA: Modified Algorithm

Replace discrete action enumeration with continuous sampling:

**Original (discrete) RF-SARSA:**
```
For each a^(i) ∈ A:
    Query Q+(s, a^(i))
Select a ~ Boltzmann(Q+)
```

**Continuous RF-SARSA:**
```
Initialize θ_0 randomly (or from heuristic)
For k = 1 to K_sample:
    Compute ∇_θ Q+(s, θ_{k-1})  [via Riesz representer]
    θ_k ← θ_{k-1} + ε ∇_θ Q+(s, θ_{k-1}) + √(2ελ) ξ_k
Return θ_K
```

**Key change**: Action inference becomes **gradient-based optimization** on the field.

---

### 2.3 How to Update Q Without Discrete Actions?

**Problem**: SARSA update requires $Q(s, a)$ as a scalar, but now $\theta$ is continuous.

**Solution 1: Function Approximation** (Standard Approach)

Use a parametric function approximator (e.g., neural network):
$$Q_w(s, \theta)$$

Update via gradient descent:
$$w \leftarrow w - \alpha \nabla_w [Q_w(s, \theta) - (r + \gamma Q_w(s', \theta'))]^2$$

**But wait—this is just deep RL!** We've abandoned the particle-based field representation.

**Solution 2: Particle-Based Continuous SARSA** (GRL Way)

Keep the particle representation $Q^+(z) = \sum_i w_i k(z, z_i)$ but eliminate primitive $Q(s, a)$ table.

**Modified update**:

1. Sample action: $\theta \sim \pi(\cdot | s)$ via Langevin
2. Execute: observe $r$, $s'$
3. Sample next action: $\theta' \sim \pi(\cdot | s')$ via Langevin
4. **Compute TD target directly from field**:
   $$\delta = r + \gamma Q^+(s', \theta') - Q^+(s, \theta)$$
5. Form particle: $\omega = ((s, \theta), r + \gamma Q^+(s', \theta'))$ (bootstrap target, not table Q)
6. MemoryUpdate: $\Omega \leftarrow \text{MemoryUpdate}(\omega, \delta, k, \tau, \Omega)$

**Key difference**: No primitive $Q(s, a)$ table! TD target computed **directly** from field queries.

---

### 2.4 Algorithm: Continuous RF-SARSA

**Inputs:**

- Kernel $k(\cdot, \cdot; \theta)$
- Langevin step size $\epsilon$, temperature $\lambda$
- Number of Langevin steps $K_{\text{sample}}$
- TD learning rate $\alpha$ (for particle weight updates)
- Discount $\gamma$, association threshold $\tau$

**Initialization:**

- Particle memory $\Omega \leftarrow \emptyset$
- Kernel hyperparameters $\theta$ (via ARD or prior)

**For each episode:**

1. Observe initial state $s_0$

**For each step $t$:**

2. **Action sampling via Langevin**:

   - Initialize $\theta_0$ randomly or from heuristic
   - For $k = 1, \ldots, K_{\text{sample}}$:
     $$\nabla_\theta Q^+(s_t, \theta_{k-1}) = \sum_{i=1}^N w_i \nabla_\theta k((s_t, \theta_{k-1}), z_i)$$
     $$\theta_k \leftarrow \theta_{k-1} + \epsilon \nabla_\theta Q^+(s_t, \theta_{k-1}) + \sqrt{2\epsilon\lambda} \, \xi_k$$
   - Set $\theta_t \leftarrow \theta_{K_{\text{sample}}}$

3. **Execute action**:

   - Execute $\theta_t$ in environment
   - Observe $r_t$, $s_{t+1}$

4. **Next action sampling**:

   - Repeat step 2 for $s_{t+1}$ to get $\theta_{t+1}$

5. **Field-based TD update**:

   - Query field: $Q_t^+ \leftarrow Q^+(s_t, \theta_t)$, $Q_{t+1}^+ \leftarrow Q^+(s_{t+1}, \theta_{t+1})$
   - Compute TD error: $\delta_t \leftarrow r_t + \gamma Q_{t+1}^+ - Q_t^+$
   - Form TD target: $y_t \leftarrow r_t + \gamma Q_{t+1}^+$ (bootstrap from field)

6. **Particle reinforcement**:

   - Form particle: $\omega_t \leftarrow ((s_t, \theta_t), y_t)$
   - Update memory: $\Omega \leftarrow \text{MemoryUpdate}(\omega_t, \delta_t, k, \tau, \Omega)$

7. **Periodic ARD**:

   - Every $T$ steps, update kernel hyperparameters $\theta$ via ARD on $\Omega$

**Repeat until terminal.**

---

### 2.5 Advantages and Limitations

**✅ Advantages**:

1. **No manual action mapping**: $\theta$ is sampled directly from field
2. **Fully continuous**: No discretization of action space
3. **Principled**: Langevin sampling from Boltzmann distribution (Chapter 03a)
4. **Natural exploration**: Temperature $\lambda$ controls stochasticity

**⚠️ Limitations**:

1. **Gradient computation**: Requires $\nabla_\theta k(z, z')$ (analytic or autodiff)
2. **Langevin convergence**: Need $K_{\text{sample}}$ steps per action (slower)
3. **Local optima**: Gradient descent can get stuck (non-convex $Q^+$)
4. **No primitive Q-table**: Loses SARSA's tabular grounding

**When to use**: High-dimensional continuous actions where discrete enumeration is impossible.

---

## 3. Solution 2: Actor-Critic in RKHS

**Idea**: Decouple policy (actor) from value function (critic), as in standard actor-critic methods.

### 3.1 The Actor-Critic Framework

**Actor**: Parametric policy $\pi_\phi(\theta | s)$
- Could be Gaussian: $\pi_\phi(\theta | s) = \mathcal{N}(\mu_\phi(s), \sigma_\phi(s))$
- Trained via policy gradient

**Critic**: Value function $Q^+(s, \theta)$ in RKHS (as before)
- Trained via TD learning (using particles)

**Advantage**: Policy is flexible, efficient to sample; critic provides value estimates for learning.

---

### 3.2 GRL Actor-Critic Algorithm

**Modifications to RF-SARSA:**

**Action selection** (no Langevin needed):

- Sample from actor: $\theta_t \sim \pi_\phi(\cdot | s_t)$
- Fast sampling (single forward pass)

**Critic update** (unchanged):

- Particle-based TD: $\delta_t = r_t + \gamma Q^+(s_{t+1}, \theta_{t+1}) - Q^+(s_t, \theta_t)$
- MemoryUpdate as before

**Actor update** (policy gradient):

- Compute advantage: $A_t = Q^+(s_t, \theta_t) - V(s_t)$ where $V(s) = \mathbb{E}_{\theta \sim \pi_\phi}[Q^+(s, \theta)]$
- Update policy: $\phi \leftarrow \phi + \beta \nabla_\phi \log \pi_\phi(\theta_t | s_t) A_t$

---

### 3.3 Pseudocode

```python
class GRL_ActorCritic:
    def __init__(self, actor_net, kernel, alpha=0.1, beta=0.01, gamma=0.9, tau=0.1):
        self.actor = actor_net  # Neural network: s → μ(s), σ(s)
        self.kernel = kernel
        self.particles = []  # Critic: particle memory Ω
        self.alpha = alpha  # Critic learning rate
        self.beta = beta  # Actor learning rate
        self.gamma = gamma
        self.tau = tau
    
    def sample_action(self, s):
        """Sample action from actor policy."""
        mu, sigma = self.actor(s)
        theta = np.random.normal(mu, sigma)
        return theta, mu, sigma
    
    def critic_predict(self, s, theta):
        """Predict Q+(s, θ) via GPR on particles."""
        z = np.concatenate([s, theta])
        if len(self.particles) == 0:
            return 0.0
        return gpr_predict(z, self.particles, self.kernel)
    
    def update(self, s, theta, r, s_next, theta_next):
        """Actor-critic update."""
        # Critic TD update
        Q_current = self.critic_predict(s, theta)
        Q_next = self.critic_predict(s_next, theta_next)
        delta = r + self.gamma * Q_next - Q_current
        
        # Particle reinforcement (critic)
        z = np.concatenate([s, theta])
        y = r + self.gamma * Q_next  # TD target
        particle = (z, y)
        self.particles = memory_update(particle, delta, self.kernel, self.tau, self.particles)
        
        # Actor policy gradient
        # Advantage: A = Q(s,θ) - V(s)
        # For simplicity, use TD error as advantage (A ≈ δ)
        log_prob = self.actor.log_prob(theta, s)  # log π_φ(θ|s)
        actor_loss = -log_prob * delta  # Policy gradient
        
        self.actor.update(actor_loss, self.beta)
```

---

### 3.4 Advantages and Limitations

**✅ Advantages**:

1. **Efficient sampling**: No Langevin iterations, single forward pass
2. **Flexible policy**: Can model complex distributions (multimodal, correlations)
3. **Standard framework**: Leverages decades of actor-critic research
4. **Scalable**: Neural networks handle high-dimensional states/actions

**⚠️ Limitations**:

1. **Parametric policy**: Loses non-parametric flexibility of particle-based field
2. **Two learning systems**: Actor and critic can be unstable (common in AC methods)
3. **Hyperparameters**: Requires tuning learning rates, network architectures
4. **Divergence risk**: Policy and value can diverge without careful tuning

**When to use**: High-dimensional, complex policies where sampling efficiency matters.

---

## 4. Solution 3: Learned Action Embeddings

**Idea**: Instead of hand-designing $f_{A^+}: a \mapsto x_a$, **learn** the embedding jointly with the value function.

### 4.1 The Embedding Problem

**Original GRL**: Requires manual mapping from primitive actions to parametric representation.

**Example** (2D navigation):

- Primitive: "move North"
- Manual embedding: North → angle $\theta = \pi/2$

**Question**: Can we learn this mapping automatically?

**Answer**: Yes! Use contrastive learning or auto-encoder.

---

### 4.2 Contrastive Action Embedding

**Objective**: Embed actions such that:

- Similar actions (similar outcomes) → close in embedding space
- Dissimilar actions → far apart

**Training**:

1. Collect transitions: $(s, a, s')$
2. Define similarity: $\text{sim}(a, a') = \exp(-\|s' - s''\|^2)$ where $s', s''$ are outcomes
3. Learn embedding: $f_\psi: a \mapsto x_a$ such that $\|x_a - x_{a'}\|^2 \propto -\log \text{sim}(a, a')$

**Loss** (InfoNCE-style):
$$\mathcal{L}_{\text{embed}} = -\log \frac{\exp(\langle f_\psi(a), f_\psi(a^+) \rangle / \tau)}{\sum_{a^-} \exp(\langle f_\psi(a), f_\psi(a^-) \rangle / \tau)}$$

where $a^+$ is a positive pair (similar action), $a^-$ are negatives.

---

### 4.3 Joint Learning: Embedding + Value Function

**Algorithm**:

1. **Initialization**: Random embedding $f_\psi$
2. **Collect experience**: $(s, a, r, s')$
3. **Embed actions**: $x_a \leftarrow f_\psi(a)$, $x_{a'} \leftarrow f_\psi(a')$
4. **TD update**: Standard RF-SARSA using embedded actions
5. **Embedding update**: Contrastive loss based on $(a, a')$ pairs with similar outcomes
6. **Repeat**

**Key insight**: Embedding evolves to make TD learning easier—actions with similar values cluster in embedding space!

---

### 4.4 Advantages and Limitations

**✅ Advantages**:

1. **No manual design**: Embedding learned from data
2. **Adaptive**: Embedding adapts to task (reward structure)
3. **Discovers structure**: May reveal latent action properties

**⚠️ Limitations**:

1. **Requires experience**: Need data to learn embedding
2. **Non-stationarity**: Embedding changes as policy improves (moving target)
3. **Computational cost**: Joint optimization is complex

**When to use**: When action space structure is unknown, or manual embedding is infeasible.

---

## 5. Solution 4: Hierarchical Action Discovery

**Idea**: Let the agent **discover** a discrete action set automatically by clustering in action parameter space.

### 5.1 The Clustering Approach

**Step 1**: Start with continuous action parameter space $\mathbb{R}^{d_a}$

**Step 2**: After initial exploration, cluster observed action parameters:

- Use k-means, GMM, or DBSCAN on $\{\theta_i\}$ from particle memory
- Cluster centers become "discovered actions"

**Step 3**: Use discovered actions as discrete set for RF-SARSA

**Step 4**: Periodically re-cluster as policy evolves

---

### 5.2 Algorithm Sketch

```python
def discover_actions(particles, n_clusters=10):
    """Discover discrete actions via clustering."""
    # Extract action parameters from particle memory
    theta_set = [z[len(state):] for (z, q) in particles]  # z = (s, θ)
    
    # Cluster in action space
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(theta_set)
    
    # Cluster centers = discovered actions
    discovered_actions = kmeans.cluster_centers_
    return discovered_actions

# In RF-SARSA loop:
if episode % T_discovery == 0:
    A_discovered = discover_actions(self.particles, n_clusters=10)
    # Use A_discovered as discrete action set for next T_discovery episodes
```

---

### 5.3 Options Framework Connection

This is closely related to the **options framework** (Sutton et al., 1999):

- **Options**: Temporally extended actions (subpolicies)
- **GRL version**: Discovered action clusters become "options"
- Can learn hierarchical policies: high-level chooses option, low-level executes

---

### 5.4 Advantages and Limitations

**✅ Advantages**:

1. **Automatic**: No manual action design
2. **Data-driven**: Discovers actions that matter for the task
3. **Hierarchical**: Enables multi-level policies

**⚠️ Limitations**:

1. **Requires exploration**: Need diverse data to cluster well
2. **K selection**: How many clusters? (hyperparameter)
3. **Non-stationary**: Discovered actions change over time

**When to use**: Complex action spaces where structure is unknown, or hierarchical RL is beneficial.

---

## 6. Solution 5: Direct Optimization on the Field

**Most radical solution**: Eliminate TD learning entirely! Directly optimize the field $Q^+$ via gradient ascent on expected return.

### 6.1 Field-Based Policy Gradient

**Objective**: Maximize expected return
$$J(\Omega) = \mathbb{E}_{\tau \sim \pi_\Omega}[\sum_{t=0}^T r_t]$$

where $\pi_\Omega$ is the policy induced by field $Q^+(\cdot; \Omega)$ (e.g., via Langevin sampling).

**Gradient**:
$$\nabla_\Omega J = \mathbb{E}_{\tau}[\sum_{t=0}^T \nabla_\Omega \log \pi_\Omega(\theta_t | s_t) R_t]$$

where $R_t = \sum_{t'=t}^T r_{t'}$ is the return from time $t$.

**Update**: Gradient ascent on particle memory:

- Add particles where policy needs reinforcement
- Remove particles where policy is suboptimal
- Adjust weights to increase expected return

---

### 6.2 Challenges

**Problem 1**: Computing $\nabla_\Omega \log \pi_\Omega(\theta | s)$ is non-trivial—policy is implicit (via GPR + Langevin).

**Problem 2**: High variance (standard policy gradient issue).

**Solution**: Use **score matching** or **diffusion models** to learn $\nabla_\theta \log \pi(\theta | s)$ directly.

**This is an active research direction!** (Connect to modern diffusion-based RL: Diffusion-QL, Diffuser, Decision Diffusion, etc.)

---

## 7. Alternative Learning Mechanisms Beyond SARSA

### 7.1 Q-Learning in RKHS

**Replace** SARSA's on-policy update with Q-learning's off-policy update:

**Original SARSA**:
$$\delta = r + \gamma Q(s', a') - Q(s, a) \quad \text{(uses actual next action } a'\text{)}$$

**Q-Learning**:
$$\delta = r + \gamma \max_{a'} Q(s', a') - Q(s, a) \quad \text{(uses max over actions)}$$

**For continuous actions**: Replace $\max_{a'}$ with optimization:
$$\theta^* = \arg\max_\theta Q^+(s', \theta)$$

Use gradient ascent or Langevin sampling to find $\theta^*$.

**Advantage**: Off-policy (can reuse experience, more sample-efficient).

**Limitation**: Still requires optimization at each step (costly).

---

### 7.2 Model-Based: Dyna-Style Planning

**Idea**: Use particle memory as a forward model, perform planning.

**Dyna-Q analog for GRL**:

1. **Direct learning**: Update $\Omega$ from real experience (as in RF-SARSA)
2. **Model learning**: Particles encode transitions $(s, \theta) \to (r, s')$
3. **Planning**: Simulate trajectories using particle memory
   - Query $Q^+(s, \theta)$ to predict $r$
   - Use GP to predict $s'$ (if we model transition dynamics)
   - Perform TD updates on simulated experience

**Advantage**: Sample efficiency (real experience + simulated).

**Limitation**: Requires modeling $p(s' | s, \theta)$, not just $Q^+$.

---

### 7.3 Successor Representations

**Idea**: Decouple environment dynamics from reward.

**Successor representation**:
$$\psi(s, \theta) = \mathbb{E}[\sum_{t=0}^\infty \gamma^t \phi(s_t) \mid s_0 = s, \theta_0 = \theta]$$

where $\phi(s)$ are state features.

**Value function**:
$$Q^+(s, \theta) = \psi(s, \theta)^\top w$$

where $w$ are reward weights.

**Advantage**: Transfer learning (reuse $\psi$ for new reward functions $w$).

**GRL connection**: Can represent $\psi$ via particles in RKHS!

---

## 8. Practical Recommendations

### 8.1 Decision Tree: Which Approach to Use?

```
Is action space discrete with < 100 actions?
├─ YES → Use original RF-SARSA (Chapter 7)
└─ NO → Is action dimensionality high (d_a > 5)?
    ├─ YES → Use Actor-Critic in RKHS (Solution 2)
    │        Efficient sampling, scalable
    └─ NO → Is gradient computation feasible?
        ├─ YES → Use Continuous RF-SARSA with Langevin (Solution 1)
        │        Principled, no extra networks
        └─ NO → Use Learned Embeddings (Solution 3) or
                 Hierarchical Discovery (Solution 4)
```

---

### 8.2 Hybrid Approach (Recommended)

**Best of both worlds**: Combine solutions!

**Stage 1: Exploration** (Episodes 1-100)
- Use Langevin sampling (Solution 1) for pure exploration
- Build diverse particle memory

**Stage 2: Discovery** (Episodes 100-200)
- Cluster actions (Solution 4) to find structure
- Use discovered discrete actions for efficiency

**Stage 3: Exploitation** (Episodes 200+)
- Use Actor-Critic (Solution 2) with structured embedding
- Fine-tune via continuous Langevin when needed

---

## 9. Implementation: Continuous RF-SARSA

Here's a complete implementation of Solution 1:

```python
import numpy as np

class ContinuousRFSARSA:
    """RF-SARSA without discrete actions: pure Langevin sampling."""
    
    def __init__(self, kernel, epsilon=0.01, lambda_temp=1.0, K_sample=10, 
                 gamma=0.9, tau=0.1, T_ard=100):
        self.kernel = kernel
        self.epsilon = epsilon  # Langevin step size
        self.lambda_temp = lambda_temp  # Temperature
        self.K_sample = K_sample  # Langevin iterations
        self.gamma = gamma
        self.tau = tau
        self.T_ard = T_ard
        
        self.particles = []  # (z, y) pairs
        self.t_ard = 0
    
    def q_plus(self, s, theta):
        """Query Q+(s, θ) via GPR."""
        if len(self.particles) == 0:
            return 0.0
        
        z = np.concatenate([s, theta])
        
        # GP prediction (assuming precomputed alpha coefficients)
        k_vec = np.array([self.kernel(z, p[0]) for p in self.particles])
        return k_vec @ self.alpha_gpr
    
    def grad_q_plus(self, s, theta):
        """Compute ∇_θ Q+(s, θ) via Riesz representer."""
        if len(self.particles) == 0:
            return np.zeros_like(theta)
        
        z = np.concatenate([s, theta])
        d_s = len(s)
        
        # Gradient via kernel derivatives
        grad = np.zeros_like(theta)
        for (z_i, q_i), alpha_i in zip(self.particles, self.alpha_gpr):
            # ∂k(z, z_i)/∂θ (assuming RBF kernel)
            diff = z - z_i
            k_val = self.kernel(z, z_i)
            grad_k = -diff[d_s:] / (self.kernel.lengthscale**2) * k_val
            grad += alpha_i * grad_k
        
        return grad
    
    def sample_action_langevin(self, s):
        """Sample action via Langevin dynamics on Q+ field."""
        # Initialize randomly
        theta = np.random.randn(self.action_dim)
        
        # Langevin iterations
        for k in range(self.K_sample):
            grad = self.grad_q_plus(s, theta)
            theta = theta + self.epsilon * grad + \
                    np.sqrt(2 * self.epsilon * self.lambda_temp) * np.random.randn(self.action_dim)
        
        return theta
    
    def update(self, s, theta, r, s_next, theta_next):
        """Continuous RF-SARSA update."""
        # Query field for TD
        Q_current = self.q_plus(s, theta)
        Q_next = self.q_plus(s_next, theta_next)
        
        # TD error
        delta = r + self.gamma * Q_next - Q_current
        
        # TD target (bootstrap from field, not from Q-table!)
        y = r + self.gamma * Q_next
        
        # Form particle
        z = np.concatenate([s, theta])
        particle = (z, y)
        
        # MemoryUpdate
        self.particles = memory_update(particle, delta, self.kernel, self.tau, self.particles)
        
        # Periodic ARD
        self.t_ard += 1
        if self.t_ard % self.T_ard == 0:
            self.update_kernel_hyperparameters()
    
    def update_kernel_hyperparameters(self):
        """Run ARD to update kernel lengthscales."""
        if len(self.particles) < 10:
            return  # Need sufficient data
        
        Z = np.array([p[0] for p in self.particles])
        q = np.array([p[1] for p in self.particles])
        
        # Fit GP with ARD
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF
        
        kernel = RBF(length_scale=np.ones(Z.shape[1]), length_scale_bounds=(1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)
        gp.fit(Z, q)
        
        # Update kernel
        self.kernel.set_lengthscales(gp.kernel_.length_scale)
        
        # Recompute GPR coefficients
        K = self.kernel.matrix(Z, Z)
        self.alpha_gpr = np.linalg.solve(K + 1e-6 * np.eye(len(Z)), q)
    
    def train(self, env, n_episodes=100):
        """Training loop."""
        for episode in range(n_episodes):
            s = env.reset()
            theta = self.sample_action_langevin(s)
            
            done = False
            while not done:
                # Execute
                s_next, r, done = env.step(theta)
                
                # Next action
                theta_next = self.sample_action_langevin(s_next)
                
                # Update
                self.update(s, theta, r, s_next, theta_next)
                
                # Advance
                s, theta = s_next, theta_next
            
            print(f"Episode {episode}: Total reward = {env.episode_return}")
```

---

## 10. Summary

### 10.1 The Discrete Action Bottleneck

RF-SARSA (Chapter 7) requires discrete primitive actions, creating limitations:

- Manual action mapping needed
- Curse of dimensionality for high-d actions
- Suboptimal actions (discrete approximation of continuous space)

### 10.2 Five Solutions

| Solution | Key Idea | Advantages | Limitations |
|----------|----------|------------|-------------|
| **1. Continuous SARSA** | Langevin sampling on field | Principled, no manual design | Gradient computation, convergence |
| **2. Actor-Critic in RKHS** | Parametric policy + particle critic | Efficient sampling, scalable | Parametric policy, instability risk |
| **3. Learned Embeddings** | Learn action representation jointly | Adaptive, discovers structure | Non-stationary, requires experience |
| **4. Hierarchical Discovery** | Cluster actions automatically | Data-driven, enables hierarchy | K selection, non-stationary |
| **5. Direct Optimization** | Policy gradient on field | No TD needed | High variance, complex |

### 10.3 Practical Recommendations

**For most problems**: Start with **Continuous SARSA** (Solution 1) or **Actor-Critic** (Solution 2).

**Hybrid approach**: Combine exploration (Langevin) → discovery (clustering) → exploitation (actor-critic).

### 10.4 Beyond SARSA

Alternative learning mechanisms:

- **Q-Learning in RKHS**: Off-policy, sample-efficient
- **Dyna-style planning**: Model-based, simulate with particles
- **Successor representations**: Transfer learning across reward functions

---

## 11. Key Takeaways

1. **Original RF-SARSA has a discrete action bottleneck** requiring manual mapping $f_{A^+}$

2. **Continuous SARSA via Langevin** eliminates discrete actions entirely—sample directly from field

3. **Actor-Critic in RKHS** combines efficiency (neural network policy) with non-parametric critic (particles)

4. **Learned embeddings** remove manual design—discover action structure from data

5. **Hierarchical discovery** via clustering enables data-driven action spaces

6. **SARSA is not the only way**—Q-learning, model-based, policy gradients all viable

7. **The least action principle** (Chapter 03a) provides theoretical foundation for Langevin-based continuous RL

---

## 12. Open Research Questions

1. **Convergence theory for continuous RF-SARSA**: Does Langevin + TD converge? Under what conditions?

2. **Optimal exploration in continuous actions**: How to balance Langevin temperature vs. ARD adaptation?

3. **Sparse particle representations**: Can we maintain performance with fewer particles in continuous action spaces?

4. **Hierarchical GRL**: How to discover and learn action hierarchies automatically?

5. **Transfer learning**: Can learned action embeddings transfer across tasks?

---

## Further Reading

**Continuous Action RL**:

- Lillicrap, T. P., et al. (2016). "Continuous control with deep reinforcement learning" (DDPG). *ICLR*.
- Haarnoja, T., et al. (2018). "Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor" (SAC). *ICML*.

**Langevin Dynamics for RL**:

- Levine, S. (2018). "Reinforcement learning and control as probabilistic inference: Tutorial and review." *arXiv:1805.00909*.
- Ajay, A., et al. (2023). "Is conditional generative modeling all you need for decision making?" *ICLR* (Diffusion-QL).

**Action Embeddings**:

- van den Oord, A., Li, Y., & Vinyals, O. (2018). "Representation learning with contrastive predictive coding." *arXiv:1807.03748*.
- Kipf, T., et al. (2020). "Contrastive learning of structured world models." *ICLR*.

**Options Framework**:

- Sutton, R. S., Precup, D., & Singh, S. (1999). "Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning." *Artificial Intelligence*.

---

**[← Back to Chapter 07: RF-SARSA](07-rf-sarsa.md)** | **[Next: Chapter 08 →]()**

**[Related: Chapter 03a - Least Action Principle](03a-least-action-principle.md)** | **[Related: Chapter 04a - Riesz Representer](04a-riesz-representer.md)**

---

**Last Updated**: January 14, 2026
