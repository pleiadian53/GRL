# Part 4: The Minimum Viable Experiment — Action Operators on Cell Perturbation

**Series:** Action Operators in Generalized Reinforcement Learning
**Part:** 4 of 4
**Audience:** Self-reference and interview prep
**Prerequisites:** Parts 1-3. Part 2 §4.4 (Lie group structure) and Part 3 §10 (the full arc) are the direct antecedents.

---

## 1. Where Part 3 Left Off

Part 3 closed by naming the frontier and stopping there:

> The full hybrid — operator-based actions with learned-kernel particle memory and equivariant feature maps — is the research frontier.

What it did not have was an experiment. Parts 1 through 3 are formal from end to end: they argue that actions should be operators, show how to construct one from parameters, and extend the kernel machinery to learned feature maps, without ever putting a number in front of a measurement.

That experiment now exists, and it is running somewhere else: in the sibling project **ssl-lab** ([github.com/pleiadian53/ssl-lab](https://github.com/pleiadian53/ssl-lab)), on real biological data. This chapter says what it is, why that domain was chosen, what has come back so far, and what is meant to return to this repository once it matures.

Two framings need to be set straight at the outset, because the honest version is more useful than the flattering one.

**First: the theory in this repository has never been executed.** Part 2 §4.4 proposes the Lie-algebra parameterization, where the policy emits an algebra element $X$ and the exponential map $\exp$ carries it into the group. Part 2 §3.6 lists the Hamiltonian operator $\exp(\tau J \nabla H)$ and marks it "(Planned)." Neither is implemented. There is no matrix exponential anywhere in this codebase. The operators in `src/grl/operators/` are raw affine, field, and kernel maps. `ComposedOperator` exists in `src/grl/operators/base.py`, but nothing outside the test suite constructs one, and it is not exported from the package's `__init__.py`. The algebraic structure that Part 2 §4 spends a section on is, as of today, a section.

**Second: ssl-lab is ahead of this repository on the operator, and its headline result is negative.** It built the exponential-map parameterization, it composes in the group, and it measured what came out. What came out is that the operator ties the baseline it was meant to beat. That is worth reporting precisely, because a negative result obtained against a real ground truth is the thing this series has been missing, and it is more informative than a fourth untested formalism would be.

---

## 2. Why Cell Perturbation

The choice looks eccentric for a reinforcement learning project. It is not, and the reason is the point of the chapter.

The GRL thesis is that an action is not a symbol you select but a function you construct: $s' = \hat{O}(s)$. Most RL benchmarks resist this framing. In CartPole the action really is a symbol, and dressing it as an operator adds machinery without adding truth. The framing earns its keep only where the action genuinely *is* a transformation of the whole state.

Gene perturbation is such a domain, on four counts.

**The action is natively an operator.** The intervention is CRISPRa activation of a target gene in a cell. The state $s$ is the cell's expression profile across roughly 20,000 genes. Activating one gene does not add a displacement to the state; it propagates through the regulatory network and transforms the entire profile. There is no sense in which "activate gene X" is a menu item. It is a map $\mathcal{S} \to \mathcal{S}$, exactly Definition 1 of Part 2.

**Real data exists at scale.** Norman et al. 2019 measured K562 cells under 237 perturbation classes: 105 single-gene activations, 131 two-gene pairs, and one non-targeting control.

**The combinatorial structure tests the algebra directly.** This is the decisive reason. Part 2 §4 claims that operator composition carries structure, and Part 3 never tests it. Norman's design holds out two-gene combinations, so a model trained on singles and some pairs must predict unseen pairs. Predicting a pair *is* composing two operators. The dataset is a test of $\hat{O}_B \circ \hat{O}_A$ against measurement.

**There is ground truth for the interaction itself.** Biology has a name and a measurement for what happens when two perturbations fail to compose additively: **genetic interaction**, or epistasis. So the commutator is not merely a quantity the model can compute. It is a quantity that can be correlated against a number the wet lab measured independently.

That last point is what makes the domain a minimum viable experiment rather than a demonstration. The claim under test has an external referent.

---

## 3. What Is Being Restricted: the $T = 1$ Corner

ssl-lab's setup is a *special case* of the framework in Parts 1-3, and being exact about which dials are pinned is what makes the generalization path legible later.

The latent operator is the object being learned. A frozen JEPA encoder $E$ maps the expression profile to $z = E(s) \in \mathbb{R}^{256}$, and the operator acts there:

$$z' = f_\theta(z), \qquad f_\theta(z) = A_\theta z, \qquad A_\theta = \exp(M_\theta).$$

This is Part 2 §4.4's parameterization, built. The generator $M_\theta$ is the Lie algebra element; $A_\theta$ is the group element. In ssl-lab's current round each target gene $g$ owns a generator $M_g$ learned directly, initialized at zero so that $A_g = \exp(0) = I$ and every operator starts at the identity.

Against the general framework, the following are pinned:

| Dial | General framework | ssl-lab's instance |
|---|---|---|
| Number of applications $T$ | A trajectory $z_0 \to z_1 \to \cdots$ | $1$: control population to perturbed population |
| Operator family | Affine, field, potential, kernel, attention | Linear only, $v(z) = M z$ |
| Policy $\pi: \mathcal{S} \to \Delta(\Theta)$ | State chooses the parameters | None. One operator per intervention, shared by every cell |
| Reward | Drives the Bellman recursion | None. No environment, no return |
| Composition rule | Ordered, a time-ordered product | Unordered: both genes are applied at once |
| Training signal | TD / actor-critic | Distribution matching between two unpaired populations |

Two of those restrictions are worth dwelling on, because they are not arbitrary simplifications. They are what the domain forces.

**There is no clock.** The Koopman framing that licenses the linear operator is a statement about *dynamics*, and this setting has no trajectory. A control population and a perturbed population are observed, and that is all. The operator models the time-1 map of a response that has already settled; the "1" is bookkeeping, not a measurement.

**The pairs are unpaired.** Reading a cell's transcriptome by sequencing destroys the cell. So for any given cell, $s$ and $s'$ never both exist. The equivariance loss that Part 3 §8 leans on,

$$\mathcal{L}_{\text{equiv}} = \lVert E(\hat{O}_\theta(s)) - f_\theta(E(s)) \rVert^2,$$

**cannot be computed at all** in this domain. There is no pair to take the difference of. ssl-lab replaces it with marginal matching between the two clouds via the energy distance, and observes that the per-pair squared error is the special case where the coupling happens to be given. That is a genuine contribution flowing back: it says the equivariance loss is not the general form, but the lucky corner of one.

**A consequence that should be recorded as a limit, not a detail:** $A = \exp(M)$ is always invertible, hence a diffeomorphism, hence it cannot split a connected cloud into two. Where a perturbation drives cells to a bimodal fate, this operator family cannot express the outcome. Invertibility is sold in Part 2 §4.3 as a feature. Here it bites.

---

## 4. The Bracket Is the Interface

This is the part that matters most for GRL, and it is why the biology is not a detour.

Part 2 §4.4 gestures at the Lie algebra without saying what the bracket buys. ssl-lab makes it concrete.

Start with the object itself. Given two generators $M_A$ and $M_B$, both $D \times D$ matrices, their **commutator** (also called the bracket) is

$$[M_A, M_B] = M_A M_B - M_B M_A.$$

Read it as a question: *does the order of multiplication matter?* For ordinary numbers it never does, so $3 \times 5 - 5 \times 3 = 0$. For matrices it usually does, and the bracket measures by how much. When $[M_A, M_B] = 0$ the two commute and order is irrelevant. When it is large, order matters a lot.

The reason this is the right object is a theorem. For near-identity operators, generators commute **if and only if** composition is additive, and the Baker-Campbell-Hausdorff (BCH) series says exactly what the correction is when they do not:

$$\log(\exp(A)\exp(B)) = A + B + \tfrac{1}{2}[A,B] + \tfrac{1}{12}([A,[A,B]] + [B,[B,A]]) + \cdots$$

Read the right-hand side as "the naive answer plus corrections." The naive answer is $A + B$, which is what you would get if you could just add the two generators. Every term after it is built from brackets and nothing else. If $[A,B] = 0$, every correction vanishes and $\exp(A)\exp(B) = \exp(A+B)$ exactly. So non-commutativity is not an analogy for non-additivity. It is what the group product computes.

Now the observation that connects the two projects. **The same bracket has two readings, and which one you get depends entirely on whether your actions are simultaneous or sequential.**

The hinge is a symmetry. Swap the labels $A \leftrightarrow B$ and ask what survives. The bracket flips sign, i.e. it is **odd** under the swap:

$$[B,A] = BA - AB = -(AB - BA) = -[A,B].$$

A worked case makes this concrete. Take two $2 \times 2$ generators, a shear and a scaling:

$$A = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}, \qquad B = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}.$$

Multiplying them both ways gives

$$AB = \begin{pmatrix} 0 & -1 \\ 0 & 0 \end{pmatrix}, \qquad BA = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix},$$

so $[A,B] = AB - BA = \begin{pmatrix} 0 & -2 \\ 0 & 0 \end{pmatrix}$, while $[B,A] = \begin{pmatrix} 0 & 2 \\ 0 & 0 \end{pmatrix}$. Same size, opposite sign. That is the oddness, in one calculation: **the magnitude of the bracket does not care which gene you call $A$, but its sign is nothing but a statement about which one you applied first.**

The consequence is the whole point. If two interventions land *simultaneously*, there is no "first," so there is no fact of the matter for the sign to encode. The odd part cancels by construction, and only the swap-**even** part of the algebra survives into anything you could measure. Its leading term is a double commutator, whose size is governed by $\lVert [M_A, M_B] \rVert_F^2$, where $\lVert \cdot \rVert_F$ is the **Frobenius norm**, the square root of the sum of the squares of all a matrix's entries. It is just "how big is this matrix," collapsing the whole thing to one non-negative number and discarding the sign along with everything else. In the worked case above, $\lVert [A,B] \rVert_F = \lVert [B,A] \rVert_F = 2$: identical, as the symmetry demands.

So a simultaneous perturbation can see the *magnitude* of non-commutativity and never its *sign*.

If the actions are *sequential*, the order is a real, observable fact. The time-ordered product does not collapse, and the odd part becomes visible.

| | ssl-lab: perturbation | GRL: world model |
|---|---|---|
| Number of actions $T$ | $1$ combined intervention | A sequence $c_0, c_1, \ldots$ |
| Composition | Unordered, so the product is swap-even | Ordered, a time-ordered product |
| What the bracket means | **Epistasis**: departure from additive | **Path-dependence**: the order of a plan matters |
| Part of the algebra probed | Even only, $\sim \lVert [M_A, M_B] \rVert_F^2$ | Even *and* odd; the raw bracket $\tfrac{1}{2}[M_A,M_B]$ becomes visible |
| Ground truth | Norman's measured genetic interaction | Reward, realized trajectory |

Read the table as a division of labour. ssl-lab takes the even half of the bracket algebra to a measurement biology already trusts. GRL inherits the same generators and tests the odd half against trajectories. Nothing about "two genes" is doing the simplifying work here. **The simultaneity is.** Lift that one restriction and the same mathematical object starts answering a question about plans instead of a question about cells.

Read ahead before relying on this: ssl-lab has since run that even-half test, and **it failed**. §5.3 reports the result and §5.4 draws the line around what it does and does not license. The division of labour still describes the structure correctly, i.e. which half of the algebra each setting can see. What it no longer describes is a validated half handing off to an untested one. Both halves are now open, and the even one has a negative result attached.

This also supplies a concrete instruction for this repository, recorded here and detailed in `dev/planning/action_operator/`: `ComposedOperator` should be rebuilt as the time-ordered product. Writing $c_k$ for the action taken at step $k$, and $M_{c_k}$ for the generator that action configures, a plan of $T$ actions carries the latent from $z_0$ to $z_T$ by

$$z_T = \left( \prod_{k=T-1}^{0} \exp(M_{c_k}) \right) z_0 = \exp(M_{c_{T-1}}) \cdots \exp(M_{c_1}) \exp(M_{c_0}) z_0.$$

The index running *downward*, from $T-1$ to $0$, is doing real work and is easy to misread. Matrices apply right-to-left, so the factor nearest $z_0$ acts first. The **first** action $c_0$ sits at the **rightmost** position, and the last action sits at the left. Written out for $T = 2$, "do $c_0$, then $c_1$" is $\exp(M_{c_1})\exp(M_{c_0}) z_0$, which reads backwards compared to the order you would say it aloud. Swapping those two factors is not a cosmetic change. By the BCH series it changes the result by $[M_{c_0}, M_{c_1}]$ to leading order, which is precisely the path-dependence the sequential setting exists to measure.

Any sum-inside-the-exponential form, $\exp(\sum_k M_{c_k})$, should be kept only where it is labelled for what it is: a commutative approximation, valid when the brackets vanish and wrong otherwise. Any model that aggregates a sequence of interventions by summing their coefficients has silently assumed they all commute.

---

## 5. What Has Come Back So Far

The status below is current as of 2026-07-16. ssl-lab maintains an append-only results ledger; that ledger, not this chapter, is authoritative.

### 5.1 The operator ties the flow, and both lose to the baseline

On the primary endpoint, the $\Delta$-correlation between predicted and true mean expression shift on each held-out perturbation's top differentially-expressed genes, across 20 held-out two-gene combinations and 3 seeds:

| Configuration | $\Delta r$ |
|---|---|
| Transport flow (control to outcome) | 0.648 |
| **Action operator** (deterministic) | 0.645 |
| Conditional NB-VAE (the baseline) | **0.766** |

With simultaneous 95% intervals from a joint bootstrap, operator minus transport flow is $-0.003$, interval $[-0.030, +0.024]$: a dead tie. Operator minus baseline is $-0.121$, interval $[-0.228, -0.014]$: the baseline wins, significantly. A from-scratch conditional variational autoencoder beats the operator framework by about $0.12$ on the task both were built for.

### 5.2 Why the tie is not evidence for anything

The temptation is to read "linear operator ties expressive nonlinear flow" as support for the Koopman premise. ssl-lab refuses that reading, and the refusal is the most transferable methodological result of the project.

A ceiling analysis decomposed the pipeline into an oracle ladder, replacing each stage in turn with a perfect one and measuring what the metric could reach:

| Step | From | To | Lost |
|---|---|---|---|
| Encoder, under a linear readout | 1.000 | 0.852 | 0.148 |
| Decoder | 0.852 | 0.679 | 0.173 |
| Transition | 0.679 | 0.648 | 0.031 |

**The transition stage is saturated.** It scores $0.648$ against a ceiling of $0.679$. There is at most $0.031$ of headroom in the stage where the operator lives, no matter how that stage is modelled. A metric with three points of room cannot distinguish a linear transition from a nonlinear one. The tie was preordained by the measurement instrument, and reporting it as evidence would have been reporting the instrument's noise floor.

The bottleneck is the decoder, at $0.173$, and the representation was never the problem: a plain linear readout of the frozen latent scores $0.852$, well above the baseline's $0.766$. The information was in the latent the whole time. The readout was losing it.

For this repository the lesson generalizes past the biology. **Before attributing a result to your operator, prove the stage the operator occupies has room to move the metric.** That is a discipline Parts 1-3 do not currently impose on themselves.

### 5.3 The bracket-equals-epistasis test: refuted

This is the round that would most directly justify Part 2 §4's algebra section. It ran, and **the claim did not survive.**

The pre-committed claim was

$$\mathrm{corr}\big(\lVert [M_A, M_B] \rVert_F, \mathrm{GI}(A,B)\big) > 0.$$

In words: across gene pairs, the bigger the model's bracket, the bigger the interaction biology actually measured. The left argument is the model's quantity, the size of the commutator from §4. The right argument, $\mathrm{GI}(A,B)$, is the measured **genetic interaction**, i.e. how far the real pair's effect departs from the sum of the two single effects. If the theory is right, these two should rise together.

Two statistics report it. **Spearman $\rho$** is a rank correlation: it asks whether the pairs with the largest brackets are the same pairs with the largest measured interaction, without assuming the relationship is a straight line. It runs from $-1$ (perfectly opposed) through $0$ (no relation) to $+1$ (perfectly aligned). The **permutation $p$** is how often you would see a correlation at least this strong by luck alone: shuffle the labels thousands of times, recompute $\rho$ each time, and count what fraction of the shuffles beat the real one. A small $p$ means chance rarely does this well.

| Test | In-sample ($n = 91$) | Held-out ($n = 20$) |
|---|---|---|
| **Raw bracket vs measured interaction** (*the endpoint*) | $-0.070$ ($p = 0.75$) | $-0.262$ ($p = 0.87$) |
| Raw vs directional part (*post-hoc*) | $+0.114$ | $+0.042$ |
| Normalized bracket vs interaction (*post-hoc*) | $-0.141$ | $-0.344$ |
| Normalized vs directional (*post-hoc*) | $-0.023$ | $-0.120$ |

Four tests, both splits, all 111 pairs: nothing. The two post-hoc rescues also failed, which makes the negative **stronger, not weaker**, since a claim that cannot be saved by retrofitting is a claim that was simply wrong. The pre-registered kill rule fired on the in-sample null and the round stopped there.

Two things about how this was measured are worth more to GRL than the result.

**The near-identity trap, which nearly produced a fake refutation.** The first run returned the same null, but it was *uninformative*, and only a diagnostic caught it. The learned generators had median $\lVert M_g \rVert_F = 12.4$, giving $\lVert A - I \rVert \approx 11$. That is nowhere near the identity. Recall from §4 that the bracket-equals-non-additivity theorem holds *for near-identity operators*; far from the identity, large generic matrices fail to commute simply because they are large, and indeed the bracket correlated at $\rho = 0.916$ with $\lVert M_A \rVert \cdot \lVert M_B \rVert$. It had become a pure magnitude readout. It was measuring how strong each perturbation is, not how the two interact. Sweeping the least-action weight until $\lVert M_g \rVert_F = 1.10$ brought the model back inside the regime where the math applies; the endpoint did not move. **The refutation is credible precisely because it was re-measured where the theorem is valid.** The trainer now gates itself on this premise rather than trusting it.

**The requirements conflict, which is the finding worth more than the refutation.**

| Least-action weight | $\lVert M_g \rVert_F$ | Near-identity gate | $\Delta r$ |
|---|---|---|---|
| $10^{-4}$ | $12.40$ | **failed** | **0.629** |
| $10^{-1}$ | $1.10$ | passed | **0.497** |

Fitting the response demands a **large** generator. An interpretable bracket demands a **small** one. The operator cannot do both, and it pays $0.13$ of $\Delta r$ to buy a valid bracket. That is not a tuning problem. It indicts the premise the whole design rested on: *"effects are small shifts on a large baseline."* That premise is true of **expression**, where a few genes move among thousands. It does **not** survive the encoder into **latent** coordinates, where the same response is a large rotation. **The near-identity prior was imported from the wrong space.** Part 2 §4.4's exponential map and Part 2 §4.2's least-action energy both quietly assume the operator lives near the identity. This is the first evidence about *which space* that assumption is a statement about, and the answer is that it does not transfer across an encoder for free.

The diagnosed cause of the null itself is **identifiability**. Each $M_g$ carries $256 \times 256 = 65{,}536$ parameters, and the loss only ever asks it to push one cloud of control cells onto one target marginal. The bracket lives in directions that loss never constrains, so it is free to be arbitrary. Even at near-identity it still correlated at $\rho = 0.63$ with the norm product. A shared low-rank basis, which would confine brackets to $\mathrm{span}([B_i, B_j])$, is the obvious next parameterization, but it needs pre-registering as a new experiment rather than bolting on as a rescue.

For contrast: the algebra itself is fine. On synthetic generators the bracket tracks non-additivity at Spearman $0.89$. And the biology has real signal, with the pairs spanning $\lambda \in [0.655, 1.266]$ (9 super-additive, 7 sub-additive, 4 neither). The mathematics works and the biology varies. **What failed is the attempt to connect them through generators this under-determined.**

### 5.4 What the refutation does and does not license

A negative result is only useful if its scope is stated precisely, so here is the scope.

**The mathematics is untouched, and cannot be otherwise.** That $[A,B] = 0$ if and only if $\exp(A)\exp(B) = \exp(A+B)$ near the identity is a theorem. Data does not get a vote. It was confirmed numerically on synthetic generators at Spearman $0.89$, and if it had failed there the bug would have been in the code, not the world. So nothing here says the bracket is the wrong object for compositional structure.

**What was refuted is an empirical bridge**, and a narrow one: that *these* generators, dense and per-gene, fit *this way*, by matching marginals of unpaired populations under a single simultaneous intervention, produce brackets that track measured epistasis. Every qualifier in that sentence is load-bearing.

The useful question is therefore not "is the theory dead" but **which of the diagnosed causes travel to another domain.** Take digital phenotyping as the concrete comparison, meaning passive sensor streams from a phone or wearable encoded into a latent state, with interventions such as a medication change, a therapy session, or a behavioral nudge.

| Diagnosed cause here | Does it follow to a sequential domain? |
|---|---|
| **Identifiability.** Each $M_g$ is $65{,}536$ parameters constrained only to push one cloud onto one marginal, so the bracket is free in every direction the loss ignores | **Much weaker.** A trajectory constrains the operator at every step. Constraints scale with subjects times timesteps, not one marginal per condition |
| **Unpaired observations.** Sequencing destroys the cell, so $\mathcal{L}_{\text{equiv}}$ cannot be computed and only marginal matching is available | **Gone.** The same subject is observed at $t$ and $t+1$. The paired equivariance loss of Part 3 §8 is simply available |
| **The Koopman form without the Koopman training.** The encoder was trained by masked prediction, never for dynamics, then frozen | **Fixable, and only there.** A domain with a clock permits training the encoder on the dynamics. Cell perturbation structurally cannot, having no trajectory to train on |
| **The near-identity prior imported from the wrong space** | **This one follows you.** It is a statement about encoders, not about biology |

That last row deserves the emphasis rather than the first three. "Effects are small shifts on a large baseline" was true in expression coordinates and false in latent coordinates. Any encoder whose objective never mentioned dynamics can do the same thing in any domain, placing states that are semantically adjacent far apart because nothing asked it not to. The gain is not that the problem disappears elsewhere. It is that it is now **checkable**: measure $\lVert M \rVert_F$ and $\lVert A - I \rVert$ before trusting a bracket, and gate on them. That diagnostic is the most portable artifact this round produced.

**A sequential domain is not a retry of this experiment.** It is a different test. Cell perturbation is simultaneous, so as §4 established, only the swap-even half of the bracket was ever observable, and that half reads as epistasis. Sequencing the actions exposes the swap-odd half, which reads as path-dependence, and that is GRL's own claim. It has never been tested, here or anywhere. So the honest position is that the even half has been tried once, in the domain least able to support it, and failed for reasons that are largely diagnosable and largely local; while the odd half remains untested and lives in domains where three of the four failure modes are structurally weaker.

**The caution against reading that too warmly.** Three levers have now failed in a row, and a fourth should inherit a lower prior, not a fresh one. The mechanism that killed this round, a high-capacity operator fit to a thin training signal, recurs anywhere supervision is weak, and digital phenotyping supplies its own thin signals: missing sensor data, noisy labels, non-stationarity, and heterogeneity between subjects. The correct move is not to assume the next domain rescues the idea. It is to carry the diagnostic forward and let it fail fast again if it is going to.

### 5.5 A finding this repository should adopt

Three separate times in ssl-lab, a change that improved the training objective made the model worse on the endpoint: an optimal-transport coupling, a dispersion anchor, and a stochastic operator arm whose energy distance fell while its downstream score dropped. **A better training loss is not a better model.** Part 2 §4.2 offers a subadditive composition-energy bound as though the energy functional were self-evidently the right thing to minimize. Three instances of that inference failing on real data is worth more than the bound.

---

## 6. What Comes Home, and When

The intended flow is that ssl-lab develops the constructs against a real domain, and once they hold, the abstractions refactor into `src/grl/`. §5.3 changes what "hold" means for one of them, so the candidates are now ranked by what actually survived:

- **The near-identity diagnostic. This is the most valuable artifact of the round**, and it was not even on the original list. Measure $\lVert M \rVert_F$ and $\lVert A - I \rVert$, and gate on them, because a bracket read outside the near-identity regime is a magnitude readout wearing a theorem's clothes. It nearly produced a fake refutation here. Any GRL code that exponentiates a generator should carry this check from the first commit.
- **The exponential-map parameterization.** Per-action generators with $A = \exp(M)$, zero-initialized to the identity, with a least-action penalty on $\lVert M \rVert_F^2$. This is Part 2 §4.4 in working code, and it is buildable and stable independent of the negative result. Carry the caveat with it: the near-identity prior is a claim about a *space*, and it did not survive the encoder here.
- **The group composition rule**, generalized from the symmetric product ssl-lab uses for simultaneous pairs to the time-ordered product GRL needs for sequences. Untouched by the refutation, which tested the even half; the time-ordered product is how the odd half becomes visible at all.
- **The bracket as a first-class computable.** $[M_A, M_B]$ and its Frobenius norm. **Demoted to a diagnostic, not a training signal.** As a quantity to inspect it is fine. As a thing to fit, it failed here, and the diagnosed cause was identifiability rather than anything about biology. If it returns as a training signal it needs a parameterization that constrains it, e.g. a shared low-rank basis confining brackets to $\mathrm{span}([B_i, B_j])$, and it needs pre-registering as a new experiment rather than bolting on as a rescue.
- **Marginal matching via energy distance**, for the regime where paired transitions do not exist, with per-pair error kept as the given-coupling special case. Lower priority for GRL, which usually has the pair, but conceptually clarifying.

The trigger for the `examples/` convention is worth stating plainly, because it governs when this repository grows a topic directory. Per the working convention, `examples/<topic>` exists to put code into real use *while* the reusable abstraction is built under `src/`, so the examples test whether the abstraction is correct rather than demonstrating it after the fact. That means a topic directory earns its place the moment there is a generator or exponential map in `src/grl/operators/` for it to exercise. Creating one before then would produce a plan filed in a directory reserved for software. The plan lives in `dev/planning/action_operator/` until the code arrives.

What returns is not a toy coming home to the real thing. It is a parameterization, a composition rule, a diagnostic, and a bracket-versus-interaction result that was put in front of real data and told to hold up. It did not.

---

## 7. Where This Leaves the Series

Parts 1-3 argued that actions should be operators, showed how to construct them from parameters, and extended the kernel machinery to learned feature maps. Every step was formal. The series ended with a frontier and no experiment.

Part 4's contribution is that the experiment now exists and has produced findings that constrain the theory rather than decorate it:

1. **The exponential map works and is buildable.** Part 2 §4.4 is not aspirational. It runs.
2. **The bracket remains the right object mathematically, and the one attempt to tie it to a measurement failed.** Both halves of that sentence are load-bearing. The theorem is a theorem, and confirmed synthetically. The empirical bridge, from a learned generator's bracket to a measured interaction, broke on identifiability, in the domain least able to support it. §5.4 draws the boundary: the even half has one negative result attached; the odd half, which is GRL's own claim, is untested.
3. **The operator did not beat a simpler baseline**, and, more importantly, the domain lacked the headroom to tell whether it could have. Both facts stand.
4. **Ceiling analysis before attribution** is a discipline the operator framework needs and does not yet have.
5. **A near-identity claim is a claim about a coordinate system**, and it does not cross an encoder for free. This is the finding this series should absorb most carefully, because Part 2 §4.4's exponential map and §4.2's least-action energy both assume the operator lives near the identity without ever asking near the identity *of what space*. Here the assumption held in the observation coordinates and failed in the latent ones, and buying it back cost $0.13$ of the headline metric. Measure it; do not assume it.
6. **Sequential actions remain the frontier**, unchanged, and are now the *only* place the compositional claim can still be tested as intended. Crediting which step of a multi-action skill earned the outcome is the classic hard problem, and it returns the instant operators compose in time.

The honest summary is that the operator formalism is now known to be *implementable*, known *not* to be better on the one benchmark that could measure it, and its compositional claim is refuted in the even half and untested in the odd. Whether that reads as a dead end or as a well-scoped remaining question depends entirely on whether the next experiment can supply what this one structurally could not: a clock, paired transitions, and an encoder trained on the dynamics it is asked to linearize.

---

**Related documents:**

- Part 1: [The GRL0 Gap](./01-the-grl0-gap.md)
- Part 2: [Action Operator Formalization](./02-action-operator-formalization.md). Sections 3.6 and 4.4 are what ssl-lab built.
- Part 3: [From Fixed Kernels to Learned Kernels](./03-from-fixed-to-learned-kernels.md). Section 10's frontier is this chapter's subject.
- Refactor plan and milestone tracking: `dev/planning/action_operator/grl-4-ssl-lab-refactor-plan.md`

**In ssl-lab** ([github.com/pleiadian53/ssl-lab](https://github.com/pleiadian53/ssl-lab)):

| Document | What it covers |
|---|---|
| `docs/action_operator/` | The on-ramp series: operator gallery, the algebra of composition, and the JEPA connection |
| `docs/operator_world_models/` | The world-model reading: conditioning JEPA on actions, generator bases, a worked clinical example |
| `examples/perturbation_response/docs/conditional-flow-jepa/07-modeling-the-transition-action-operators.md` | The operator chapter: the commuting square, the unpaired crack, marginal matching |
| `examples/perturbation_response/docs/conditional-flow-jepa/09-why-the-operator-is-linear-koopman.md` | The Koopman argument for why the operator may be linear, and the premise the encoder never satisfied |
| `examples/perturbation_response/docs/conditional-flow-jepa/results-ledger.md` | The authoritative running scoreboard |
| `docs/experimental-method/07-the-ceiling.md` | The ceiling analysis |
