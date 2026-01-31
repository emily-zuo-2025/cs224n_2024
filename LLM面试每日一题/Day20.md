# Day 20

## Question:
### <mark>Compare the objectives and boundaries of SFT, PEFT, and RLHF</mark>

## Answer:
### Core Comparison: Essential Differences Among Three Fine-tuning Methods

**SFT (Supervised Fine-Tuning)**, **PEFT (Parameter-Efficient Fine-Tuning)**, and **RLHF (Reinforcement Learning from Human Feedback)** are three different model optimization methods, each with distinct objectives and boundaries.

---

### 1. Training Objective Differences 🎯

#### SFT: Learning Task Mapping

**Core Objective**: Through supervised learning, enable the model to learn the mapping from input to output.

**Technical Principle**:

SFT is based on **next token prediction**, which maximizes the probability of correct tokens:

$$\mathcal{L}_{\text{SFT}} = -\sum_{t=1}^{T} \log P_{\theta}(y_t | x, y_{<t})$$

Gradient update for all parameters: $\theta_{t+1} = \theta_t - \eta \nabla_{\theta} \mathcal{L}_{\text{SFT}}$
**Boundaries**:

- ✅ Suitable for tasks with **clear answers** (translation, summarization, code generation)
- ❌ Not suitable for **subjective preference** tasks (conversational style, creative writing)
- ❌ Easy to learn **biases and errors** in the data

---

#### PEFT: Parameter-Efficient Adaptation

**Core Objective**: Freeze the original model and train only a small number of parameters to adapt to new tasks.

**Technical Principle (Using LoRA as Example)**:

Assume weight updates are low-rank:

$$h = W_0 x + \frac{\alpha}{r} B A x$$

Where:

- $W_0$: Pre-trained weights (frozen)
- $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$: Low-rank matrices (trainable)
- $r \ll \min(d, k)$: Rank (typically 8-64)

**Parameter Comparison**: Assume $d = k = 4096$, $r = 8$:

- Full fine-tuning: 16.7M parameters
- LoRA: 65K parameters (reduced by 99.6%)

**Why is it effective?**

1. **Intrinsic dimensionality hypothesis**: Task adaptation occurs in low-dimensional subspace
2. **Implicit regularization**: Restricts overfitting by limiting rank
3. **Parameter sharing**: Low-rank decomposition enables information sharing

**Boundaries**:

- ✅ Suitable for resource-limited, multi-task, rapid experimentation scenarios
- ❌ Not suitable for large distribution shift tasks
- ❌ Extreme cases may have slightly lower performance than full fine-tuning (1-3%)

---

#### RLHF: Aligning with Human Preferences

**Core Objective**: Not learning correct answers, but learning human preferences to maximize human satisfaction.
**Technical Principle (Three Stages)**:

**Stage 1: Train Reward Model**

Given preference pairs $(x, y_w, y_l)$, train the reward model:

$$\mathcal{L}_{\text{RM}} = -\mathbb{E}\left[\log \sigma(r_{\phi}(x, y_w) - r_{\phi}(x, y_l))\right]$$

Based on Bradley-Terry model: $P(y_w \succ y_l) = \sigma(r(y_w) - r(y_l))$

**Stage 2: PPO Optimization Strategy**

$$\max_{\pi_\theta} \mathbb{E}\left[r_{\phi}(x, y) - \beta \cdot D_{\text{KL}}(\pi_\theta || \pi_{\text{ref}})\right]$$

**Role of KL Divergence Constraint**:

- Prevent reward model overfitting
- Maintain language fluency
- Stabilize training process

**PPO Clipping Mechanism**:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}\left[\min\left(\rho_t(\theta)\hat{A}_t, \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

Where $\hat{A}_t = r_{\phi}(x, y) - V(x)$ is the advantage function, $\rho_t = \frac{\pi_\theta}{\pi_{\text{old}}}$ is the importance ratio.

**Boundaries**:

- ✅ Suitable for tasks with strong subjective preferences (conversational style, emotional expression)
- ✅ Can capture difficult-to-formalize biases (politeness, empathy, credibility)
- ❌ Not suitable for tasks requiring objective facts (mathematics, code correctness)
- ❌ Training is unstable, with risk of reward hacking

---

### 2. Data Requirement Differences 📊

#### SFT Data Requirements

**Format**: `(prompt, response)` - requires standard answers

**Data Volume Rule**: Performance ∝ log(Data Size)

- 1K samples → basic capability
- 10K samples → better results
- 100K samples → near optimal

**Characteristics**: Clear answers required, batch labeling possible, can use synthetic data

#### PEFT Data Requirements

**Format**: Same as SFT, but requires less data

**Why require less data?**

$$\text{Generalization Error} \propto \frac{\text{Model Complexity}}{\text{Data Volume}}$$

PEFT reduces model complexity, thus requiring less data to achieve the same effect.

**Data Volume Recommendations**:

- 500 samples → visible results
- 1K-5K samples → better results

**Characteristics**: Low data requirements, fast iteration, suitable for small sample scenarios

#### RLHF Data Requirements

**Format**: `(prompt, response_chosen, response_rejected)` - requires human ranking

**Why use preference pairs?**

- Easier to judge relative comparisons than absolute scoring
- Higher annotation consistency
- Fits Bradley-Terry model assumptions

**Data Volume Requirements**:

- 10K preference pairs → basic results
- 50K preference pairs → better results (InstructGPT used 60K)
- 100K+ preference pairs → SOTA results

**Annotation Cost**: 50K pairs × 2 minutes ≈ 1667 hours

**Cost Reduction**: AI-assisted annotation, active learning, consistency checks

**Characteristics**: Captures subjective preferences, high annotation cost, requires continuous collection

---

### 3. When to Choose? 🤔

#### Choose SFT Scenarios

- ✅ Clear standard answers exist
- ✅ Sufficient data with high quality
- ✅ Adequate storage (50-80GB)

**Examples**: Code generation, machine translation, text summarization, mathematical reasoning

#### Choose PEFT Scenarios

- ✅ Resource-constrained (storage 10-15GB)
- ✅ Multi-task scenarios
- ✅ Small data volume (<5K samples)
- ✅ Need fast iteration (2-3x faster)

**Examples**: Customer service robots (multi-product lines), medical assistants (multi-departments), education assistants (multi-subjects)

#### Choose RLHF Scenarios

- ✅ Tasks with strong subjectivity
- ✅ Need to align with human values
- ✅ Adequate budget (5-10x cost of SFT)

**When worthwhile**: Large user base, unclear explicit rules, high safety requirements

**Examples**: ChatGPT, content creation assistants, customer service

---

### 4. Effect and Cost Trade-offs 💰
#### Performance and Cost Comparison Table

| Method | Performance | Storage | Training Time | Total Cost | Application |
|--------|-------------|---------|---------------|------------|-------------|
| **SFT** | Good task performance | 50-80GB | 2-4h | $2K-6K | Standard answer tasks |
| **PEFT** | 95-98% of SFT | 10-15GB | 1-2h | $200-3K | Resource-limited scenarios |
| **RLHF** | Best subjective tasks | 80GB+ | 8-16h | $20K-60K | Large-scale C-end products |

#### Cost Breakdown:

- **SFT**: Compute 50-100 + data annotation 2K-5K
- **PEFT**: Compute 10-30 + data annotation 200-2.5K
- **RLHF**: Compute 500-1.5K + data annotation 20K-50K

---

### 5. Practical Application Strategies 🎯

#### Combined Use (Most Common)

**Combination 1: SFT → PEFT**

- Base model (SFT general capability) + multiple lightweight adapters (PEFT for specific tasks)
- **Examples**: Customer service systems, code assistants

**Combination 2: SFT → RLHF (Strongest Effect)**

- **OpenAI's InstructGPT Pipeline**:
  1. Pre-train GPT
  2. SFT (13K instruction samples)
  3. RLHF (60K preference pairs)

**Why SFT first?** Direct RLHF training is unstable; SFT needed to establish basic capability.

**Combination 3: PEFT → RLHF**

- Compromise solution for resource-limited scenarios
- Note: PEFT capacity constraints may affect RLHF results

#### Decision Flow

**Step 1: Clarify Task**

- Clear answers exist? → SFT/PEFT
- Need subjective preferences? → RLHF

**Step 2: Evaluate Resources**

- Storage: <24GB → PEFT; 24-80GB → SFT; >80GB → RLHF
- Data: <1K → PEFT; 1K-10K → SFT/PEFT; >10K → All possible
- Budget: <$5K → PEFT; 5K-20K → SFT; >$20K → RLHF

**Step 3: Evaluate Team**
- Junior team → SFT/PEFT
- Advanced team → Can try RLHF

#### Decision Tree:

```
Does task have clear answers?
├─ Yes → Sufficient resources?
│         ├─ Yes → SFT
│         └─ No → PEFT
└─ No → Sufficient budget?
          ├─ Yes → SFT + RLHF
          └─ No → PEFT
```

---

## Summary

### Core Comparison Table

| Dimension | SFT | PEFT | RLHF |
|-----------|-----|------|------|
| **Objective** | Learn input-output mapping | Parameter-efficient adaptation | Align with human preferences |
| **Data Format** | (prompt, response) | (prompt, response) | (prompt, chosen, rejected) |
| **Data Volume** | 10K-100K | 500-5K | 10K-100K pairs |
| **Training Cost** | $2K-6K | $200-3K | $20K-60K |
| **Storage** | 50-80GB | 10-15GB | 80GB+ |
| **Best Scenario** | Clear standard answers | Resource-limited | Subjective preferences |
| **Limitations** | Learns data biases | May affect performance | High cost, unstable training |

**Key Takeaway**: Choose the appropriate method based on task characteristics, resource constraints, and team capabilities. In practice, combined use often yields the best results.

- Junior team → SFT/PEFT
- Advanced team → Can try RLHF

#### Decision Tree:

```
Does task have clear answers?
├─ Yes → Sufficient resources?
│         ├─ Yes → SFT
│         └─ No → PEFT
└─ No → Sufficient budget?
          ├─ Yes → SFT + RLHF
          └─ No → PEFT
```

---

## Summary

### Core Comparison Table

| Dimension | SFT | PEFT | RLHF |
|-----------|-----|------|------|
| **Core Formula** | $-\sum \log P(y_t \|\| x, y_{<t})$ | $W_0 x + \frac{\alpha}{r} BAx$ | $\max \mathbb{E}[r - \beta D_{\text{KL}}]$ |
| **Optimization Goal** | Maximize correct token probability | Low-rank adaptation | Maximize reward - KL constraint |
| **Parameter Updates** | All parameters | <1% parameters | All parameters (strategy gradient) |
| **Data Format** | (prompt, response) | (prompt, response) | (prompt, chosen, rejected) |
| **Data Volume** | 10K-100K | 0.5K-5K | 50K-100K preference pairs |
| **Storage Requirement** | 50-80GB | 10-15GB | 80GB+ |
| **Total Cost** | $2K-6K | $200-3K | $20K-60K |
| **Typical Applications** | Translation, summarization, code | Domain adaptation, multi-task | ChatGPT, dialogue systems |

### Three Fundamental Differences at Technical Level

#### 1. Optimization Objectives

- **SFT**: Supervised learning, optimizes likelihood function (assuming data is optimal)
- **PEFT**: Structural constraints, optimizes low-rank subspace (assumes low-dimensional adaptation)
- **RLHF**: Reinforcement learning, optimizes expected reward (assumes preference can be modeled)

#### 2. Mathematical Essence
#### 2. Mathematical Essence

- **SFT**: $\mathcal{L} = -\sum \log \frac{\exp(z_{y_t})}{\sum_j \exp(z_j)}$ → Maximize correct token logit
- **PEFT**: $W' = W_0 + BA$ → Weight updates occur on low-rank flow
- **RLHF**: $J = \mathbb{E}[r] - \beta D_{\text{KL}}$ → Exploration and exploitation trade-off

#### 3. Failure Modes

- **SFT**: Data has errors → Model learns errors
- **PEFT**: Large demand changes → Insufficient low-rank constraints
- **RLHF**: Reward model is exploited → reward hacking

---

### Practical Recommendations

**Beginners**: Start with PEFT to validate ideas → Confirm effective then use SFT

**Small teams**: Base model + LoRA adapters, avoid RLHF

**Large companies**: SFT builds foundation + RLHF aligns preferences + continuous iteration

**Decision Criteria**:

- Has standard answers → SFT/PEFT
- Need subjective preferences → RLHF
- Resource-limited → PEFT
- Pursue extreme performance → SFT + RLHF

**Key Takeaway**: Choose the appropriate method based on task characteristics, resource constraints, and team capabilities. In practice, combined use often yields the best results.
