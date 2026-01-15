# Interactive Notebooks

Welcome to the GRL interactive demonstrations! These Jupyter notebooks provide hands-on visualizations for understanding reinforcement fields.

---

## 📚 Notebook Series

This is a **3-part progressive series** building intuition from classical vector fields to GRL's functional fields:

### **[Part 1: Classical Vector Fields](01_classical_vector_fields/)**
🔄 **In Development** | ⏱️ ~15-20 min

Arrows at each point, gradient fields, optimization basics.

### **Part 2: Functional Fields** 
📋 **Planned** | ⏱️ ~20-25 min

Functions as vectors, RKHS foundations, explicit comparison.

### **Part 3: Reinforcement Fields in GRL**
📋 **Planned** | ⏱️ ~25-30 min

Particle memory, Q⁺ emergence, policy inference, learning dynamics.

---

## 🗺️ Complete Roadmap

**See the full roadmap with learning paths, FAQ, and development status:**

👉 **[Notebook Series Roadmap](https://github.com/pleiadian53/GRL/blob/main/notebooks/README.md)**

---

## 💡 Quick Start

**Best viewing experience:**
1. Click notebook links above (rendered on this site)
2. Math and plots display correctly
3. No GitHub timeout issues!

**Want to run interactively?**
```bash
git clone https://github.com/pleiadian53/GRL.git
cd GRL/notebooks
jupyter notebook
```

---

## 📖 Related Resources

- **[Tutorial Series](../tutorials/README.md)** — Mathematical depth
- **[Implementation Guide](../implementation/README.md)** — Technical specs
- **[Quantum-Inspired Extensions](../quantum_inspired/README.md)** — Advanced topics

---

## 📍 Note for Contributors

These notebooks are **rendered copies** from `/notebooks/` in the repository.

**Development workflow:**
1. Develop in `/notebooks/` (primary location)
2. Copy to `/docs/notebooks/` when ready to publish
3. GitHub Actions deploys automatically

**Why two locations?**
- `/notebooks/` — Source of truth, easy to find
- `/docs/notebooks/` — Rendered for reliable display
