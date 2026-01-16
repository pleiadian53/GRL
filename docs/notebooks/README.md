# Interactive Notebooks

Welcome to the GRL interactive demonstrations! These Jupyter notebooks provide hands-on visualizations for understanding reinforcement fields.

---

## 📚 Field Series: Understanding GRL Through Visualization

A progressive notebook series building intuition from classical vector fields to GRL's reinforcement fields.

### **[Field Series Overview](field_series/README.md)**

**Complete Series** | ⏱️ ~60-90 minutes total

| # | Notebook | Status | Time |
|---|----------|--------|------|
| 0 | [Introduction to Vector Fields](field_series/00_intro_vector_fields.ipynb) | ✅ Complete | ~10-15 min |
| 1 | [Classical Vector Fields](field_series/01_classical_vector_fields.ipynb) | ✅ Complete | ~20-25 min |
| 1a | [Vector Fields and ODEs](field_series/01a_vector_fields_and_odes.ipynb) | ✅ Complete | ~25-30 min |
| 2 | [Functional Fields](field_series/02_functional_fields.ipynb) | ✅ Complete | ~20-25 min |
| 3 | [Reinforcement Fields](field_series/03_reinforcement_fields/) | ✅ Complete | ~30 min |

---

## 🗺️ Complete Documentation

**Learn More:**
- 📖 **[Field Series Roadmap](field_series/ROADMAP.md)** — Planned future notebooks (Policy Inference, Memory Update, RF-SARSA)
- 🎯 **[Learning Paths](https://github.com/pleiadian53/GRL/blob/main/notebooks/README.md)** — How to use these notebooks
- 🔗 **[genai-lab Connection](https://github.com/pleiadian53/genai-lab)** — Flow Matching & Diffusion Models

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

- **[Tutorial Series](../GRL0/tutorials/README.md)** — Mathematical depth
- **[Implementation Guide](../GRL0/implementation/README.md)** — Technical specs
- **[Quantum-Inspired Extensions](../GRL0/quantum_inspired/README.md)** — Advanced topics

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
