# Interactive Notebooks

Welcome to the GRL interactive demonstrations! These Jupyter notebooks provide hands-on visualizations and examples.

---

## Available Notebooks

### **[Vector Field Visualization](vector_field.ipynb)**
Interactive exploration of:
- Reinforcement field topology
- Particle dynamics and memory evolution  
- Energy landscape navigation
- Action inference from field gradients

---

## Note for Contributors

These notebooks are **copies** from the main `notebooks/` directory in the repository, rendered here for reliable display with full math and plot support.

**Development workflow:**
1. Work on notebooks in `/notebooks/` (root directory)
2. Copy to `/docs/notebooks/` when ready to publish
3. Notebooks here are built and deployed to GitHub Pages automatically

**Why two locations?**
- `/notebooks/` — Standard location for repository browsing
- `/docs/notebooks/` — Rendered version for documentation site (reliable, math-enabled)

---

## Running Locally

To run these notebooks interactively:

```bash
git clone https://github.com/pleiadian53/GRL.git
cd GRL/notebooks  # Use the original location
jupyter notebook
```
