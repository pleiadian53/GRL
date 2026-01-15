# GRL Documentation Setup

This repository uses **MkDocs with Material theme** to generate beautiful, math-enabled documentation.

## 📚 Live Documentation

**Published at**: https://pleiadian53.github.io/GRL/

Documentation auto-deploys on every push to `main` that modifies:
- `docs/**`
- `mkdocs.yml`
- `.github/workflows/docs.yml`

---

## 🚀 Local Development

### Install Dependencies

```bash
pip install -r requirements-docs.txt
```

### Serve Locally

```bash
mkdocs serve
```

Visit **http://localhost:8000** to preview with live reload.

### Build Static Site

```bash
mkdocs build
```

Output in `site/` directory.

---

## ✨ Features

### Math Rendering

**Your existing LaTeX works as-is!**

- Inline math: `$Q^+(z)$` → $Q^+(z)$
- Display math: `$$E(z) = -Q^+(z)$$` → $$E(z) = -Q^+(z)$$
- Physics notation: `$|\psi\rangle$`, `$\hat{H}$`, `$\nabla_\theta$` all work!

Powered by **MathJax 3** via `pymdownx.arithmatex`.

### Search

Full-text search including:
- Document content
- Headers
- Code blocks
- Math expressions (as text)

### Navigation

- **Tabs**: Top-level sections (GRL v0, Roadmap, About)
- **Sidebar**: Auto-generated from `nav` in `mkdocs.yml`
- **Breadcrumbs**: Show current location
- **Back to top**: Floating button

### Code Blocks

```python
# Syntax highlighting with copy button
def grl_example():
    Q_plus = compute_field(particles, kernel)
    return Q_plus
```

### Mermaid Diagrams

Your existing Mermaid diagrams render automatically!

---

## 📝 Adding New Pages

### Option 1: Add to Navigation (Recommended)

Edit `mkdocs.yml`:

```yaml
nav:
  - 'New Section':
      - 'New Page': path/to/new-page.md
```

### Option 2: Auto-discovery

Place `.md` files in `docs/` — they'll be accessible by URL.

---

## 🎨 Customization

### Theme Colors

Edit `mkdocs.yml` → `theme.palette`:

```yaml
theme:
  palette:
    primary: indigo  # Change to red, blue, green, etc.
    accent: indigo
```

### Extra CSS/JS

- CSS: `docs/stylesheets/extra.css`
- JS: `docs/javascripts/mathjax.js` (MathJax config)

---

## 🐛 Troubleshooting

### Math Not Rendering?

**Check**:
1. Use `$...$` or `$$...$$` (not `\(...\)` or `\[...\]` in source)
2. Escape special chars: `\{`, `\}`, `\\`
3. Run `mkdocs serve` locally to debug

### Build Fails?

```bash
mkdocs build --strict
```

Shows all warnings as errors (helps catch broken links).

### Links Not Working?

- **Internal links**: Use relative paths from current file
  - Good: `[Chapter 2](02-rkhs-foundations.md)`
  - Bad: `[Chapter 2](/docs/GRL0/tutorials/02-rkhs-foundations.md)`

---

## 📦 Project Structure

```
GRL/
├── docs/                      # Documentation source
│   ├── index.md              # Landing page (copy of README.md)
│   ├── GRL0/                 # Tutorial paper
│   │   ├── tutorials/        # Main tutorial chapters
│   │   ├── quantum_inspired/ # Quantum extensions
│   │   └── implementation/   # Implementation guide
│   ├── javascripts/          # MathJax config
│   └── stylesheets/          # Custom CSS
├── mkdocs.yml                # MkDocs configuration
├── requirements-docs.txt     # Python dependencies
└── .github/workflows/docs.yml # Auto-deploy workflow
```

---

## 🔧 Advanced: Manual Deployment

```bash
# Build and deploy to gh-pages branch
mkdocs gh-deploy

# Or specify remote
mkdocs gh-deploy --remote-name origin
```

GitHub Actions handles this automatically!

---

## 📚 Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [PyMdown Extensions](https://facelessuser.github.io/pymdown-extensions/)
- [MathJax Documentation](https://docs.mathjax.org/)

---

## ✅ Checklist After Setup

- [ ] Install dependencies: `pip install -r requirements-docs.txt`
- [ ] Test locally: `mkdocs serve`
- [ ] Visit http://localhost:8000
- [ ] Check math rendering (pick any tutorial chapter)
- [ ] Push to GitHub → auto-deploy triggered
- [ ] Wait 2-3 minutes for deployment
- [ ] Visit https://pleiadian53.github.io/GRL/
- [ ] Enable GitHub Pages in repo settings (if first time)

---

**Questions?** Open an issue or check the [MkDocs documentation](https://www.mkdocs.org/).
