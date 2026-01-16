# Contributing to GRL

Thank you for your interest in contributing to Generalized Reinforcement Learning (GRL)!

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:

1. Check if it's already reported in [GitHub Issues](https://github.com/pleiadian53/GRL/issues)
2. If not, create a new issue with:

   - Clear description
   - Steps to reproduce (for bugs)
   - Expected vs. actual behavior

### Contributing Code

1. **Fork the repository**

2. **Create a branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**

4. **Test your changes**: Ensure all tests pass
5. **Commit**: Use clear, descriptive commit messages
6. **Push**: `git push origin feature/your-feature-name`
7. **Create a Pull Request**

### Contributing Documentation

Documentation improvements are highly valued!

- Tutorial chapters: `docs/GRL0/tutorials/`
- Implementation guides: `docs/GRL0/implementation/`
- Examples and notebooks: `notebooks/`

**For math-heavy docs**:

- Use standard LaTeX: `$...$` for inline, `$$...$$` for display
- Preview locally: `mkdocs serve`
- The documentation site will render math automatically

### Code Style

- **Python**: Follow PEP 8
- **Docstrings**: Use Google style
- **Type hints**: Preferred for new code
- **Comments**: Explain "why," not "what"

### Running Tests

```bash
pytest tests/
```

### Building Documentation Locally

```bash
pip install -r requirements-docs.txt
mkdocs serve
```

Then visit `http://localhost:8000`

## Questions?

Feel free to open an issue for questions or join discussions!

---

**Thank you for contributing to GRL!** 🎉
