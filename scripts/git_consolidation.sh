#!/bin/bash
# Git Consolidation Script for GRL
# This script consolidates the new local workspace with the existing GitHub repo

set -e  # Exit on error

echo "=== GRL Git Consolidation ==="
echo ""

# 1. Navigate to project root
cd /Users/pleiadian53/work/GRL
echo "Working directory: $(pwd)"
echo ""

# 2. Initialize git if not already
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    echo ""
else
    echo "Git already initialized"
    echo ""
fi

# 3. Create initial commit with new content
echo "Adding all new files..."
git add .
echo ""

echo "Creating initial commit..."
git commit -m "feat: Complete GRL-v0 documentation and project structure

- Add comprehensive tutorial paper structure (docs/GRL0/)
- Add Chapter 0: Overview and Chapter 1: Core Concepts
- Add paper-ready sections framework
- Add implementation specifications framework
- Add project infrastructure (environment.yml, pyproject.toml)
- Add installation guide and verification script
- Update README with tutorial paper focus"
echo ""

# 4. Add remote (existing GitHub repo)
echo "Adding remote origin..."
git remote add origin https://github.com/pleiadian53/GRL.git 2>/dev/null || echo "Remote 'origin' already exists"
echo ""

# 5. Fetch existing content
echo "Fetching existing content from GitHub..."
git fetch origin
echo ""

# 6. Create a backup branch for the old content
echo "Creating backup of old content..."
git branch old-content origin/main 2>/dev/null || echo "Branch may already exist"
echo ""

# 7. Merge with allow-unrelated-histories
echo "Merging old content with new structure..."
echo "(This will preserve your GRL-basics.ipynb, GRL2-paper.pdf, and demo/ folder)"
git merge origin/main --allow-unrelated-histories -m "Merge existing GRL content with new documentation structure

Preserved from original repo:
- GRL-basics.ipynb (Jupyter notebook)
- GRL2-paper.pdf (original paper)
- demo/ folder

New additions:
- Complete docs/GRL0/ tutorial paper structure
- Project infrastructure and environment
- Implementation specifications"
echo ""

# 8. Review what we have
echo "=== Current Status ==="
git status
echo ""

echo "=== Files from old repo preserved ==="
ls -la GRL-basics.ipynb GRL2-paper.pdf demo/ 2>/dev/null || echo "Old files will be added after merge"
echo ""

# 9. Push to GitHub
echo "Ready to push? (The script will pause here)"
echo "To push, run: git push -u origin main"
echo ""
echo "=== Consolidation Complete ==="
