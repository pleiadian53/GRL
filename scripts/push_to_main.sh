#!/bin/bash
# Push to main branch on GitHub

set -e

echo "=== Pushing GRL to GitHub (main branch) ==="
echo ""

cd /Users/pleiadian53/work/GRL

# 1. Rename branch to main
echo "Renaming branch master → main..."
git branch -M main
echo ""

# 2. Push to main
echo "Pushing to GitHub..."
git push -u origin main
echo ""

# 3. Show status
echo "=== Push Complete! ==="
echo ""
echo "Your repository is now live at:"
echo "https://github.com/pleiadian53/GRL"
echo ""
echo "View your new documentation:"
echo "https://github.com/pleiadian53/GRL/tree/main/docs/GRL0"
echo ""
