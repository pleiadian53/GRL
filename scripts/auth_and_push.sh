#!/bin/bash
# Authenticate and push to GitHub using gh CLI

set -e

echo "=== GitHub Authentication and Push ==="
echo ""

cd /Users/pleiadian53/work/GRL

# Check if gh is authenticated
echo "Checking GitHub CLI authentication..."
if gh auth status >/dev/null 2>&1; then
    echo "✓ Already authenticated with GitHub"
else
    echo "Need to authenticate with GitHub..."
    echo "Running: gh auth login"
    gh auth login
fi
echo ""

# Configure git to use gh for credentials
echo "Configuring git to use GitHub CLI for authentication..."
gh auth setup-git
echo ""

# Push to main
echo "Pushing to GitHub (main branch)..."
git push -u origin main
echo ""

echo "=== Success! ==="
echo ""
echo "Repository URL: https://github.com/pleiadian53/GRL"
echo "Documentation: https://github.com/pleiadian53/GRL/tree/main/docs/GRL0"
echo ""
