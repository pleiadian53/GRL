#!/usr/bin/env python3
"""
Check for markdown formatting issues in GRL documentation.

Issues checked:
1. Missing blank lines before lists (after "where:" or text ending with ":")
2. Missing blank lines before/after display math blocks ($$)
3. Lists immediately after bold text ending with ":"
"""

import re
from pathlib import Path

def check_file(filepath):
    """Check a single markdown file for formatting issues."""
    issues = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines, start=1):
        # Check for text ending with ":" followed by list without blank line
        if line.strip().endswith(':') and i < len(lines):
            next_line = lines[i].strip() if i < len(lines) else ''
            # Check if next line is a list item (starts with -, *, or digit.)
            if next_line and (next_line.startswith('- ') or 
                             next_line.startswith('* ') or 
                             re.match(r'^\d+\.', next_line)):
                issues.append({
                    'line': i,
                    'type': 'missing_blank_before_list',
                    'context': line.strip()[:60]
                })
        
        # Check for $$ followed by non-blank line (that's not another $$)
        if line.strip() == '$$' and i < len(lines):
            prev_line = lines[i-2].strip() if i > 1 else ''
            next_line = lines[i].strip() if i < len(lines) else ''
            
            # Check if this is an opening $$
            # Look back to see if there's a previous $$ nearby
            is_closing = False
            for j in range(max(0, i-10), i-1):
                if lines[j].strip() == '$$':
                    is_closing = True
                    break
            
            if is_closing:
                # This is a closing $$, check if followed by non-blank
                if next_line and next_line != '$$' and not next_line.startswith('#'):
                    issues.append({
                        'line': i,
                        'type': 'missing_blank_after_math',
                        'context': f"$$ followed by: {next_line[:40]}"
                    })
            else:
                # This is an opening $$, check if preceded by non-blank
                if prev_line and not prev_line.endswith(':') and not prev_line.startswith('#'):
                    issues.append({
                        'line': i,
                        'type': 'missing_blank_before_math',
                        'context': f"$$ preceded by: {prev_line[:40]}"
                    })
        
        # Check for "where:" followed directly by list
        if line.strip() == 'where:' and i < len(lines):
            next_line = lines[i].strip() if i < len(lines) else ''
            if next_line.startswith('- ') or next_line.startswith('* '):
                issues.append({
                    'line': i,
                    'type': 'missing_blank_after_where',
                    'context': 'where: followed by list'
                })
        
        # Check for bold text with ":" followed by list
        if re.search(r'\*\*[^*]+\*\*:\s*$', line.strip()) and i < len(lines):
            next_line = lines[i].strip() if i < len(lines) else ''
            if next_line.startswith('- ') or next_line.startswith('* '):
                issues.append({
                    'line': i,
                    'type': 'missing_blank_after_bold_colon',
                    'context': line.strip()[:60]
                })
    
    return issues

def main():
    """Check all markdown files in docs/GRL0/."""
    docs_dir = Path('/Users/pleiadian53/work/GRL/docs/GRL0')
    
    all_issues = {}
    total_issues = 0
    
    for md_file in sorted(docs_dir.rglob('*.md')):
        issues = check_file(md_file)
        if issues:
            rel_path = md_file.relative_to(Path('/Users/pleiadian53/work/GRL'))
            all_issues[str(rel_path)] = issues
            total_issues += len(issues)
    
    if not all_issues:
        print("✅ No formatting issues found!")
        return
    
    print(f"Found {total_issues} potential formatting issues in {len(all_issues)} files:\n")
    
    for filepath, issues in all_issues.items():
        print(f"\n📄 {filepath}")
        for issue in issues:
            print(f"   Line {issue['line']:4d}: {issue['type']}")
            print(f"            {issue['context']}")
    
    print(f"\n\nTotal: {total_issues} issues in {len(all_issues)} files")

if __name__ == '__main__':
    main()
