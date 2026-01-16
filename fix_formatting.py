#!/usr/bin/env python3
"""
Automatically fix markdown formatting issues in GRL documentation.
"""

import re
from pathlib import Path

def fix_markdown_formatting(content):
    """Fix common markdown formatting issues."""
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        current_line = lines[i]
        next_line = lines[i+1] if i < len(lines) - 1 else ''
        prev_line = lines[i-1] if i > 0 else ''
        
        # Fix: text ending with ":" followed by list without blank line
        if (current_line.strip().endswith(':') and 
            current_line.strip() != '' and
            next_line.strip() != '' and
            (next_line.strip().startswith('- ') or 
             next_line.strip().startswith('* ') or 
             re.match(r'^\d+\.', next_line.strip())) and
            not current_line.startswith('http')):  # Avoid URLs
            fixed_lines.append(current_line)
            fixed_lines.append('')  # Add blank line
            i += 1
            continue
        
        # Fix: "$$" closing tag followed by non-blank line
        if (current_line.strip() == '$$' and 
            next_line.strip() != '' and
            next_line.strip() != '$$' and
            not next_line.strip().startswith('#')):
            # Check if this is a closing $$
            is_closing = False
            for j in range(max(0, i-30), i):
                if lines[j].strip() == '$$':
                    is_closing = True
                    break
            
            if is_closing:
                fixed_lines.append(current_line)
                fixed_lines.append('')  # Add blank line after closing $$
                i += 1
                continue
        
        # Fix: non-blank line followed by opening "$$"
        if (current_line.strip() == '$$' and
            prev_line.strip() != '' and
            not prev_line.strip().endswith(':') and
            not prev_line.strip().startswith('#')):
            # Check if this is an opening $$
            is_opening = True
            for j in range(max(0, i-30), i):
                if lines[j].strip() == '$$':
                    is_opening = False
                    break
            
            if is_opening:
                # Need to add blank line before this $$
                # Go back and add it
                if fixed_lines and fixed_lines[-1].strip() != '':
                    fixed_lines.append('')
        
        fixed_lines.append(current_line)
        i += 1
    
    return '\n'.join(fixed_lines)

def process_file(filepath):
    """Process a single markdown file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        fixed_content = fix_markdown_formatting(content)
        
        # Only write if changed
        if fixed_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    """Process all markdown files in docs/GRL0/."""
    docs_dir = Path('/Users/pleiadian53/work/GRL/docs/GRL0')
    
    fixed_count = 0
    for md_file in sorted(docs_dir.rglob('*.md')):
        if process_file(md_file):
            rel_path = md_file.relative_to(Path('/Users/pleiadian53/work/GRL'))
            print(f"✓ Fixed: {rel_path}")
            fixed_count += 1
    
    print(f"\nDone! Fixed {fixed_count} files.")

if __name__ == '__main__':
    main()
