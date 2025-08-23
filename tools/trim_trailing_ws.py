from pathlib import Path
p_root = Path('src')
for p in p_root.rglob('*.py'):
    txt = p.read_text(encoding='utf-8')
    lines = [l.rstrip() for l in txt.splitlines()]
    # ensure newline at EOF
    new_txt = '\n'.join(lines) + ('\n' if len(lines) == 0 or lines[-1] != '' else '')
    p.write_text(new_txt, encoding='utf-8')
print('trim complete')
