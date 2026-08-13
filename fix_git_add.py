import pathlib, re

fixes = {
    '.github/workflows/auto_bluesky.yml': '''          git add state.json || true
          git add images/ || true
          git add *.flag || true''',
    '.github/workflows/auto_instagram.yml': '''          git add state.json || true''',
    '.github/workflows/auto_linkedin.yml': '''          git add state.json || true
          git add images/ || true
          git add *.flag || true''',
    '.github/workflows/auto_pinterest.yml': '''          git add state.json || true
          git add images/ || true
          git add *.flag || true''',
    '.github/workflows/auto_threads.yml': '''          git add state.json || true
          git add images/ || true
          git add *.flag || true''',
    '.github/workflows/auto_youtube.yml': '''          git add state.json || true
          git add images/ || true
          git add reels/ || true
          git add *.flag || true''',
}

for wf, replacement in fixes.items():
    text = pathlib.Path(wf).read_text()
    new = re.sub(
        r"( - name: Commit state changes\n\s*run: \|\n)((?:.*\n)*?)(\s*git commit)",
        lambda m: m.group(1) + replacement + "\n" + m.group(3),
        text
    )
    pathlib.Path(wf).write_text(new)
    print(f"Patched {wf}")
