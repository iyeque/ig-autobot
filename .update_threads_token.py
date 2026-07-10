import pathlib

p = pathlib.Path('.env')
text = p.read_text(encoding='utf-8')

new_token = os.environ.get('NEW_THREADS_TOKEN', '')
if not new_token:
    print('Error: NEW_THREADS_TOKEN env var not set')
    exit(1)

for i, line in enumerate(text.splitlines()):
    if line.startswith('THREADS_ACCESS_TOKEN='):
        text = text.replace(line, f'THREADS_ACCESS_TOKEN="{new_token}"')
        p.write_text(text, encoding='utf-8')
        print(f'Updated line {i+1}: THREADS_ACCESS_TOKEN')
        break
else:
    print('THREADS_ACCESS_TOKEN not found')
