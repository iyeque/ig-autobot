import pathlib

env_path = pathlib.Path('.env')
content = env_path.read_text(encoding='utf-8')

new_content = []
for line in content.splitlines():
    if line.startswith('THREADS_ACCESS_TOKEN='):
        new_content.append('THREADS_ACCESS_TOKEN=***    else:
        new_content.append(line)

env_path.write_text('\n'.join(new_content) + '\n', encoding='utf-8')
print('Updated THREADS_ACCESS_TOKEN')
