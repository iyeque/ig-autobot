import subprocess, json, datetime, urllib.request, os, sys

WORKFLOWS = [
    'auto_bluesky.yml',
    'auto_instagram.yml',
    'auto_linkedin.yml',
    'auto_pinterest.yml',
    'auto_threads.yml',
    'auto_wilma_bluesky.yml',
    'auto_wilma_linkedin.yml',
    'auto_youtube.yml',
    'master_content_gen.yml',
    'master_wilma_gen.yml',
]

REPO = 'iyeque/ig-autobot'
CWD = 'C:/Users/Huawei/Downloads/ig-autobot'
CUTOFF = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=24)

TRANSIENT_KW = ['timeout','timed out','5xx','502','503','504','network error','github pages deploy race','git push rejected','temporarily unable','service unavailable','request failed','connectionerror','connecterror','socket error']
PERSISTENT_FIXABLE_KW = ['jsondecodeerror','json syntax','state.json','filenotfound','no such file','missing image','nameerror','importerror','modulenotfounderror']

def gh(args):
    result = subprocess.run(['gh'] + args, capture_output=True, text=True, cwd=CWD)
    if result.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()

def download_logs(job_id):
    token = gh(['auth','token'])
    url = f"https://api.github.com/repos/{REPO}/actions/jobs/{job_id}/logs"
    req = urllib.request.Request(url, headers={'Authorization': f'Bearer {token}'})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
    except Exception as e:
        return f"[Failed to download logs: {e}]"
    try:
        return data.decode('utf-8', errors='replace')
    except Exception:
        return "[binary log data]"

report = []
for wf in WORKFLOWS:
    try:
        raw = gh(['run','list','--workflow',wf,'--limit','1','--json','status,conclusion,createdAt,databaseId'])
        runs = json.loads(raw)
    except Exception as e:
        report.append(f"{wf}: error fetching runs ({e})")
        continue
    if not runs:
        report.append(f"{wf}: no runs found")
        continue
    run = runs[0]
    created = datetime.datetime.fromisoformat(run['createdAt'].replace('Z','+00:00'))
    status = run['status']
    conclusion = run.get('conclusion') or ''
    if created < CUTOFF:
        report.append(f"{wf}: latest run older than 24h ({created.strftime('%Y-%m-%d %H:%M UTC')})")
        continue
    if status == 'completed' and conclusion == 'success':
        report.append(f"{wf}: SUCCESS")
    elif status == 'completed' and conclusion not in ('success', ''):
        run_id = str(run['databaseId'])
        try:
            jobs_raw = gh(['run','view', run_id, '--json','jobs'])
            jobs = json.loads(jobs_raw)['jobs']
            failed_job = next((j for j in jobs if j['conclusion'] == 'failure'), None)
        except Exception as e:
            report.append(f"{wf}: FAILED ({conclusion}) but could not get jobs ({e})")
            continue
        if not failed_job:
            report.append(f"{wf}: {conclusion.upper()}")
            continue
        job_id = failed_job['databaseId']
        log_text = download_logs(job_id)
        lower = log_text.lower()
        classification = 'PERSISTENT'
        for kw in TRANSIENT_KW:
            if kw in lower:
                classification = 'TRANSIENT'
                break
        action = 'none'
        if classification == 'TRANSIENT':
            try:
                gh(['workflow','run',wf,'--ref','master'])
                action = 'retry'
            except Exception as e:
                action = f'retry-failed({e})'
            report.append(f"{wf}: FAILED ({conclusion}) - {classification} - action={action}")
        else:
            fixable = any(k in lower for k in PERSISTENT_FIXABLE_KW)
            action = 'alert'
            snippet = log_text[:300].replace('\n',' ')
            report.append(f"{wf}: FAILED ({conclusion}) - {classification} (fixable={fixable}) - action={action} - {snippet}")
    else:
        report.append(f"{wf}: {status.upper()} / {conclusion or 'none'}")

print('\n'.join(report))
