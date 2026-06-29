#!/usr/bin/env python3
"""HDOCK docking client — web submission scaffold, binary install pending.

STATUS (2026-06-29):
  Web mode: form submission works but job_id extraction fails (no redirect).
  Binary mode: GitHub repo huanglabpku/HDOCK deleted, source unavailable.
  Current: hdock_client.py preserved as architecture reference.
  Fallback: modules/idp_dock_score.py (pure-Python contact+electrostatics).

Usage (when HDOCK binary is available):
    from modules.hdock_client import dock_complex
    result = dock_complex('receptor.pdb', 'ligand.pdb')
    # → {'score': -185.3, 'contacts': 45, 'complex_pdb': 'docked.pdb'}
"""
import os, time, re, requests, tempfile, shutil

HDOCK_URL = "http://hdock.phys.hust.edu.cn/"


def dock_complex(receptor_pdb: str, ligand_pdb: str,
                 timeout: int = 600, poll_interval: int = 10) -> dict:
    """Run HDOCK docking and return top result.

    Args:
        receptor_pdb: path to receptor PDB (antibody)
        ligand_pdb:   path to ligand PDB (peptide/epitope)
        timeout:      max wait time in seconds
        poll_interval: seconds between status checks

    Returns:
        {'score': float, 'ligand_rmsd': float, 'complex_pdb': path}
        or {'error': str} on failure
    """
    session = requests.Session()

    # ── Step 1: Submit job ──
    with open(receptor_pdb, 'rb') as fr, open(ligand_pdb, 'rb') as fl:
        files = {
            'pdbfile1': (os.path.basename(receptor_pdb), fr, 'application/octet-stream'),
            'pdbfile2': (os.path.basename(ligand_pdb), fl, 'application/octet-stream'),
        }
        data = {'email': '', 'jobname': 'bfn_dock'}
        try:
            resp = session.post(HDOCK_URL, files=files, timeout=30)
        except requests.exceptions.RequestException as e:
            return {'error': f'Submit failed: {e}'}

    if resp.status_code != 200:
        return {'error': f'Submit HTTP {resp.status_code}'}

    # Extract job ID from response URL or content
    html = resp.text
    job_url = resp.url
    job_id = None

    # HDOCK redirects to result.php?jobid=XXXXX or returns a page with the jobid
    for pattern in [r'jobid=([A-Za-z0-9_-]+)', r'id=([A-Za-z0-9_-]+)',
                    r'job_id=([A-Za-z0-9_-]+)', r'name="jobid"[^>]*value="([^"]+)"']:
        m = re.search(pattern, job_url)
        if not m:
            m = re.search(pattern, html)
        if m:
            job_id = m.group(1)
            break

    if not job_id:
        # Try to match the download link pattern
        m = re.search(r'href="([^"]*result[^"]*\.tar\.gz[^"]*)"', html)
        if m:
            return {'error': 'No job_id found, direct result link: ' + m.group(1)[:80]}
        # Check for error message
        if 'error' in html.lower() or 'fail' in html.lower():
            err_match = re.search(r'(error|fail)[^<]*', html, re.IGNORECASE)
            return {'error': f'HDOCK error: {err_match.group(0)[:200] if err_match else html[:200]}'}
        return {'error': f'Could not extract job_id. URL: {job_url[:80]}, content_len: {len(html)}'}

    print(f'  HDOCK job submitted: {job_id}')

    # ── Step 2: Poll for completion ──
    # HDOCK uses result.php for job status
    status_url = f'{HDOCK_URL}result.php?jobid={job_id}'
    t0 = time.time()
    result_url = None

    while time.time() - t0 < timeout:
        try:
            resp = session.get(status_url, timeout=15)
        except requests.exceptions.RequestException:
            time.sleep(poll_interval)
            continue

        html = resp.text

        # Look for result download link
        m = re.search(r'href="([^"]*result[^"]*\.tar\.gz[^"]*)"', html)
        if m:
            result_url = m.group(1)
            if not result_url.startswith('http'):
                result_url = HDOCK_URL.rstrip('/') + '/' + result_url.lstrip('/')
            break

        # Check for error
        if 'error' in html.lower() or 'failed' in html.lower():
            return {'error': f'HDOCK job failed: {html[:200]}'}

        time.sleep(poll_interval)

    if not result_url:
        return {'error': f'Timeout after {timeout}s'}

    print(f'  HDOCK job complete in {time.time()-t0:.0f}s')

    # ── Step 3: Download and extract ──
    tmpdir = tempfile.mkdtemp(prefix='hdock_')
    tar_path = os.path.join(tmpdir, 'result.tar.gz')
    try:
        resp = session.get(result_url, timeout=60)
        with open(tar_path, 'wb') as f:
            f.write(resp.content)
    except requests.exceptions.RequestException as e:
        shutil.rmtree(tmpdir)
        return {'error': f'Download failed: {e}'}

    import tarfile
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(tmpdir)
    except tarfile.TarError as e:
        shutil.rmtree(tmpdir)
        return {'error': f'Extract failed: {e}'}

    # ── Step 4: Parse results ──
    # HDOCK outputs: model_1.pdb (top), model_2.pdb, ..., hdock.out (scores)
    score_file = os.path.join(tmpdir, 'hdock.out')
    top_pdb = os.path.join(tmpdir, 'model_1.pdb')

    if not os.path.exists(score_file):
        # Try alternate names
        for f in os.listdir(tmpdir):
            if f.endswith('.out') or 'score' in f.lower():
                score_file = os.path.join(tmpdir, f)
                break

    score = None
    if os.path.exists(score_file):
        with open(score_file) as f:
            for line in f:
                # Parse: "model_1.pdb  -185.23  0.00"  (score, ligand_rmsd)
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0].startswith('model'):
                    try:
                        score = float(parts[1])
                        break
                    except ValueError:
                        continue

    # ── Step 5: Copy top complex back ──
    result_pdb = None
    if os.path.exists(top_pdb):
        result_pdb = tempfile.mktemp(suffix='_docked.pdb')
        shutil.copy(top_pdb, result_pdb)

    shutil.rmtree(tmpdir)

    return {
        'score': score,
        'complex_pdb': result_pdb,
        'job_id': job_id,
    }


def score_design(receptor_pdb: str, ligand_pdb: str,
                 timeout: int = 600) -> dict:
    """Score a single design via HDOCK docking.

    Returns:
        {'hdock_score': float, 'success': bool, 'docked_pdb': path}
    """
    result = dock_complex(receptor_pdb, ligand_pdb, timeout=timeout)
    if 'error' in result:
        return {'hdock_score': 0, 'success': False, 'error': result['error']}
    return {
        'hdock_score': result['score'] or 0,
        'success': True,
        'docked_pdb': result.get('complex_pdb'),
    }


# ── Quick test ──
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('receptor')
    ap.add_argument('ligand')
    args = ap.parse_args()
    result = dock_complex(args.receptor, args.ligand)
    for k, v in result.items():
        if k != 'complex_pdb':
            print(f'{k}: {v}')
