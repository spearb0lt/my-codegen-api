#!/usr/bin/env python3
"""
Higher-level automate script that prepares (optionally uploads) and runs locally,
and only contacts server on error.

Usage examples:

# Use an existing solution_id on server, provide local test file:
python test2_automate_post_on_error.py --server https://my-codegen-api2.onrender.com --solution-id <ID> --test-file pp_input.txt

# Upload local code + test file in one call (server will store it if you post failures):
python test2_automate_post_on_error.py --server https://my-codegen-api2.onrender.com --solution-file a.py --test-file pp_input.txt

"""
import argparse
import requests
import time
import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

DOWNLOAD_TIMEOUT = 60
DEFAULT_MAX_ITERS = 4
DEFAULT_TIMEOUT = 60

def call_prepare(server, solution_id=None, solution_file=None, test_file=None, test_input_str=None):
    """
    Prepare step: if solution_file present, we don't yet POST to server.
    We return a dict which may include 'solution_id' and 'program_filename' if server had one.
    For this simplified flow we just return a minimal dict if we didn't call server.
    """
    # If we have solution_id and want to retrieve program metadata, attempt to call /solutions/{id}/files
    if solution_id:
        try:
            r = requests.get(server.rstrip('/') + f'/solutions/{solution_id}/files', timeout=DOWNLOAD_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception:
            # ignore; calling server not mandatory here
            return {}
    return {}

def download_program(server, solution_id, program_filename, out_path):
    url = server.rstrip('/') + f'/download/{solution_id}/{program_filename}'
    r = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
    r.raise_for_status()
    with open(out_path, 'wb') as f:
        shutil.copyfileobj(r.raw, f)
    return out_path

def run_program_locally(python_exe, program_path, test_input_path, time_limit):
    try:
        with open(test_input_path, 'r', encoding='utf-8') as inf:
            p = subprocess.run([python_exe, str(program_path)], stdin=inf, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=time_limit)
            return {"stdout": p.stdout, "stderr": p.stderr, "timed_out": False, "returncode": p.returncode}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"Time Limit Exceeded ({time_limit} seconds)", "timed_out": True, "returncode": None}
    except Exception as e:
        return {"stdout": "", "stderr": f"Local runner exception: {repr(e)}", "timed_out": False, "returncode": None}

def post_result(server, payload):
    url = server.rstrip('/') + '/test2'
    r = requests.post(url, json=payload, timeout=DOWNLOAD_TIMEOUT)
    r.raise_for_status()
    return r.json()

def normalize_out(s: str) -> str:
    return "\n".join(line.rstrip() for line in s.strip().splitlines())

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--server', required=True)
    p.add_argument('--solution-id', help='Existing solution_id on the server (optional)')
    p.add_argument('--solution-file', help='Local solution code file to upload (optional)')
    p.add_argument('--test-file', help='Local test input file (required unless --test-input is passed)')
    p.add_argument('--test-input', help='Direct test input string (alternative)')
    p.add_argument('--program-filename', default='coding_solution.py', help='Name to save downloaded program locally')
    p.add_argument('--python-exe', default=sys.executable, help='Python executable to run locally')
    p.add_argument('--max-iters', type=int, default=DEFAULT_MAX_ITERS)
    p.add_argument('--time-limit', type=int, default=DEFAULT_TIMEOUT)
    p.add_argument('--upload-local-program', action='store_true', help="Include program code in POST when failing")
    args = p.parse_args()

    if not args.test_file and args.test_input is None:
        print("Either --test-file or --test-input must be provided.", file=sys.stderr)
        sys.exit(2)

    # If solution_file provided, use it locally; else attempt to download if solution_id provided
    if args.solution_file:
        local_program_path = Path(args.solution_file)
        if not local_program_path.exists():
            print("solution_file not found:", local_program_path); sys.exit(2)
    elif args.solution_id:
        # attempt download
        try:
            print("Attempting to download program from server...")
            download_program(args.server, args.solution_id, args.program_filename, Path(args.program_filename))
            local_program_path = Path(args.program_filename)
            print("Downloaded program to", local_program_path)
        except Exception as e:
            print("Failed to download program:", e)
            print("Provide --solution-file if you have a local copy.")
            sys.exit(1)
    else:
        print("Provide either --solution-file or --solution-id to obtain a program to run.", file=sys.stderr)
        sys.exit(2)

    # ensure test input exists
    if args.test_file:
        local_test_input = Path(args.test_file)
        if not local_test_input.exists():
            print("test file not found:", local_test_input); sys.exit(2)
    else:
        local_test_input = Path('test_input_temp.txt')
        local_test_input.write_text(args.test_input or '', encoding='utf-8')

    # Run locally first; only call server if failure
    for attempt in range(1, args.max_iters + 1):
        print(f"\n=== ITER {attempt}/{args.max_iters} ===")
        run_res = run_program_locally(args.python_exe, local_program_path, local_test_input, args.time_limit)
        print("Local run: timed_out=", run_res['timed_out'], " returncode=", run_res['returncode'])
        Path('latest_run_stdout.txt').write_text(run_res['stdout'] or '', encoding='utf-8')
        Path('latest_run_stderr.txt').write_text(run_res['stderr'] or '', encoding='utf-8')

        success = (not run_res['timed_out']) and (not run_res['stderr'].strip()) and (run_res['returncode'] in (0, None))
        if success:
            print("Success locally. No server contact required. Exiting.")
            sys.exit(0)

        # Local failure -> post to server /test2
        payload = {
            'solution_id': args.solution_id,
            'stdout': run_res['stdout'] or '',
            'stderr': run_res['stderr'] or '',
            'timed_out': bool(run_res['timed_out']),
            'returncode': run_res['returncode'],
            'test_input': local_test_input.read_text(encoding='utf-8')
        }
        if args.upload_local_program:
            payload['solution'] = local_program_path.read_text(encoding='utf-8')

        print("Posting failure to server /test2 ...")
        try:
            server_resp = post_result(args.server, payload)
        except Exception as e:
            print("Failed to post to server:", e); sys.exit(1)

        status = server_resp.get('status')
        if status == 'regenerated':
            candidate = server_resp.get('candidate_solution') or server_resp.get('solution') or server_resp.get('candidate')
            if not candidate:
                candidate = server_resp.get('raw_llm_text') or ""
            if not candidate:
                print("No candidate returned by server. Response:", server_resp); sys.exit(1)
            print("Saving regenerated candidate to", local_program_path)
            local_program_path.write_text(candidate, encoding='utf-8')
            if 'solution_id' in server_resp:
                args.solution_id = server_resp['solution_id']
            continue
        elif status == 'ok':
            print("Server returned ok (candidate verified server-side). Exiting.")
            sys.exit(0)
        else:
            print("Server returned failure. Response:", server_resp)
            sys.exit(1)

    print("Reached max iterations without success.")
    sys.exit(2)

if __name__ == "__main__":
    main()
