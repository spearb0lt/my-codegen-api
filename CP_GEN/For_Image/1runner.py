#!/usr/bin/env python3
"""
test2_automate.py

Higher-level automation: (1) optionally upload local program to server to create/reuse solution_id,
(2) run locally, (3) post results to /test2, (4) download regenerated candidate if server provides one and re-run automatically.

Usage:
  python test2_automate.py --server https://my-codegen-api2.onrender.com --solution-file a.py --test-file pp_input.txt
  python test2_automate.py --server https://my-codegen-api2.onrender.com --solution-id <id> --test-file pp_input.txt
"""
import argparse, requests, shutil, sys, json, time
from pathlib import Path

DOWNLOAD_TIMEOUT = 60

def call_test2_prepare(server, solution_id=None, solution_file=None, test_file=None, test_input_str=None):
    # This function will upload solution_file (if provided) and test_input to server via /test2 as a prepare step
    url = server.rstrip('/') + '/test2'
    files = {}
    data = {}
    # The /test2 endpoint expects JSON, so if we are uploading files, we read them and include as fields
    payload = {}
    if solution_id:
        payload['solution_id'] = solution_id
    if solution_file:
        payload['solution'] = Path(solution_file).read_text(encoding='utf-8')
    if test_file:
        payload['test_input'] = Path(test_file).read_text(encoding='utf-8')
    elif test_input_str is not None:
        payload['test_input'] = test_input_str

    r = requests.post(url, json=payload, timeout=DOWNLOAD_TIMEOUT)
    r.raise_for_status()
    return r.json()

def download_program(server, solution_id, program_filename, out_path):
    url = server.rstrip('/') + f'/download/{solution_id}/{program_filename}'
    r = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
    r.raise_for_status()
    with open(out_path, 'wb') as f:
        shutil.copyfileobj(r.raw, f)
    return out_path

def run_program_locally(python_exe, program_path, test_input_path, time_limit):
    import subprocess, sys
    timed_out = False
    stdout = ''
    stderr = ''
    returncode = None
    try:
        with open(test_input_path, 'r', encoding='utf-8') as inf:
            proc = subprocess.run([python_exe, str(program_path)], stdin=inf, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=time_limit)
            stdout = proc.stdout
            stderr = proc.stderr
            returncode = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        stderr = f"Time Limit Exceeded ({time_limit} seconds)"
    except Exception as e:
        stderr = f"Local runner exception: {repr(e)}"
    return {'stdout': stdout, 'stderr': stderr, 'timed_out': timed_out, 'returncode': returncode}

def post_to_test2(server, payload):
    url = server.rstrip('/') + '/test2'
    r = requests.post(url, json=payload, timeout=DOWNLOAD_TIMEOUT)
    r.raise_for_status()
    return r.json()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--server', required=True)
    p.add_argument('--solution-id', help='Existing solution_id on the server (optional)')
    p.add_argument('--solution-file', help='Local solution code file to upload (optional)')
    p.add_argument('--test-file', help='Local test input file (required unless --test-input is passed)')
    p.add_argument('--test-input', help='Direct test input string (alternative to --test-file)')
    p.add_argument('--program-filename', default=None, help='Name to save downloaded program locally (default: coding_solution.py)')
    p.add_argument('--python-exe', default=sys.executable, help='Python executable to run locally')
    p.add_argument('--max-iters', type=int, default=4)
    p.add_argument('--time-limit', type=int, default=60, help='Timeout when running candidate locally (seconds)')
    args = p.parse_args()

    if not args.test_file and args.test_input is None:
        print("Either --test-file or --test-input must be provided.", file=sys.stderr)
        sys.exit(2)

    # 1) Prepare: call /test2 to optionally upload solution and get solution_id/program_filename
    print("Calling server /test2 to prepare (upload if provided)...") 
    resp = call_test2_prepare(args.server, solution_id=args.solution_id, solution_file=args.solution_file, test_file=args.test_file, test_input_str=args.test_input)
    solution_id = resp.get('solution_id') or args.solution_id
    program_filename = resp.get('program_filename') or args.program_filename or 'coding_solution.py'
    test_input_filename = resp.get('test_input_filename') or 'test_input.txt'
    print(f"Server returned solution_id={solution_id}, program_filename={program_filename}")

    # Ensure we have a local copy of program
    local_program_path = Path(program_filename)
    if args.solution_file:
        if Path(args.solution_file).resolve() != local_program_path.resolve():
            shutil.copyfile(args.solution_file, local_program_path)
    else:
        # try to download program if server saved one
        try:
            print(f"Attempting to download program from server: /download/{solution_id}/{program_filename} ...")
            download_program(args.server, solution_id, program_filename, local_program_path)
        except Exception as e:
            print("Warning: failed to download program from server:", e)
            if not local_program_path.exists():
                print("No local program file found. Exiting.", file=sys.stderr)
                sys.exit(1)

    # ensure test input file exists locally
    local_test_input = Path(test_input_filename)
    if args.test_file:
        if Path(args.test_file).resolve() != local_test_input.resolve():
            shutil.copyfile(args.test_file, local_test_input)
    else:
        local_test_input.write_text(args.test_input or '', encoding='utf-8')

    python_exe = args.python_exe
    max_iters = args.max_iters
    time_limit = args.time_limit

    for attempt in range(1, max_iters + 1):
        print(f"\n=== ITERATION {attempt}/{max_iters} ===")
        run_res = run_program_locally(python_exe, local_program_path, local_test_input, time_limit)
        stdout = run_res['stdout']; stderr = run_res['stderr']; timed_out = run_res['timed_out']; returncode = run_res['returncode']

        Path('latest_run_stdout.txt').write_text(stdout or '', encoding='utf-8')
        Path('latest_run_stderr.txt').write_text(stderr or '', encoding='utf-8')

        payload = {
            'solution_id': solution_id,
            'stdout': stdout or '',
            'stderr': stderr or '',
            'timed_out': bool(timed_out),
            'returncode': returncode,
            'test_input': local_test_input.read_text(encoding='utf-8')
        }
        # Let server save program if it didn't have it
        if args.solution_file:
            payload['solution'] = Path(args.solution_file).read_text(encoding='utf-8')

        print("Posting results to server /test2 ...")
        server_resp = post_to_test2(args.server, payload)
        print("Server response:", json.dumps(server_resp, indent=2)[:2000])

        status = server_resp.get('status')
        if status == 'ok':
            print("Server considers the run successful. Exiting with success.")
            sys.exit(0)
        elif status == 'regenerated':
            candidate = server_resp.get('candidate_solution') or server_resp.get('solution') or server_resp.get('raw_llm_text')
            if not candidate:
                print("Server indicated regeneration but no candidate code was returned. Exiting.")
                sys.exit(1)
            print("Saving regenerated candidate to", local_program_path)
            local_program_path.write_text(candidate, encoding='utf-8')
            # loop will rerun
            continue
        else:
            print("Server returned failure or no candidate. Response:", server_resp)
            sys.exit(1)

    print("Reached max iterations without success.")
    sys.exit(2)

if __name__ == '__main__':
    main()
