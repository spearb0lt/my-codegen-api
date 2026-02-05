# api_server_with_test_input.py
# FastAPI server for generation/run pipeline with two-stage validation:
# 1) Ensure generated code matches sample_in -> sample_out
# 2) Optionally, run the same code on a provided test_input and ensure it runs successfully
#
# Expects GOOGLE_API_KEY as env var for LLM calls.
# Usage: uvicorn api_server_with_test_input:app --host 0.0.0.0 --port $PORT

import os
import shutil
import zipfile
import tempfile
import subprocess
import difflib
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="Code-Gen Solve API (two-stage)")

MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-pro")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MAX_GENERATION_ATTEMPTS = int(os.getenv("MAX_GENERATION_ATTEMPTS", "3"))
EXECUTION_TIMEOUT = int(os.getenv("EXECUTION_TIMEOUT", "6"))  # seconds


def extract_python_from_markdown(text: str) -> Optional[str]:
    if not text:
        return None
    if "```python" in text:
        block = text.split("```python", 1)[1]
        block = block.split("```", 1)[0]
        return block.strip()
    if "```" in text:
        block = text.split("```", 1)[1].split("```", 1)[0]
        lines = block.splitlines()
        if lines and lines[0].strip().lower().startswith("python"):
            block = "\n".join(lines[1:])
        return block.strip()
    return None


def unpack_zip_to_dir(zip_bytes: bytes, dest_dir: Path) -> None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmpf:
        tmpf.write(zip_bytes)
        tmpf.flush()
        tmpf_path = Path(tmpf.name)

    with zipfile.ZipFile(tmpf_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            target_name = Path(member.filename).name
            target_path = dest_dir / target_name
            with zf.open(member) as src, open(target_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
    try:
        tmpf_path.unlink()
    except Exception:
        pass


def find_problem_files(workdir: Path):
    file_keys = {
        "statement": ["statement", "problem"],
        "sample_in": ["sample_in", "sample.in", "input", "sample-input"],
        "sample_out": ["sample_out", "sample.out", "output", "sample-output"],
    }
    found = {}
    for p in workdir.iterdir():
        name = p.name.lower()
        for key, patterns in file_keys.items():
            if any(pat in name for pat in patterns):
                found[key] = p
                break
    return found


def run_python_code_str(code_str: str, input_str: str, timeout=EXECUTION_TIMEOUT):
    """
    Run Python code provided as a string using `python -c`.
    Not fully sandboxed. Use with caution or replace with containerized execution.
    """
    try:
        p = subprocess.Popen(
            [os.getenv("PYTHON_PATH", "python"), "-c", code_str],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        out, err = p.communicate(input=input_str, timeout=timeout)
        return {"stdout": out, "stderr": err, "timed_out": False, "returncode": p.returncode}
    except subprocess.TimeoutExpired:
        try:
            p.kill()
        except Exception:
            pass
        return {"stdout": "", "stderr": f"Timeout after {timeout}s", "timed_out": True, "returncode": None}
    except Exception as e:
        return {"stdout": "", "stderr": f"Runtime error: {e}", "timed_out": False, "returncode": None}


@app.get("/health")
def health():
    return {"status": "ok", "model_configured": bool(GOOGLE_API_KEY)}


@app.post("/solve")
async def solve(
    file: UploadFile = File(...),
    mode: str = Form("generate"),
    test_input: Optional[str] = Form(None),
    test_expected: Optional[str] = Form(None),
):
    """
    Two-stage behavior:
      - mode='generate': run LLM generation loop until candidate matches sample_in -> sample_out.
                        After sample success, if test_input provided, run code on test_input as well.
                        If test_expected provided, compare test output to it; otherwise, require no runtime errors and non-empty stdout.
      - mode='run-only': expects solution.py inside zip; runs it on test_input if provided (or empty stdin otherwise).
    """
    tmp_root = Path(tempfile.mkdtemp(prefix="solve_api_"))
    try:
        content = await file.read()
        unpack_zip_to_dir(content, tmp_root)
        found = find_problem_files(tmp_root)

        if mode == "generate":
            # require sample files
            if "statement" not in found or "sample_in" not in found or "sample_out" not in found:
                raise HTTPException(status_code=400, detail="Zip must contain statement, sample_in, sample_out files.")

            statement_text = found["statement"].read_text(encoding="utf-8")
            sample_in_text = found["sample_in"].read_text(encoding="utf-8")
            sample_out_text = found["sample_out"].read_text(encoding="utf-8")

            # LLM client library
            try:
                import google.generativeai as genai
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Missing LLM client library: {e}")

            if not GOOGLE_API_KEY:
                raise HTTPException(status_code=500, detail="Server missing GOOGLE_API_KEY env var.")

            genai.configure(api_key=GOOGLE_API_KEY)
            model = genai.GenerativeModel(MODEL_NAME)

            last_code = None
            last_error = ""

            for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
                # Build prompt (first attempt vs feedback attempts)
                if attempt == 1:
                    prompt = f"""You are an expert competitive programmer. Write a Python 3 solution that reads from standard input and writes to standard output.

Problem statement:
{statement_text}

Sample Input:
{sample_in_text}

Sample Output:
{sample_out_text}

Provide only the final Python code in a single markdown code block (```python ... ```).
"""
                else:
                    # Provide feedback including last code, sample mismatch or runtime info.
                    prompt = f"""Previous submission produced incorrect output or runtime errors.

Problem statement:
{statement_text}

Previous code:
```python
{last_code}
```

Reason for failure (most recent):
{last_error}

Sample Input:
{sample_in_text}

Expected Sample Output:
{sample_out_text}

If available, also ensure the solution runs on the provided test input (without crashing). If a test input was provided, we want the program to produce some valid output for it (and match the expected if one was supplied).

Please provide a corrected complete Python solution in one markdown block.
"""

                # call LLM
                try:
                    resp = model.generate_content(prompt)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

                code_candidate = extract_python_from_markdown(resp.text or "") or (resp.text or "").strip()
                last_code = code_candidate

                # Stage A: run candidate against sample_in and compare to sample_out
                sample_run = run_python_code_str(code_candidate, sample_in_text, timeout=EXECUTION_TIMEOUT)
                sample_out_norm = "\n".join(line.rstrip() for line in sample_run["stdout"].strip().splitlines())
                expected_norm = "\n".join(line.rstrip() for line in sample_out_text.strip().splitlines())

                if sample_run["timed_out"]:
                    last_error = f"Sample run timed out: {sample_run['stderr']}"
                    # continue to next attempt
                    continue
                if sample_run["stderr"]:
                    last_error = f"Sample runtime error: {sample_run['stderr']}"
                    continue
                if sample_out_norm != expected_norm:
                    diff = "".join(difflib.unified_diff(expected_norm.splitlines(keepends=True),
                                                       sample_out_norm.splitlines(keepends=True),
                                                       fromfile="expected", tofile="actual"))
                    last_error = f"Sample mismatch. Diff:\n{diff}\nStdout:\n{sample_run['stdout']}\nStderr:\n{sample_run['stderr']}"
                    continue

                # At this point, the candidate passes the sample test.
                # Stage B: if test_input is provided, run the candidate on it and validate.
                test_result = None
                if test_input is not None:
                    test_run = run_python_code_str(code_candidate, test_input, timeout=EXECUTION_TIMEOUT)
                    # If expected output for test was provided, compare; otherwise require no stderr and non-empty stdout.
                    if test_run["timed_out"]:
                        last_error = f"Test input timed out: {test_run['stderr']}"
                        continue
                    if test_run["stderr"]:
                        last_error = f"Test runtime error: {test_run['stderr']}"
                        continue
                    test_out_norm = "\n".join(line.rstrip() for line in test_run["stdout"].strip().splitlines())
                    if test_expected is not None:
                        # compare to expected
                        expected_test_norm = "\n".join(line.rstrip() for line in test_expected.strip().splitlines())
                        if test_out_norm != expected_test_norm:
                            diff = "".join(difflib.unified_diff(expected_test_norm.splitlines(keepends=True),
                                                               test_out_norm.splitlines(keepends=True),
                                                               fromfile="expected_test", tofile="actual_test"))
                            last_error = f"Test mismatch. Diff:\n{diff}\nStdout:\n{test_run['stdout']}\nStderr:\n{test_run['stderr']}"
                            continue
                    else:
                        # require some non-empty stdout (you can relax/modify this rule)
                        if test_out_norm == "":
                            last_error = "Test run produced empty stdout."
                            continue
                    test_result = {"stdout": test_run["stdout"], "stderr": test_run["stderr"]}

                # If we reach here, candidate passed sample tests and (if provided) test input checks.
                resp_payload = {
                    "status": "success",
                    "attempt": attempt,
                    "sample_stdout": sample_run["stdout"],
                    "sample_stderr": sample_run["stderr"],
                    "solution": last_code,
                }
                if test_result is not None:
                    resp_payload["test_stdout"] = test_result["stdout"]
                    resp_payload["test_stderr"] = test_result["stderr"]

                return JSONResponse(resp_payload)

            # exhausted attempts
            return JSONResponse({"status": "failed", "attempts": MAX_GENERATION_ATTEMPTS, "last_error": last_error, "last_solution": last_code}, status_code=400)

        elif mode == "run-only":
            # Expect a solution.py in the zip
            sol = tmp_root / "solution.py"
            if not sol.exists():
                raise HTTPException(status_code=400, detail="Zip must contain solution.py in run-only mode.")
            run_input = test_input if test_input is not None else ""
            code = sol.read_text(encoding="utf-8")
            run_res = run_python_code_str(code, run_input, timeout=EXECUTION_TIMEOUT)
            return JSONResponse({"status": "ran", "stdout": run_res["stdout"], "stderr": run_res["stderr"], "timed_out": run_res["timed_out"], "solution": code})

        else:
            raise HTTPException(status_code=400, detail="Unsupported mode. Use 'generate' or 'run-only'.")
    finally:
        try:
            shutil.rmtree(tmp_root)
        except Exception:
            pass
