# api_server.py
# FastAPI server for generation/run pipeline.
# Expects GOOGLE_API_KEY as env var for LLM calls.
# Usage: uvicorn api_server:app --host 0.0.0.0 --port $PORT

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

app = FastAPI(title="Code-Gen Solve API")

MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-pro")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MAX_GENERATION_ATTEMPTS = int(os.getenv("MAX_GENERATION_ATTEMPTS", "3"))
EXECUTION_TIMEOUT = int(os.getenv("EXECUTION_TIMEOUT", "6"))  # seconds


def extract_python_from_markdown(text: str) -> Optional[str]:
    if not text:
        return None
    # Prefer explicit ```python blocks
    if "```python" in text:
        block = text.split("```python", 1)[1]
        block = block.split("```", 1)[0]
        return block.strip()
    # Fallback: any triple-backtick block
    if "```" in text:
        block = text.split("```", 1)[1].split("```", 1)[0]
        # if first line says "python", drop it
        lines = block.splitlines()
        if lines and lines[0].strip().lower().startswith("python"):
            block = "\\n".join(lines[1:])
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
            # extract file flattening directories (take only basename)
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
    This is simple and not fully sandboxed. Use with caution.
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
):
    tmp_root = Path(tempfile.mkdtemp(prefix="solve_api_"))
    try:
        # read uploaded zip and extract
        content = await file.read()
        unpack_zip_to_dir(content, tmp_root)
        found = find_problem_files(tmp_root)

        if mode == "generate":
            # require statement + sample_in + sample_out
            if "statement" not in found or "sample_in" not in found or "sample_out" not in found:
                raise HTTPException(status_code=400, detail="Zip must contain statement, sample_in, sample_out files.")

            statement_text = found["statement"].read_text(encoding="utf-8")
            sample_in_text = found["sample_in"].read_text(encoding="utf-8")
            sample_out_text = found["sample_out"].read_text(encoding="utf-8")
            run_input = test_input if test_input is not None else sample_in_text

            # LLM client
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
                if attempt == 1:
                    prompt = f"""
You are an expert competitive programmer. Write a Python 3 solution that reads from standard input and writes to standard output.

Problem statement:
{statement_text}

Sample Input:
{sample_in_text}

Sample Output:
{sample_out_text}

Provide only the final Python code in a single markdown code block (```python ... ```).
""" 
                else:
                    prompt = f"""
Previous submission produced incorrect output.

Problem statement:
{statement_text}

Previous code:
```python
{last_code}
```

Reason for failure:
{last_error}

Please provide a corrected complete Python solution in one markdown block.
 """

                # call LLM
                try:
                    resp = model.generate_content(prompt)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

                code_candidate = extract_python_from_markdown(resp.text or "") or (resp.text or "").strip()
                last_code = code_candidate

                # run candidate against sample/run_input
                run_res = run_python_code_str(code_candidate, run_input, timeout=EXECUTION_TIMEOUT)
                out_norm = "\\n".join(line.rstrip() for line in run_res["stdout"].strip().splitlines())
                expected_norm = "\\n".join(line.rstrip() for line in sample_out_text.strip().splitlines())

                if run_res["timed_out"]:
                    last_error = run_res["stderr"]
                elif run_res["stderr"]:
                    # runtime errors or warnings
                    last_error = run_res["stderr"]
                elif out_norm == expected_norm:
                    # success
                    return JSONResponse(
                        {
                            "status": "success",
                            "attempt": attempt,
                            "stdout": run_res["stdout"],
                            "stderr": run_res["stderr"],
                            "solution": last_code,
                        }
                    )
                else:
                    # mismatch -> produce diff and continue loop
                    diff = "".join(
                        difflib.unified_diff(
                            expected_norm.splitlines(keepends=True),
                            out_norm.splitlines(keepends=True),
                            fromfile="expected",
                            tofile="actual",
                        )
                    )
                    last_error = f"Wrong output. Diff:\n{diff}\nStdout:\n{run_res['stdout']}\nStderr:\n{run_res['stderr']}"

            # after attempts exhausted
            return JSONResponse(
                {
                    "status": "failed",
                    "attempts": MAX_GENERATION_ATTEMPTS,
                    "last_error": last_error,
                    "last_solution": last_code,
                },
                status_code=400,
            )

        elif mode == "run-only":
            sol = tmp_root / "solution.py"
            if not sol.exists():
                raise HTTPException(status_code=400, detail="Zip must contain solution.py in run-only mode.")
            run_input = test_input if test_input is not None else ""
            code = sol.read_text(encoding="utf-8")
            run_res = run_python_code_str(code, run_input, timeout=EXECUTION_TIMEOUT)
            return JSONResponse(
                {
                    "status": "ran",
                    "stdout": run_res["stdout"],
                    "stderr": run_res["stderr"],
                    "timed_out": run_res["timed_out"],
                    "solution": code,
                }
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported mode. Use 'generate' or 'run-only'.")
    finally:
        # always cleanup temp dir
        try:
            shutil.rmtree(tmp_root)
        except Exception:
            pass
