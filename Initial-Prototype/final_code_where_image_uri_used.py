# api_server_images_ready.py
"""
Code-Gen API server with robust image-URI and PHOTO_ID handling.
Save as api_server_images_ready.py and run with:
    uvicorn api_server_images_ready:app --host 0.0.0.0 --port 8000

Features:
- /health
- /generate : upload ZIP with statement, sample_in, sample_out, plus any number of image *_uri.txt or image files.
- /test : test an existing solution_id OR upload solution_file + problem_zip + test_file OR solution text + test_file.
- If test fails, server attempts LLM regeneration (requires GOOGLE_API_KEY and google.generativeai installed).
- PHOTO_ID placeholders in statements like {{PHOTO_ID:12345|WIDTH:700}} are parsed and mapped to detected image files/URIs.
- Saved artifacts for each solution_id include copied image files and metadata.
Security/Caveats:
- The server executes arbitrary Python code via subprocess — DO NOT run in an untrusted public environment without sandboxing and auth.
- Be mindful of LLM quota and prompt token size when images are included.
"""

import os
import shutil
import zipfile
import tempfile
import subprocess
import difflib
import uuid
import html
import re
import mimetypes
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse

app = FastAPI(title="Code-Gen API (images-ready)")

# Configuration via environment
SOLUTIONS_DIR = Path(os.getenv("SOLUTIONS_DIR", "solutions"))
SOLUTIONS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-pro")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MAX_GENERATION_ATTEMPTS = int(os.getenv("MAX_GENERATION_ATTEMPTS", "4"))
EXECUTION_TIMEOUT = int(os.getenv("EXECUTION_TIMEOUT", "60"))  # seconds

# Max number of images to include in prompt to avoid token explosion
MAX_IMAGES_IN_PROMPT = int(os.getenv("MAX_IMAGES_IN_PROMPT", "6"))
# Max characters from each URI/text file to include in prompt
MAX_IMAGE_TEXT_CHARS = int(os.getenv("MAX_IMAGE_TEXT_CHARS", "800"))
# Max file download size
ALLOWED_DOWNLOAD_MAX_SIZE = int(os.getenv("ALLOWED_DOWNLOAD_MAX_SIZE", str(50 * 1024 * 1024)))

CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n([\s\S]*?)```", re.IGNORECASE)
PHOTO_PLACEHOLDER_RE = re.compile(r"\{\{\s*PHOTO_ID:(\d+)(?:\|WIDTH:(\d+))?\s*\}\}")

def extract_python_from_markdown(text: str) -> Optional[str]:
    if not text:
        return None
    try:
        if ("\n" in text or "\t" in text or '\"' in text) and text.count("\n") > text.count("\n"):
            decoded = bytes(text, "utf-8").decode("unicode_escape")
            if decoded.count("\n") >= text.count("\n"):
                text = decoded
    except Exception:
        pass
    m = CODE_FENCE_RE.search(text)
    code = m.group(1) if m else text
    code = html.unescape(code)
    code = code.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    return code

def unpack_zip_to_dir(zip_bytes: bytes, dest_dir: Path) -> None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmpf:
        tmpf.write(zip_bytes)
        tmpf.flush()
        tmpf_path = Path(tmpf.name)
    with zipfile.ZipFile(tmpf_path, 'r') as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            target_name = Path(member.filename).name
            target_path = dest_dir / target_name
            with zf.open(member) as src, open(target_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
    try:
        tmpf_path.unlink(missing_ok=True)
    except Exception:
        pass

def find_problem_files(workdir: Path) -> Dict[str, Any]:
    """
    Find statement, sample_in, sample_out and image-like files (URI text files or image files).
    Returns a dict with keys 'statement','sample_in','sample_out' and optionally 'images' (list of Paths).
    """
    file_keys = {
        'statement': ['statement', 'problem'],
        'sample_in': ['sample_in', 'sample.in', 'input', 'sample-input'],
        'sample_out': ['sample_out', 'sample.out', 'output', 'sample-output']
    }
    found: Dict[str, Any] = {}
    images: List[Path] = []
    for p in workdir.iterdir():
        if not p.is_file():
            continue
        name = p.name.lower()
        for key, patterns in file_keys.items():
            if key not in found and any(pat in name for pat in patterns):
                found[key] = p
        # detect explicit uri text files for images e.g. image_1110089084618291_uri.txt
        if 'image' in name and 'uri' in name:
            images.append(p)
            continue
        # detect files named like photo_ or containing a long numeric id used in placeholders
        if re.search(r'\d{6,}', name) and ('image' in name or 'photo' in name or '_uri' in name):
            images.append(p)
            continue
        # accept common image extensions
        if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
            images.append(p)
            continue
        # fallback: mimetype says image
        mtype, _ = mimetypes.guess_type(str(p))
        if mtype and mtype.startswith('image/'):
            images.append(p)
    if images:
        found['images'] = images
    return found

def read_image_uri_entry(p: Path, max_chars: int = MAX_IMAGE_TEXT_CHARS) -> str:
    """
    Read small text files containing URIs; for binary images return a filename marker.
    """
    try:
        if p.suffix.lower() in ['.txt', '.uri', '.url', '.link']:
            txt = p.read_text(encoding='utf-8', errors='replace').strip()
            txt = " ".join(txt.splitlines())
            if len(txt) > max_chars:
                txt = txt[:max_chars] + "...(truncated)"
            return txt
        size = p.stat().st_size
        if size <= 4096:
            txt = p.read_text(encoding='utf-8', errors='replace').strip()
            txt = " ".join(txt.splitlines())
            if len(txt) > max_chars:
                txt = txt[:max_chars] + "...(truncated)"
            return txt
    except Exception:
        pass
    return f"[binary image file: {p.name}]"

def extract_photo_placeholders(statement: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    Replace PHOTO_ID placeholders with readable markers and return list of placeholders found.
    """
    entries: List[Dict[str,str]] = []
    def _repl(m):
        pid = m.group(1)
        width = m.group(2) or ""
        entries.append({"id": pid, "width": width})
        if width:
            return f"[IMAGE id={pid} width={width}]"
        else:
            return f"[IMAGE id={pid}]"
    new_stmt = PHOTO_PLACEHOLDER_RE.sub(_repl, statement)
    return new_stmt, entries

def run_python_code_str(code_str: str, input_str: str, timeout: int = EXECUTION_TIMEOUT) -> Dict[str, Any]:
    try:
        p = subprocess.Popen(
            [os.getenv("PYTHON_PATH", "python"), "-c", code_str],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
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

def save_solution_on_server(solution_text: str, solution_dir: Path, metadata: Dict[str, Any]) -> Path:
    solution_dir.mkdir(parents=True, exist_ok=True)
    coding_path = solution_dir / 'coding_solution.py'
    solution_text_norm = solution_text.replace("\r\n", "\n").replace("\r", "\n")
    coding_path.write_text(solution_text_norm, encoding='utf-8')
    (solution_dir / 'metadata.json').write_text(str(metadata), encoding='utf-8')
    if 'raw_llm' in metadata and metadata['raw_llm'] is not None:
        (solution_dir / 'llm_response.txt').write_text(metadata['raw_llm'], encoding='utf-8')
    return coding_path

@app.get("/health")
def health():
    return {"status": "ok", "model_configured": bool(GOOGLE_API_KEY)}

@app.post("/generate")
async def generate(file: UploadFile = File(...)):
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="Server missing GOOGLE_API_KEY env var for generation.")

    tmp_root = Path(tempfile.mkdtemp(prefix="gen_"))
    try:
        content = await file.read()
        unpack_zip_to_dir(content, tmp_root)
        found = find_problem_files(tmp_root)
        if 'statement' not in found or 'sample_in' not in found or 'sample_out' not in found:
            raise HTTPException(status_code=400, detail="Zip must contain statement, sample_in, sample_out files.")

        statement_text = found['statement'].read_text(encoding='utf-8')
        statement_text_replaced, photo_placeholders = extract_photo_placeholders(statement_text)

        sample_in_text = found['sample_in'].read_text(encoding='utf-8')
        sample_out_text = found['sample_out'].read_text(encoding='utf-8')

        images = found.get('images', [])
        image_entries = []
        for p in images[:MAX_IMAGES_IN_PROMPT]:
            image_entries.append({"file": p.name, "uri": read_image_uri_entry(p, max_chars=MAX_IMAGE_TEXT_CHARS)})
        image_uris_text = ""
        if image_entries:
            image_uris_text = "\n\nImage entries detected:\n" + "\n".join(f"- {e['file']}: {e['uri']}" for e in image_entries)
            if photo_placeholders:
                pid_map_lines = []
                for ph in photo_placeholders:
                    pid = ph['id']
                    width = ph['width']
                    match_file = None
                    for p in images:
                        if pid in p.name:
                            match_file = p.name
                            break
                    if match_file:
                        pid_map_lines.append(f"- PHOTO_ID {pid} -> file {match_file} (width={width})")
                    else:
                        pid_map_lines.append(f"- PHOTO_ID {pid} -> [no matching file found] (width={width})")
                image_uris_text += "\n\nPhoto placeholders mapping:\n" + "\n".join(pid_map_lines)

        # LLM client
        try:
            import google.generativeai as genai
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Missing LLM client library: {e}")

        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)

        last_code = None
        last_error = ""
        raw_llm_text = None
        for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
            if attempt == 1:
                prompt = f"""You are an expert competitive programmer. Write a Python 3 solution that reads from standard input and writes to standard output.

Problem statement:
{statement_text_replaced}

Sample Input:
{sample_in_text}

Sample Output:
{sample_out_text}

{image_uris_text}

Provide only the final Python code in a single markdown code block (```python ... ```)."""
            else:
                prompt = f"""Previous submission produced incorrect output or runtime errors.

Problem statement:
{statement_text_replaced}

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

{image_uris_text}

Please provide a corrected complete Python solution in one markdown block."""
            try:
                resp = model.generate_content(prompt)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

            raw_llm_text = resp.text or ""
            code_candidate = extract_python_from_markdown(raw_llm_text) or (resp.text or "").strip()
            last_code = code_candidate

            sample_run = run_python_code_str(code_candidate, sample_in_text, timeout=EXECUTION_TIMEOUT)
            sample_out_norm = "\n".join(line.rstrip() for line in sample_run["stdout"].strip().splitlines())
            expected_norm = "\n".join(line.rstrip() for line in sample_out_text.strip().splitlines())

            if sample_run["timed_out"]:
                last_error = f"Sample run timed out: {sample_run['stderr']}"
                continue
            if sample_run["stderr"]:
                last_error = f"Sample runtime error: {sample_run['stderr']}"
                continue
            if sample_out_norm != expected_norm:
                diff = "".join(difflib.unified_diff(expected_norm.splitlines(keepends=True), sample_out_norm.splitlines(keepends=True), fromfile='expected', tofile='actual'))
                last_error = f"Sample mismatch. Diff:\n{diff}\nStdout:\n{sample_run['stdout']}\nStderr:\n{sample_run['stderr']}"
                continue

            # Success -> persist and return
            solution_id = str(uuid.uuid4())
            solution_dir = SOLUTIONS_DIR / solution_id
            metadata = {"solution_id": solution_id, "attempt": attempt, "raw_llm": raw_llm_text}
            coding_path = save_solution_on_server(code_candidate, solution_dir, metadata)

            # save artifacts and copy detected images into solution dir
            (solution_dir / 'sample_stdout.txt').write_text(sample_run['stdout'], encoding='utf-8')
            (solution_dir / 'statement.txt').write_text(statement_text, encoding='utf-8')
            (solution_dir / 'sample_in.txt').write_text(sample_in_text, encoding='utf-8')
            (solution_dir / 'sample_out.txt').write_text(sample_out_text, encoding='utf-8')
            (solution_dir / 'gen_response.json').write_text(str({"attempt": attempt, "raw_llm": raw_llm_text}), encoding='utf-8')
            for p in images:
                try:
                    shutil.copy(p, solution_dir / p.name)
                except Exception:
                    pass
            metadata['images'] = [p.name for p in images]
            (solution_dir / 'metadata.json').write_text(str(metadata), encoding='utf-8')

            return JSONResponse({
                'status': 'generated',
                'solution_id': solution_id,
                'sample_stdout': sample_run['stdout'],
                'solution_path': str(coding_path),
                'solution': code_candidate,
                'raw_llm': raw_llm_text
            })

        return JSONResponse({'status': 'failed', 'attempts': MAX_GENERATION_ATTEMPTS, 'last_error': last_error, 'last_solution': last_code}, status_code=400)
    finally:
        try:
            shutil.rmtree(tmp_root)
        except Exception:
            pass

@app.post("/test")
async def test_solution(
    solution_id: Optional[str] = Form(None),
    test_input: Optional[str] = Form(None),
    test_file: Optional[UploadFile] = File(None),
    solution: Optional[str] = Form(None),
    solution_file: Optional[UploadFile] = File(None),
    problem_zip: Optional[UploadFile] = File(None),
    statement: Optional[str] = Form(None),
    sample_in: Optional[str] = Form(None),
    sample_out: Optional[str] = Form(None),
    test_expected: Optional[str] = Form(None)
):
    if test_file is None and test_input is None:
        raise HTTPException(status_code=400, detail="Provide test_file (upload) or test_input (form field).")

    created_new = False
    if solution_file is not None or solution is not None or problem_zip is not None:
        if solution_id is None:
            solution_id = str(uuid.uuid4())
            created_new = True
        solution_dir = SOLUTIONS_DIR / solution_id
        solution_dir.mkdir(parents=True, exist_ok=True)

        if problem_zip is not None:
            zip_bytes = await problem_zip.read()
            tempd = Path(tempfile.mkdtemp())
            try:
                unpack_zip_to_dir(zip_bytes, tempd)
                found = find_problem_files(tempd)
                if 'statement' in found:
                    shutil.copy(found['statement'], solution_dir / 'statement.txt')
                if 'sample_in' in found:
                    shutil.copy(found['sample_in'], solution_dir / 'sample_in.txt')
                if 'sample_out' in found:
                    shutil.copy(found['sample_out'], solution_dir / 'sample_out.txt')
                # copy images if any
                for p in found.get('images', []):
                    try:
                        shutil.copy(p, solution_dir / p.name)
                    except Exception:
                        pass
            finally:
                try:
                    shutil.rmtree(tempd)
                except Exception:
                    pass

        if solution_file is not None:
            sol_bytes = await solution_file.read()
            sol_text = sol_bytes.decode('utf-8', errors='replace')
            sol_text = extract_python_from_markdown(sol_text) or sol_text.strip()
            metadata = {"solution_id": solution_id, "provided_file": True}
            coding_path = save_solution_on_server(sol_text, solution_dir, metadata)
        elif solution is not None:
            sol_text = extract_python_from_markdown(solution) or solution.strip()
            metadata = {"solution_id": solution_id, "provided_directly": True}
            coding_path = save_solution_on_server(sol_text, solution_dir, metadata)
        else:
            coding_path = solution_dir / 'coding_solution.py'
            if not coding_path.exists():
                raise HTTPException(status_code=400, detail="No solution provided and no existing coding_solution.py for this solution_id")
    else:
        if solution_id is None:
            raise HTTPException(status_code=400, detail="Provide either an existing solution_id or upload solution_file/solution text")
        solution_dir = SOLUTIONS_DIR / solution_id
        if not solution_dir.exists():
            raise HTTPException(status_code=404, detail="solution_id not found")
        coding_path = solution_dir / 'coding_solution.py'
        if not coding_path.exists():
            raise HTTPException(status_code=404, detail="coding_solution.py not found for this solution_id")

    if test_file is not None:
        test_input_text = (await test_file.read()).decode('utf-8')
    else:
        test_input_text = test_input

    current_code = coding_path.read_text(encoding='utf-8')
    run_res = run_python_code_str(current_code, test_input_text, timeout=EXECUTION_TIMEOUT)

    def normalize_out(s: str) -> str:
        return "\n".join(line.rstrip() for line in s.strip().splitlines())

    if not run_res['timed_out'] and run_res['stderr'] == '' and (test_expected is None or normalize_out(run_res['stdout']) == normalize_out(test_expected)):
        (solution_dir / 'test_output.txt').write_text(run_res['stdout'], encoding='utf-8')
        return JSONResponse({
            'status': 'ok',
            'solution_id': solution_id,
            'test_stdout': run_res['stdout'],
            'test_stderr': run_res['stderr'],
            'test_output_path': str(solution_dir / 'test_output.txt'),
            'solution': current_code
        })

    (solution_dir / 'test_output.txt').write_text(run_res['stdout'] + "\n[stderr]\n" + run_res['stderr'], encoding='utf-8')

    if not GOOGLE_API_KEY:
        return JSONResponse({'status': 'failed', 'reason': 'no_google_api_key', 'run_result': run_res, 'solution': current_code}, status_code=400)

    try:
        import google.generativeai as genai
    except Exception as e:
        return JSONResponse({'status': 'failed', 'reason': f'missing_llm_lib: {e}', 'run_result': run_res, 'solution': current_code}, status_code=500)

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)

    last_code = current_code
    last_error = f"Initial run failed. stdout:\n{run_res['stdout']}\nstderr:\n{run_res['stderr']}"

    # prepare image_uris_text from files saved in solution_dir (if any)
    images_in_dir = [p for p in solution_dir.iterdir() if p.is_file() and ('image' in p.name.lower() or 'photo' in p.name.lower() or p.suffix.lower() in ['.png','.jpg','.jpeg','.gif','.txt'])]
    image_entries = []
    for p in images_in_dir[:MAX_IMAGES_IN_PROMPT]:
        image_entries.append({"file": p.name, "uri": read_image_uri_entry(p, max_chars=MAX_IMAGE_TEXT_CHARS)})
    image_uris_text = ""
    if image_entries:
        image_uris_text = "\n\nImage entries detected:\n" + "\n".join(f"- {e['file']}: {e['uri']}" for e in image_entries)

    statement_text = (solution_dir / 'statement.txt').read_text(encoding='utf-8') if (solution_dir / 'statement.txt').exists() else (statement or "")
    statement_text_replaced, _ = extract_photo_placeholders(statement_text)
    sample_in_text = (solution_dir / 'sample_in.txt').read_text(encoding='utf-8') if (solution_dir / 'sample_in.txt').exists() else (sample_in or "")
    sample_out_text = (solution_dir / 'sample_out.txt').read_text(encoding='utf-8') if (solution_dir / 'sample_out.txt').exists() else (sample_out or "")

    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        prompt = f"""You are an expert competitive programmer. Previously the following solution was produced for the problem statement below. It passed the sample tests but it failed on a later test input. Please provide a corrected complete Python 3 solution that (1) still passes the provided sample input/output and (2) runs correctly on the failing test input.

Problem statement:
{statement_text_replaced}

Sample Input:
{sample_in_text}

Sample Output:
{sample_out_text}

{image_uris_text}

Previous code:
```python
{last_code}
```

Failure when running on this test input:
Test Input:
{test_input_text}

Failure details:
{last_error}

If a corrected solution is provided, reply with the full Python code in a single markdown code block (```python ... ```).
"""
        try:
            resp = model.generate_content(prompt)
        except Exception as e:
            return JSONResponse({'status': 'failed', 'reason': f'LLM_call_failed: {e}', 'run_result': run_res, 'solution': current_code}, status_code=500)

        raw_llm_text = resp.text or ""
        code_candidate = extract_python_from_markdown(raw_llm_text) or (raw_llm_text or "").strip()
        last_code = code_candidate

        if sample_in_text and sample_out_text:
            sample_run = run_python_code_str(code_candidate, sample_in_text, timeout=EXECUTION_TIMEOUT)
            sample_out_norm = "\n".join(line.rstrip() for line in sample_run["stdout"].strip().splitlines())
            expected_norm = "\n".join(line.rstrip() for line in sample_out_text.strip().splitlines())

            if sample_run["timed_out"]:
                last_error = f"Sample run timed out: {sample_run['stderr']}"
                continue
            if sample_run["stderr"]:
                last_error = f"Sample runtime error after regen: {sample_run['stderr']}"
                continue
            if sample_out_norm != expected_norm:
                diff = "".join(difflib.unified_diff(expected_norm.splitlines(keepends=True), sample_out_norm.splitlines(keepends=True), fromfile='expected', tofile='actual'))
                last_error = f"Sample mismatch after regen. Diff:\\n{diff}\\nStdout:\\n{sample_run['stdout']}\\nStderr:\\n{sample_run['stderr']}"
                continue

        test_run = run_python_code_str(code_candidate, test_input_text, timeout=EXECUTION_TIMEOUT)
        test_out_norm = "\n".join(line.rstrip() for line in test_run["stdout"].strip().splitlines())

        if test_run["timed_out"]:
            last_error = f"Test run timed out: {test_run['stderr']}"
            continue
        if test_run["stderr"]:
            last_error = f"Test runtime error after regen: {test_run['stderr']}"
            continue
        if test_expected is not None:
            expected_test_norm = "\n".join(line.rstrip() for line in test_expected.strip().splitlines())
            if test_out_norm != expected_test_norm:
                diff = "".join(difflib.unified_diff(expected_test_norm.splitlines(keepends=True), test_out_norm.splitlines(keepends=True), fromfile='expected_test', tofile='actual_test'))
                last_error = f"Test mismatch after regen. Diff:\\n{diff}\\nStdout:\\n{test_run['stdout']}\\nStderr:\\n{test_run['stderr']}"
                continue
        else:
            if test_out_norm == "":
                last_error = "Test run produced empty stdout after regen."
                continue

        coding_path.write_text(code_candidate, encoding='utf-8')
        (solution_dir / 'test_output.txt').write_text(test_run['stdout'], encoding='utf-8')
        (solution_dir / 'llm_response.txt').write_text(raw_llm_text, encoding='utf-8')
        (solution_dir / 'metadata.json').write_text(str({'solution_id': solution_id, 'regenerated_attempt': attempt}), encoding='utf-8')

        return JSONResponse({'status': 'ok', 'solution_id': solution_id, 'test_stdout': test_run['stdout'], 'test_stderr': test_run['stderr'], 'test_output_path': str(solution_dir / 'test_output.txt'), 'solution': code_candidate, 'attempts': attempt})

    (solution_dir / 'test_output.txt').write_text(run_res['stdout'] + "\n[stderr]\n" + run_res['stderr'], encoding='utf-8')
    return JSONResponse({'status': 'failed', 'reason': 'regeneration_exhausted', 'last_error': last_error, 'last_solution': last_code, 'solution': last_code}, status_code=400)

@app.get("/download/{solution_id}/{filename}")
def download_file(solution_id: str, filename: str):
    solution_dir = SOLUTIONS_DIR / solution_id
    if not solution_dir.exists():
        raise HTTPException(status_code=404, detail="solution_id not found")
    target = (solution_dir / filename).resolve()
    if not str(target).startswith(str(solution_dir.resolve()) + os.sep):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    if target.stat().st_size > ALLOWED_DOWNLOAD_MAX_SIZE:
        raise HTTPException(status_code=413, detail="File too large to download")
    return FileResponse(str(target), media_type='application/octet-stream', filename=target.name)

@app.get("/solutions/{solution_id}/files")
def list_solution_files(solution_id: str):
    solution_dir = SOLUTIONS_DIR / solution_id
    if not solution_dir.exists():
        raise HTTPException(status_code=404, detail="solution_id not found")
    files = [p.name for p in solution_dir.iterdir() if p.is_file()]
    return {"solution_id": solution_id, "files": files}

@app.get("/solutions")
def list_solutions():
    ids = []
    for p in SOLUTIONS_DIR.iterdir():
        if p.is_dir():
            ids.append({"solution_id": p.name, "files": [f.name for f in p.iterdir() if f.is_file()]})
    return {"solutions": ids}
