# api_server_final_multimodal.py
"""
Fully-featured Code-Gen API server (multimodal updated).
Saves artifacts, generate/test/regenerate, and can pass PIL.Image objects to Gemini (preferred method).
Run: uvicorn api_server_final_multimodal:app --host 0.0.0.0 --port 8000
"""

import os, re, html, uuid, shutil, tempfile, zipfile, difflib, subprocess, mimetypes, io
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse

# optional libs
try:
    import requests
except Exception:
    requests = None
try:
    from PIL import Image
except Exception:
    Image = None

# Configuration
SOLUTIONS_DIR = Path(os.getenv("SOLUTIONS_DIR", "solutions"))
SOLUTIONS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-pro")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MAX_GENERATION_ATTEMPTS = int(os.getenv("MAX_GENERATION_ATTEMPTS", "4"))
EXECUTION_TIMEOUT = int(os.getenv("EXECUTION_TIMEOUT", "60"))
MULTIMODAL_MODE = os.getenv("MULTIMODAL_MODE", "1") == "1"   # enable the PIL-image-in-prompt approach
ENABLE_IMAGE_DOWNLOAD = os.getenv("ENABLE_IMAGE_DOWNLOAD", "1") == "1"
MAX_IMAGES_IN_PROMPT = int(os.getenv("MAX_IMAGES_IN_PROMPT", "6"))

ALLOWED_DOWNLOAD_FILES = {
    'coding_solution.py', 'test_output.txt', 'sample_stdout.txt',
    'statement.txt', 'sample_in.txt', 'sample_out.txt', 'llm_response.txt',
    'metadata.json', 'gen_response.json'
}

CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n([\s\S]*?)```", re.IGNORECASE)
PHOTO_PLACEHOLDER_RE = re.compile(r"\{\{\s*PHOTO_ID:(\d+)(?:\|WIDTH:(\d+))?\s*\}\}")

app = FastAPI(title="Code-Gen Two-Step Final API (multimodal)")

def extract_python_from_markdown(text: str) -> Optional[str]:
    if not text:
        return None
    try:
        if "\\n" in text and text.count("\\n") > text.count("\n"):
            try:
                decoded = bytes(text, "utf-8").decode("unicode_escape")
                if decoded.count("\n") >= text.count("\\n"):
                    text = decoded
            except Exception:
                pass
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
        # heuristics for images / uri text files
        if ('image' in name or 'photo' in name) and ('uri' in name or name.endswith('.txt') or name.endswith('.url')):
            images.append(p)
            continue
        if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
            images.append(p)
            continue
        mtype, _ = mimetypes.guess_type(str(p))
        if mtype and mtype.startswith('image/'):
            images.append(p)
    if images:
        found['images'] = images
    return found

def extract_photo_placeholders(statement: str) -> Tuple[str, List[Dict[str,str]]]:
    entries: List[Dict[str,str]] = []
    def repl(m):
        pid = m.group(1)
        width = m.group(2) or ""
        entries.append({"id": pid, "width": width})
        return f"[IMAGE id={pid}" + (f" width={width}]" if width else "]")
    new_stmt = PHOTO_PLACEHOLDER_RE.sub(repl, statement)
    return new_stmt, entries

def read_image_uri_text(p: Path) -> str:
    try:
        txt = p.read_text(encoding='utf-8', errors='replace').strip()
        return " ".join(txt.splitlines())
    except Exception:
        return ""

def fetch_image_as_pil(url: str, timeout: int = 10):
    if requests is None or Image is None:
        return None
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200 and resp.content:
            img = Image.open(io.BytesIO(resp.content))
            return img
    except Exception:
        return None
    return None

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
    coding_path.write_text(solution_text.replace("\r\n", "\n"), encoding='utf-8')
    (solution_dir / 'metadata.json').write_text(str(metadata), encoding='utf-8')
    if 'raw_llm' in metadata and metadata['raw_llm'] is not None:
        (solution_dir / 'llm_response.txt').write_text(metadata['raw_llm'], encoding='utf-8')
    return coding_path


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_configured": bool(GOOGLE_API_KEY),
        "multimodal_mode": MULTIMODAL_MODE,
        "requests_installed": requests is not None,
        "PIL_installed": Image is not None
    }

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

        image_files = found.get('images', [])[:MAX_IMAGES_IN_PROMPT]
        image_urls = []
        pil_images = []  # list of {'name':..., 'img':PIL.Image, 'uri':...}

        # Collect URIs and prefetched PIL images (for prompt_parts)
        for p in image_files:
            if p.suffix.lower() in ['.txt', '.uri', '.url', '.link']:
                uri = read_image_uri_text(p)
                if uri.startswith("http://") or uri.startswith("https://"):
                    image_urls.append(uri)
                    # Try fetch as PIL if available (preferred method)
                    pil = fetch_image_as_pil(uri)
                    if pil is not None:
                        pil_images.append({"name": p.name, "img": pil, "uri": uri})
                else:
                    image_urls.append(uri)
            else:
                # local image file inside ZIP
                try:
                    if Image is not None:
                        pil = Image.open(str(p))
                        pil_images.append({"name": p.name, "img": pil, "uri": None})
                    else:
                        image_urls.append(str(p.name))
                except Exception:
                    image_urls.append(str(p.name))

        # import LLM client
        try:
            import google.generativeai as genai
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Missing LLM client library: {e}")

        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)

        last_code = None
        last_error = ""
        raw_llm_text = None
        a=""

        for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
            if attempt == 1:
                prompt = f"""
You are an expert competitive programmer. Write a Python 3 solution that reads from standard input and writes to standard output. I will also provide you some images, tell me were they useful for you and what you could understand from them

Problem statement:
{statement_text_replaced}

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
{statement_text_replaced}

Previous code:
```python
{last_code}

Reason for failure:
{last_error}

Sample Input:
{sample_in_text}

Expected Sample Output:
{sample_out_text}

Please provide a corrected complete Python solution in one markdown block.
"""
            resp = None
            multimodal_errors = []

            # --- Preferred new approach: pass PIL.Image objects as prompt parts ---
            if MULTIMODAL_MODE and pil_images:
                a+="ONLY 1 WAS USED"
                # build prompt_parts list: prompt text followed by PIL images
                prompt_parts = [prompt]
                for pi in pil_images[:MAX_IMAGES_IN_PROMPT]:
                    prompt_parts.append(pi['img'])
                try:
                    resp = model.generate_content(prompt_parts)
                except Exception as e:
                    multimodal_errors.append(f"prompt_parts (PIL) attempt failed: {e}")
                    resp = None

            # --- Fallbacks if the above fails ---
            if resp is None and MULTIMODAL_MODE and image_urls:
                a+="2 WAS ALSO USED"
                try:
                    resp = model.generate_content(prompt, referenced_image_urls=image_urls)
                except TypeError:
                    try:
                        resp = model.generate_content(prompt, image_urls=image_urls)
                    except Exception as e:
                        multimodal_errors.append(f"image_urls attempt failed: {e}")
                except Exception as e:
                    multimodal_errors.append(f"referenced_image_urls attempt failed: {e}")

            # Final fallback: text-only prompt, with mapping of PHOTO_IDs -> filenames/urls
            if resp is None:
                a+="+3RD METHOD USED"
                image_map_text = ""
                if photo_placeholders:
                    pid_map_lines = []
                    for ph in photo_placeholders:
                        pid = ph['id']
                        width = ph['width']
                        match_file = None
                        for p in image_files:
                            if pid in p.name:
                                match_file = p.name
                                break
                        pid_map_lines.append(f"- PHOTO_ID {pid} -> file {match_file or '[no match]'} (width={width})")
                    image_map_text += "\n".join(pid_map_lines)
                if image_urls:
                    image_map_text += "\nImage URLs:\n" + "\n".join(image_urls)
                if image_map_text:
                    prompt = prompt + "\n\n" + image_map_text
                try:
                    resp = model.generate_content(prompt)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"LLM generate_content failed: {e}; multimodal_errors: {multimodal_errors}")

            raw_llm_text = getattr(resp, "text", str(resp)) or ""
            code_candidate = extract_python_from_markdown(raw_llm_text) or raw_llm_text.strip()
            last_code = code_candidate

            # Run candidate on sample input
            sample_run = run_python_code_str(code_candidate, sample_in_text, timeout=EXECUTION_TIMEOUT)
            out_norm = "\n".join(line.rstrip() for line in sample_run["stdout"].strip().splitlines())
            expected_norm = "\n".join(line.rstrip() for line in sample_out_text.strip().splitlines())

            if sample_run["timed_out"]:
                last_error = sample_run["stderr"]
                continue
            if sample_run["stderr"]:
                last_error = sample_run["stderr"]
                continue
            if out_norm == expected_norm:
                # success -> save artifacts
                solution_id = str(uuid.uuid4())
                solution_dir = SOLUTIONS_DIR / solution_id
                metadata = {"solution_id": solution_id, "attempt": attempt, "raw_llm": raw_llm_text}
                coding_path = save_solution_on_server(code_candidate, solution_dir, metadata)
                (solution_dir / 'sample_stdout.txt').write_text(sample_run['stdout'], encoding='utf-8')
                (solution_dir / 'statement.txt').write_text(statement_text, encoding='utf-8')
                (solution_dir / 'sample_in.txt').write_text(sample_in_text, encoding='utf-8')
                (solution_dir / 'sample_out.txt').write_text(sample_out_text, encoding='utf-8')
                (solution_dir / 'gen_response.json').write_text(str({"attempt": attempt}), encoding='utf-8')
                for p in image_files:
                    try:
                        shutil.copy(p, solution_dir / p.name)
                    except Exception:
                        pass
                (solution_dir / 'metadata.json').write_text(str({"solution_id": solution_id, "images": [p.name for p in image_files]}), encoding='utf-8')
                return JSONResponse({
                    "status": "generated",
                    "solution_id": solution_id,
                    "Which_method_used": a,
                    "sample_stdout": sample_run["stdout"],
                    "solution": code_candidate,
                    "raw_llm_text": raw_llm_text

                })
            else:
                diff = "".join(difflib.unified_diff(expected_norm.splitlines(keepends=True), out_norm.splitlines(keepends=True), fromfile="expected", tofile="actual"))
                last_error = f"Wrong output. Diff:\n{diff}\nStdout:\n{sample_run['stdout']}\nStderr:\n{sample_run['stderr']}"

        return JSONResponse({
            "status": "failed",
            "attempts": MAX_GENERATION_ATTEMPTS,
            "Which_method_used": a,
            "last_error": last_error,
            "last_solution": last_code,
            "raw_llm_text": raw_llm_text

        }, status_code=400)
    finally:
        try:
            shutil.rmtree(tmp_root)
        except Exception:
            pass


@app.post('/test')
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
        raise HTTPException(status_code=400, detail='Provide test_file or test_input.')

    created_new = False
    if solution_file is not None or solution is not None or problem_zip is not None:
        if solution_id is None:
            solution_id = str(uuid.uuid4()); created_new = True
        solution_dir = SOLUTIONS_DIR / solution_id; solution_dir.mkdir(parents=True, exist_ok=True)
        if problem_zip is not None:
            zip_bytes = await problem_zip.read(); tmpd = Path(tempfile.mkdtemp());
            try:
                unpack_zip_to_dir(zip_bytes, tmpd); found = find_problem_files(tmpd)
                if 'statement' in found: shutil.copy(found['statement'], solution_dir / 'statement.txt')
                if 'sample_in' in found: shutil.copy(found['sample_in'], solution_dir / 'sample_in.txt')
                if 'sample_out' in found: shutil.copy(found['sample_out'], solution_dir / 'sample_out.txt')
                for p in found.get('images', []):
                    try:
                        shutil.copy(p, solution_dir / p.name)
                    except Exception:
                        pass
            finally:
                try:
                    shutil.rmtree(tmpd)
                except Exception:
                    pass
        if solution_file is not None:
            sol_bytes = await solution_file.read(); sol_text = sol_bytes.decode('utf-8', errors='replace'); sol_text = extract_python_from_markdown(sol_text) or sol_text.strip(); save_solution_on_server(sol_text, solution_dir, {'solution_id': solution_id, 'provided_file': True})
        elif solution is not None:
            sol_text = extract_python_from_markdown(solution) or solution.strip(); save_solution_on_server(sol_text, solution_dir, {'solution_id': solution_id, 'provided': True})
    else:
        if solution_id is None: raise HTTPException(status_code=400, detail='Provide solution_id or upload solution_file/solution text')
        solution_dir = SOLUTIONS_DIR / solution_id
        if not solution_dir.exists(): raise HTTPException(status_code=404, detail='solution_id not found')

    if test_file is not None:
        test_input_text = (await test_file.read()).decode('utf-8')
    else:
        test_input_text = test_input

    coding_path = solution_dir / 'coding_solution.py'
    if not coding_path.exists():
        raise HTTPException(status_code=404, detail='coding_solution.py not found for this solution_id')
    current_code = coding_path.read_text(encoding='utf-8')

    run_res = run_python_code_str(current_code, test_input_text, timeout=EXECUTION_TIMEOUT)

    def normalize_out(s: str) -> str:
        return '\n'.join(line.rstrip() for line in s.strip().splitlines())

    if not run_res['timed_out'] and run_res['stderr'] == '' and (test_expected is None or normalize_out(run_res['stdout']) == normalize_out(test_expected)):
        (solution_dir / 'test_output.txt').write_text(run_res['stdout'], encoding='utf-8')
        return JSONResponse({'status': 'ok', 'solution_id': solution_id, 'test_stdout': run_res['stdout'], 'test_stderr': run_res['stderr'], 'test_output_path': str(solution_dir / 'test_output.txt'), 'solution': current_code})

    (solution_dir / 'test_output.txt').write_text(run_res['stdout'] + '\n[stderr]\n' + run_res['stderr'], encoding='utf-8')

    if not GOOGLE_API_KEY:
        return JSONResponse({'status': 'failed', 'reason': 'no_google_api_key', 'run_result': run_res, 'solution': current_code}, status_code=400)

    try:
        import google.generativeai as genai
    except Exception as e:
        return JSONResponse({'status': 'failed', 'reason': f'missing_llm_lib: {e}', 'run_result': run_res, 'solution': current_code}, status_code=500)
    genai.configure(api_key=GOOGLE_API_KEY); model = genai.GenerativeModel(MODEL_NAME)

    last_code = current_code; last_error = f"Initial run failed. stdout:\n{run_res['stdout']}\nstderr:\n{run_res['stderr']}"

    images_in_dir = [p for p in solution_dir.iterdir() if p.is_file() and (('image' in p.name.lower()) or ('photo' in p.name.lower()) or p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.txt'])][:MAX_IMAGES_IN_PROMPT]
    image_urls = []; downloaded_local_paths = []
    for p in images_in_dir:
        if p.suffix.lower() in ['.txt', '.uri', '.url']:
            uri = read_image_uri_text(p)
            if uri.startswith('http://') or uri.startswith('https://'):
                image_urls.append(uri)
                # if ENABLE_IMAGE_DOWNLOAD:
                #     tmpd = Path(tempfile.mkdtemp()); local_path = tmpd / (p.stem + os.path.splitext(uri.split('?')[0])[1] if os.path.splitext(uri.split('?')[0])[1] else p.stem + '.jpg');
                #     if download_image_from_url(uri, local_path): downloaded_local_paths.append(local_path)
        else:
            downloaded_local_paths.append(p)

    statement_text = (solution_dir / 'statement.txt').read_text(encoding='utf-8') if (solution_dir / 'statement.txt').exists() else (statement or "")
    statement_text_replaced, photo_placeholders = extract_photo_placeholders(statement_text)
    sample_in_text = (solution_dir / 'sample_in.txt').read_text(encoding='utf-8') if (solution_dir / 'sample_in.txt').exists() else (sample_in or "")
    sample_out_text = (solution_dir / 'sample_out.txt').read_text(encoding='utf-8') if (solution_dir / 'sample_out.txt').exists() else (sample_out or "")

    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        prompt_text = f"""You are an expert competitive programmer. Previously the following solution was produced for the problem statement below. It passed the sample tests but it failed on a later test input. Please provide a corrected complete Python 3 solution that (1) still passes the provided sample input/output and (2) runs correctly on the failing test input.

Problem statement:
{statement_text_replaced}

Sample Input:
{sample_in_text}

Sample Output:
{sample_out_text}

Previous code:
{last_code}

Failure when running on this test input:
Test Input:
{test_input_text}

Failure details:
{last_error}

If a corrected solution is provided, reply with the full Python code in a single markdown code block (python ... ).
"""

        resp = None
        if MULTIMODAL_MODE and (image_urls or downloaded_local_paths):
            try:
                resp = model.generate_content(prompt_text, referenced_image_urls=image_urls)
            except Exception:
                try:
                    resp = model.generate_content(prompt_text, image_urls=image_urls)
                except Exception:
                    pass
            if resp is None and downloaded_local_paths:
                try:
                    image_bytes = [p.read_bytes() for p in downloaded_local_paths]
                    try:
                        resp = model.generate_content(prompt_text, referenced_images=image_bytes)
                    except Exception:
                        resp = model.generate_content(prompt_text, images=image_bytes)
                except Exception:
                    pass

        if resp is None:
            image_map_text = ''
            if photo_placeholders:
                map_lines = []
                for ph in photo_placeholders:
                    pid = ph.get('id')
                    match = None
                    for p in images_in_dir:
                        if pid in p.name: match = p.name; break
                    map_lines.append(f"PHOTO_ID {pid} maps to file {match or '[no-match]'}")
                image_map_text += '\n'.join(map_lines)
            if image_urls: image_map_text += '\nImage URLs:\n' + '\n'.join(image_urls)
            if image_map_text: prompt_text = prompt_text + '\n\n' + image_map_text
            try:
                resp = model.generate_content(prompt_text)
            except Exception as e:
                return JSONResponse({'status': 'failed', 'reason': f'LLM_call_failed: {e}', 'run_result': run_res, 'solution': current_code}, status_code=500)

        raw_llm_text = getattr(resp, 'text', str(resp)) or ""
        code_candidate = extract_python_from_markdown(raw_llm_text) or (raw_llm_text or "").strip()
        last_code = code_candidate

        if sample_in_text and sample_out_text:
            sample_run = run_python_code_str(code_candidate, sample_in_text, timeout=EXECUTION_TIMEOUT)
            sample_out_norm = '\n'.join(line.rstrip() for line in sample_run['stdout'].strip().splitlines())
            expected_norm = '\n'.join(line.rstrip() for line in sample_out_text.strip().splitlines())
            if sample_run['timed_out']:
                last_error = f"Sample run timed out: {sample_run['stderr']}"; continue
            if sample_run['stderr']:
                last_error = f"Sample runtime error after regen: {sample_run['stderr']}"; continue
            if sample_out_norm != expected_norm:
                diff = ''.join(difflib.unified_diff(expected_norm.splitlines(keepends=True), sample_out_norm.splitlines(keepends=True), fromfile='expected', tofile='actual'))
                last_error = f"Sample mismatch after regen. Diff:\n{diff}\nStdout:\n{sample_run['stdout']}\nStderr:\n{sample_run['stderr']}"; continue

        test_run = run_python_code_str(code_candidate, test_input_text, timeout=EXECUTION_TIMEOUT)
        test_out_norm = '\n'.join(line.rstrip() for line in test_run['stdout'].strip().splitlines())
        if test_run['timed_out']:
            last_error = f"Test run timed out: {test_run['stderr']}"; continue
        if test_run['stderr']:
            last_error = f"Test runtime error after regen: {test_run['stderr']}"; continue
        if test_expected is not None:
            expected_test_norm = '\n'.join(line.rstrip() for line in test_expected.strip().splitlines())
            if test_out_norm != expected_test_norm:
                diff = ''.join(difflib.unified_diff(expected_test_norm.splitlines(keepends=True), test_out_norm.splitlines(keepends=True), fromfile='expected_test', tofile='actual_test'))
                last_error = f"Test mismatch after regen. Diff:\n{diff}\nStdout:\n{test_run['stdout']}\nStderr:\n{test_run['stderr']}"; continue
        else:
            if test_out_norm == "":
                last_error = "Test run produced empty stdout after regen."; continue

        coding_path.write_text(code_candidate, encoding='utf-8')
        (solution_dir / 'test_output.txt').write_text(test_run['stdout'], encoding='utf-8')
        (solution_dir / 'llm_response.txt').write_text(raw_llm_text, encoding='utf-8')
        (solution_dir / 'metadata.json').write_text(str({'solution_id': solution_id, 'regenerated_attempt': attempt}), encoding='utf-8')
        return JSONResponse({'status': 'ok', 'solution_id': solution_id, 'test_stdout': test_run['stdout'], 'test_stderr': test_run['stderr'], 'test_output_path': str(solution_dir / 'test_output.txt'), 'solution': code_candidate, 'attempts': attempt})

    (solution_dir / 'test_output.txt').write_text(run_res['stdout'] + '\n[stderr]\n' + run_res['stderr'], encoding='utf-8')
    return JSONResponse({'status': 'failed', 'reason': 'regeneration_exhausted', 'last_error': last_error, 'last_solution': last_code, 'solution': last_code}, status_code=400)


@app.get('/download/{solution_id}/{filename}')
def download_file(solution_id: str, filename: str):
    solution_dir = SOLUTIONS_DIR / solution_id
    if not solution_dir.exists():
        raise HTTPException(status_code=404, detail='solution_id not found')
    target = (solution_dir / filename).resolve()
    if not str(target).startswith(str(solution_dir.resolve()) + os.sep):
        raise HTTPException(status_code=400, detail='Invalid filename')
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail='File not found')
    return FileResponse(str(target), media_type='application/octet-stream', filename=target.name)


@app.get('/solutions/{solution_id}/files')
def list_solution_files(solution_id: str):
    solution_dir = SOLUTIONS_DIR / solution_id
    if not solution_dir.exists():
        raise HTTPException(status_code=404, detail='solution_id not found')
    files = [p.name for p in solution_dir.iterdir() if p.is_file()]
    return {'solution_id': solution_id, 'files': files}


@app.get('/solutions')
def list_solutions():
    ids = []
    for p in SOLUTIONS_DIR.iterdir():
        if p.is_dir():
            ids.append({'solution_id': p.name, 'files': [f.name for f in p.iterdir() if f.is_file()]})
    return {'solutions': ids}


# import requests
# url = "[https://my-codegen-api2.onrender.com/generate](https://my-codegen-api2.onrender.com/generate)"
# files = {"file": open("MyQ.zip", "rb")}
# r = requests.post(url, files=files, timeout=120)
# print(r.status_code)
# print(r.text)
# open("gen_response.json","wb").write(r.content)

# $Url = "[https://my-codegen-api2.onrender.com](https://my-codegen-api2.onrender.com)"
# curl.exe -s -X POST "$Url/generate" -F "file=@MyQ.zip" -o gen_response.json
# Get-Content gen_response.json -Raw | Out-File -FilePath gen_response_pretty.json


# r = requests.post("[https://my-codegen-api2.onrender.com/test](https://my-codegen-api2.onrender.com/test)", data={"solution_id": "<SOLUTION_ID>"}, files={"test_file": open("test_input.txt","rb")})
# open("test_response.json","wb").write(r.content)
# print(r.text)
# curl.exe -s -X POST "$Url/test" -F "solution_id=<SOLUTION_ID>" -F "test_file=@test_input.txt" -o test_response.json
# Get-Content test_response.json -Raw | ConvertFrom-Json | Format-List


# r = requests.get(f"[https://my-codegen-api2.onrender.com/download/](https://my-codegen-api2.onrender.com/download/)<SOLUTION_ID>/coding_solution.py", stream=True)
# open("coding_solution.py","wb").write(r.content)
# r = requests.get(f"[https://my-codegen-api2.onrender.com/download/](https://my-codegen-api2.onrender.com/download/)<SOLUTION_ID>/coding_solution.py", stream=True)
# open("coding_solution.py","wb").write(r.content)











# import requests
# url = "https://my-codegen-api2.onrender.com/generate"
# files = {"file": open("MyQ.zip", "rb")}
# r = requests.post(url, files=files, timeout=120)
# print(r.status_code)
# print(r.text)
# open("gen_response.json","wb").write(r.content)

# $Url = "https://my-codegen-api2.onrender.com"
# curl.exe -s -X POST "$Url/generate" -F "file=@MyQ.zip" -o gen_response.json
# Get-Content gen_response.json -Raw | Out-File -FilePath gen_response_pretty.json



# curl.exe -X POST "https://my-codegen-api2.onrender.com/generate" -F "file=@PP.zip" -o gen_response.json


# r = requests.post("https://my-codegen-api2.onrender.com/test", data={"solution_id": "<SOLUTION_ID>"}, files={"test_file": open("test_input.txt","rb")})
# open("test_response.json","wb").write(r.content)
# print(r.text)

# curl.exe -s -X POST "$Url/test" -F "solution_id=<SOLUTION_ID>" -F "test_file=@test_input.txt" -o test_response.json
# Get-Content test_response.json -Raw | ConvertFrom-Json | Format-List



# r = requests.get(f"https://my-codegen-api2.onrender.com/download/<SOLUTION_ID>/coding_solution.py", stream=True)
# open("coding_solution.py","wb").write(r.content)

# curl.exe -s -X GET "$Url/download/<SOLUTION_ID>/coding_solution.py" -o coding_solution.py




    #   Get-Content gen_response.json -Raw | ConvertFrom-Json | Format-List





