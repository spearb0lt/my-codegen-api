import os
import sys
import subprocess
from pathlib import Path

# --- Configuration (Edit these values) ---

# IMPORTANT: Set your Google API Key as an environment variable for security.
API_KEY = os.environ.get("GOOGLE_API_KEY", "useyourownapikey")

# The model to use for code generation and fixing.
MODEL_NAME = "gemini-2.5-pro"

# The solution file to test and refine.
SOLUTION_FILENAME = "solution.py"

# The file where the final output will be stored.
FINAL_OUTPUT_FILENAME = "output.txt"

# Original problem statement file (needed for re-prompting).
STATEMENT_FILE_PATH = "A_Question\statement.txt"

# --- Advanced Configuration ---
MAX_FIX_ATTEMPTS = 5
EXECUTION_TIMEOUT = 20 # Allow more time for larger test files

# --- End of Configuration ---

def install_dependencies():
    """Installs google-generativeai if not present."""
    try:
        import google.generativeai
    except ImportError:
        print("Installing google-generativeai...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "google-generativeai"])
        except Exception as e:
            print(f"Error installing dependency: {e}\nPlease install it manually.")
            sys.exit(1)

def extract_python_code(response_text: str) -> str | None:
    """Extracts Python code from a markdown code block."""
    if "```python" in response_text:
        return response_text.split("```python")[1].split("```")[0].strip()
    elif "```" in response_text:
        code_block = response_text.split("```")[1]
        if code_block.lower().startswith('python'):
            return '\n'.join(code_block.split('\n')[1:]).strip()
        return code_block.strip()
    return None

def run_solution(code_path: Path, input_str: str, timeout: int) -> dict:
    """Runs a Python file in a separate process."""
    try:
        process = subprocess.Popen(
            [sys.executable, str(code_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        stdout, stderr = process.communicate(input=input_str, timeout=timeout)
        return {"stdout": stdout, "stderr": stderr, "timed_out": False}
    except subprocess.TimeoutExpired:
        process.kill()
        return {"stdout": "", "stderr": f"Execution timed out after {timeout}s.", "timed_out": True}
    except Exception as e:
        return {"stdout": "", "stderr": f"An unexpected error occurred: {e}", "timed_out": False}

def main():
    """Main function to test and refine the solution."""
    install_dependencies()
    import google.generativeai as genai

    if not API_KEY:
        print("ERROR: GOOGLE_API_KEY environment variable not set.")
        sys.exit(1)

    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
    
    solution_path = Path(SOLUTION_FILENAME)
    if not solution_path.exists():
        print(f"Error: Solution file '{SOLUTION_FILENAME}' not found.")
        print("Please run 'code_generator_v2.py' first to generate it.")
        sys.exit(1)

    try:
        statement = Path(STATEMENT_FILE_PATH).read_text(encoding='utf-8')
    except FileNotFoundError:
        print(f"Error: Statement file '{STATEMENT_FILE_PATH}' not found. It's needed for fixing errors.")
        sys.exit(1)

    test_case_path_str = input("Please enter the full path to your test case input file: ")
    test_case_path = Path(test_case_path_str.strip())

    if not test_case_path.exists():
        print(f"Error: The test case file was not found at '{test_case_path}'")
        sys.exit(1)
        
    test_input = test_case_path.read_text(encoding='utf-8')

    for attempt in range(1, MAX_FIX_ATTEMPTS + 1):
        print(f"\n{'='*20} TESTING CYCLE {attempt}/{MAX_FIX_ATTEMPTS} {'='*20}")
        print(f"Running '{solution_path}' against '{test_case_path.name}'...")
        
        current_code = solution_path.read_text(encoding='utf-8')
        result = run_solution(solution_path, test_input, timeout=EXECUTION_TIMEOUT)

        # Check for errors
        if result['stderr'] or result['timed_out']:
            print("Status: FAILED")
            error_reason = result['stderr'] if not result['timed_out'] else "Code timed out."
            print(f"Error Detected:\n{error_reason}")
            
            # Re-prompt the LLM to fix the code
            print("\nAsking Gemini for a fix...")
            fix_prompt = f"""
            The following Python code failed to run correctly against a set of test cases.

            **Original Problem Statement:**
            {statement}

            **Your Failing Code:**
            ```python
            {current_code}
            ```

            **Failing Test Case Input:**
            ```
            {test_input[:3000]} 
            ```
            (Note: Input may be truncated for brevity)

            **Error Message:**
            ```
            {error_reason}
            ```

            Please analyze the code and the error, then provide a fully corrected version of the code.
            Provide only the complete, corrected Python code in a single markdown block.
            """
            
            try:
                response = model.generate_content(fix_prompt)
                fixed_code = extract_python_code(response.text)

                if fixed_code:
                    print("Received a potential fix. Overwriting '{solution_path}' and re-testing.")
                    solution_path.write_text(fixed_code, encoding='utf-8')
                else:
                    print("Warning: Could not extract a valid code fix from the response. Re-testing with old code.")

            except Exception as e:
                print(f"An error occurred with the Gemini API during fix attempt: {e}")
                
        else:
            # If there are no errors, the code is considered correct
            print("\n" + "="*60)
            print("✅ SUCCESS: The solution ran without errors on the full test suite!")
            print("="*60)
            
            output_path = Path(FINAL_OUTPUT_FILENAME)
            output_path.write_text(result['stdout'], encoding='utf-8')
            print(f"Final output saved to: {output_path.resolve()}")
            return

    print(f"\n❌ FAILURE: Could not produce a working solution after {MAX_FIX_ATTEMPTS} refinement cycles.")
    print(f"The last failing version of the code is still in '{solution_path}'.")

if __name__ == "__main__":
    main()
