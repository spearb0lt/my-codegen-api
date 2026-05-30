import os
import sys
import zipfile
import shutil
import subprocess
import difflib
from pathlib import Path
import subprocess
# --- Configuration (Edit these values) ---

# IMPORTANT: Set your Google API Key as an environment variable for security.
# In Linux/macOS: export GOOGLE_API_KEY="YOUR_API_KEY"
# In Windows PowerShell: $env:GOOGLE_API_KEY="YOUR_API_KEY"
# If you must, you can paste it here directly, but it's not recommended.
API_KEY = os.environ.get("GOOGLE_API_KEY", "useyourownapikey") # <<< CHANGE THIS TO YOUR API KEY OR SET THE ENV VARIABLE

# The model to use for code generation.
# gemini-1.5-pro is recommended for high-quality code.
# gemini-1.5-flash is faster and more cost-effective.
MODEL_NAME = "gemini-2.5-pro"

# The path to the zip file containing the problem statement, sample_in.txt, and sample_out.txt.
# Example: "/path/to/your/problem.zip" or "C:\\Users\\YourUser\\Downloads\\Zone In.zip"
ZIP_FILE_PATH = "Warm Up.zip" # <<< CHANGE THIS TO YOUR ZIP FILE PATH

# The name for the final output solution file.
OUTPUT_FILENAME = r"C:\Users\ASUS\Downloads\CHATBOT\solution.py" # <<< CHANGE THIS TO YOUR DESIRED OUTPUT FILE PATH

# --- Advanced Configuration ---

# Maximum number of attempts to generate and fix the code.
MAX_RETRIES = 5

# Timeout in seconds for running the generated code against a test case.
EXECUTION_TIMEOUT = 10

# --- End of Configuration ---

# Helper function to install google-generativeai if it's not already installed.
def install_dependencies():
    """Checks for and installs the necessary google-generativeai library."""
    try:
        import google.generativeai
        print("google-generativeai is already installed.")
    except ImportError:
        print("google-generativeai not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "google-generativeai"])
            print("Installation successful.")
        except Exception as e:
            print(f"Error installing google-generativeai: {e}")
            print("Please install it manually using: pip install google-generativeai")
            sys.exit(1)

def unpack_problem_files(zip_path: Path) -> dict:
    """
    Unzips the problem file and finds the statement, sample input, and sample output.

    Args:
        zip_path: Path to the .zip file.

    Returns:
        A dictionary with paths to the statement, input, and output files.
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"The file {zip_path} was not found.")

    # Create a unique directory for unpacked files
    unpacked_dir = r"C:\Users\ASUS\Downloads\CHATBOT\A_Question"
    # if unpacked_dir.exists():
    #     shutil.rmtree(unpacked_dir)
    # unpacked_dir.mkdir()

    print(f"Unpacking {zip_path} to {unpacked_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # zf.extractall(unpacked_dir)
        for member in zf.infolist():
    # Get the base filename only (ignore internal folders)
            filename = Path(member.filename).name
            target_path = Path(unpacked_dir) / filename

            # Skip directories
            if member.is_dir():
                continue

            # Extract file to target path
            with zf.open(member) as source, open(target_path, 'wb') as target:
                shutil.copyfileobj(source, target)

    # Find the required files
    files = {}
    file_keys = {'statement': ['statement', 'problem'],
                 'sample_in': ['sample_in', 'sample.in', 'input'],
                 'sample_out': ['sample_out', 'sample.out', 'output']}

    for root, _, filenames in os.walk(unpacked_dir):
        for filename in filenames:
            for key, patterns in file_keys.items():
                if any(p in filename.lower() for p in patterns):
                    files[key] = Path(root) / filename
                    break # Move to the next filename

    if not all(k in files for k in ['sample_in', 'sample_out', 'statement']):
        raise FileNotFoundError(
            f"Could not find all required files (statement, sample_in, sample_out) in {unpacked_dir}. Found: {files.keys()}"
        )

    print("Found problem files:")
    for key, path in files.items():
        print(f"  - {key.replace('_', ' ').title()}: {path}")

    return files

def extract_python_code(response_text: str) -> str | None:
    """Extracts Python code from a markdown code block."""
    # Look for the ```python ... ``` block
    if "```python" in response_text:
        code_block = response_text.split("```python")[1]
        return code_block.split("```")[0].strip()
    # Fallback for a generic ``` ... ``` block
    elif "```" in response_text:
        code_block = response_text.split("```")[1]
        # Remove potential language name on the first line
        if code_block.lower().startswith('python'):
            code_block = '\n'.join(code_block.split('\n')[1:])
        return code_block.strip()
    return None

def run_code(code_str: str, input_str: str, timeout: int) -> dict:
    """
    Runs the provided code string in a separate process with a timeout.

    Returns:
        A dictionary containing stdout, stderr, and a timed_out flag.
    """
    try:
        # Execute the code_str directly using the -c flag
        process = subprocess.Popen(
            [sys.executable, "-c", code_str], # <--- THE FIX IS HERE
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
        return {"stdout": "", "stderr": f"Execution timed out after {timeout} seconds.", "timed_out": True}
    except Exception as e:
        return {"stdout": "", "stderr": f"An unexpected error occurred during execution: {e}", "timed_out": False}

def normalize_text(text: str) -> str:
    """Normalizes text by stripping whitespace from each line and the entire string."""
    return "\n".join(line.strip() for line in text.strip().splitlines())

def main():
    """Main function to run the code generation and testing process."""
    install_dependencies()
    import google.generativeai as genai

    if not API_KEY:
        print("ERROR: GOOGLE_API_KEY environment variable not set.")
        print("Please set your API key to run the script.")
        sys.exit(1)

    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)

    try:
        problem_files = unpack_problem_files(Path(ZIP_FILE_PATH))
        statement = problem_files['statement'].read_text(encoding='utf-8')
        sample_in = problem_files['sample_in'].read_text(encoding='utf-8')
        sample_out = problem_files['sample_out'].read_text(encoding='utf-8')
    except Exception as e:
        print(f"\nError processing files: {e}")
        sys.exit(1)

    generated_code = ""
    last_error = ""
    feedback_prompt = ""

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n{'='*20} ATTEMPT {attempt}/{MAX_RETRIES} {'='*20}")

        if attempt == 1:
            prompt = f"""
            You are an expert competitive programmer. Your task is to write a Python 3 solution for the following problem.

            Read the problem statement carefully and write a complete, correct, and efficient program that reads from standard input and prints to standard output.

            **Problem Statement:**
            {statement}

            **Sample Input:**
            ```
            {sample_in}
            ```

            **Sample Output:**
            ```
            {sample_out}
            ```

            Please provide only the complete Python code in a single markdown block. Do not include any explanations, comments, or introductory text outside of the code block.
            """
        else:
            prompt = feedback_prompt

        print("Generating code with Gemini...")
        try:
            response = model.generate_content(prompt)
            generated_code_candidate = extract_python_code(response.text)

            if not generated_code_candidate:
                print("Error: Could not extract Python code from the model's response.")
                print("Response Text:", response.text)
                last_error = "Failed to extract code from response."
                continue
            
            generated_code = generated_code_candidate

        except Exception as e:
            print(f"An error occurred while calling the Gemini API: {e}")
            if "API key not valid" in str(e):
                print("Please check if your GOOGLE_API_KEY is correct.")
            sys.exit(1)

        print("--- Generated Code ---")
        print(generated_code)
        print("----------------------")

        print("Testing the generated code...")
        result = run_code(generated_code, sample_in, timeout=EXECUTION_TIMEOUT)
        
        actual_output_normalized = normalize_text(result['stdout'])
        expected_output_normalized = normalize_text(sample_out)

        if result['timed_out']:
            print("Status: FAILED (Timeout)")
            last_error = result['stderr']
        elif result['stderr']:
            print("Status: FAILED (Runtime Error)")
            print("--- Stderr ---")
            print(result['stderr'])
            print("--------------")
            last_error = result['stderr']
        elif actual_output_normalized == expected_output_normalized:
            # ... inside the success block in main() ...
            print("✅ SUCCESS: The generated code passed the sample test case!")
            print("="*50)
            
            final_path = Path.cwd() / OUTPUT_FILENAME
            final_path.write_text(generated_code, encoding='utf-8')
            print(f"Solution saved to: {final_path.resolve()}")
            
            # --- ADD THIS CODE ---
            print("\n--- Automatically Starting the Test and Refinement Script ---")
            subprocess.run([sys.executable, "gem1_61.py"])
            # -------------------
            
            return # Exit successfully
        else:
            print("Status: FAILED (Incorrect Output)")
            last_error = "The output did not match the expected sample output."
            
            # Generate a diff to show the model
            diff = difflib.unified_diff(
                expected_output_normalized.splitlines(keepends=True),
                actual_output_normalized.splitlines(keepends=True),
                fromfile='Expected Output',
                tofile='Actual Output',
            )
            print("--- Output Difference ---")
            print("".join(diff))
            print("-------------------------")

        # Prepare feedback for the next attempt
        feedback_prompt = f"""
        The previous code you submitted was incorrect.

        **Problem Statement:**
        {statement}

        **Your Incorrect Code:**
        ```python
        {generated_code}
        ```

        **Reason for Failure:**
        {last_error}

        **Sample Input:**
        ```
        {sample_in}
        ```

        **Expected Sample Output:**
        ```
        {sample_out}
        ```

        **Actual Output from Your Code:**
        ```
        {result['stdout']}
        ```
        
        Please analyze the error and provide a corrected version. Provide only the complete, corrected Python code in a single markdown block.
        """

    print("\n" + "="*50)
    print(f"❌ FAILURE: Could not generate a correct solution after {MAX_RETRIES} attempts.")
    print("="*50)
    if generated_code:
        final_path = Path.cwd() / f"failed_{OUTPUT_FILENAME}"
        final_path.write_text(generated_code, encoding='utf-8')
        print(f"The last generated (incorrect) code has been saved to: {final_path.resolve()}")

if __name__ == "__main__":
    main()

