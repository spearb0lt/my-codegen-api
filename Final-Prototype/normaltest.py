# import os, sys
# test_input_path="contestData\Final Product (Chapter 2)"
# if not os.path.exists(test_input_path):
#     print(f"Error: Required file '{test_input_path}' is missing.")
#     sys.exit(1)  # Exit with a non-zero status to indicate failure
# else:
#     print(f"Found required file '{test_input_path}'.")
# import json
# import requests, json, re, html
# from pathlib import Path

# # Load the JSON file
# with open("gen_response.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# # Print all top-level keys
# print("Top-level keys:")
# for key in data.keys():
#     print(key)
# aa=f.get("solution_id")
# print("Solution ID:", aa)


import requests, json, re, html
from pathlib import Path

url = "https://my-codegen-api2.onrender.com/generate"
Q_zip_path = "contestData\Final Product (Chapter 2).zip"
files = {"file": (Q_zip_path, open(Q_zip_path,"rb"), "application/zip")}
r = requests.post(url, files=files, timeout=600)
r.raise_for_status()
j = r.json()
Path("gen_response.json").write_text(json.dumps(j, indent=2))
print(j)

# Save code text if present
code_text = j.get("solution")
if code_text:
    # remove code fences if present
    m = re.search(r"```(?:python)?\s*\n([\s\S]*?)```", code_text, re.IGNORECASE)
    code = m.group(1) if m else code_text
    code = html.unescape(code).replace("\r\n","\n")
    outp = Path("coding_solution_gen.py")
    outp.write_text(code, encoding="utf-8")
    print("Saved coding_solution_gen.py")
else:
    print("No solution text returned. Use download endpoint with solution_id.")

#DOWNLOAD

solution_id = j.get("solution_id")
print( "Solution ID:", solution_id )
solution_path = j.get("solution_path")
print( "Solution Path:", solution_path )
# "solution_path": "solutions/7976324c-bc1e-4331-8a6f-2b2e24ac57ab/coding_solution.py",
r = requests.get("https://my-codegen-api2.onrender.com/download/{solution_id}/coding_solution.py")
r.raise_for_status()
open("coding_solution_downloaded.py","wb").write(r.content)
