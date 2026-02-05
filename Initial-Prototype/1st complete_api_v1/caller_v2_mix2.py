#GENERATE
ff=0



import requests, json, re, html, os, sys
from pathlib import Path

hl = requests.get("https://my-codegen-api2.onrender.com/health")
print(hl.json())
Q_zip_path = "MyQ.zip"

if (ff==0):
    url = "https://my-codegen-api2.onrender.com/generate"
    Q_zip_path = "MyQ.zip"
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
    solution_path = j.get("solution_path")
    # "solution_path": "solutions/7976324c-bc1e-4331-8a6f-2b2e24ac57ab/coding_solution.py",
    print(f"solution_id: {solution_id}")
    print(f"solution_path: {solution_path}")    
    # "solution_path": "solutions/7976324c-bc1e-4331-8a6f-2b2e24ac57ab/coding_solution.py",
    durl=f"https://my-codegen-api2.onrender.com/download/{solution_id}/coding_solution.py"
    r = requests.get(durl)
    r.raise_for_status()
    open("coding_solution_downloaded.py","wb").write(r.content)


#TEST
# ff=0
if ff!=0:
    with open("gen_response.json", 'r') as file:
        data = json.load(file)
        solution_id= data['solution_id']

    test_input_path = "test_input.txt"
    if not os.path.exists(test_input_path):
        print(f"Error: Required file '{test_input_path}' is missing.")
        sys.exit(1)  # Exit with a non-zero status to indicate failure

    #A
    files = {"test_file": (test_input_path, open(test_input_path,"rb"), "text/plain")}
    data = {"solution_id": solution_id}
    r = requests.post("https://my-codegen-api2.onrender.com/test", data=data, files=files, timeout=120)
    r.raise_for_status()
    j = r.json()
    print(json.dumps(j, indent=2))
    # Save test output locally
    if j.get("test_stdout"):
        open("test_output_at.txt","w",encoding="utf-8").write(j["test_stdout"])


    # #B
    # files = {
    # "solution_file": ("coding_solution_gen.py", open("coding_solution_gen.py","rb"), "text/x-python"),
    # "problem_zip": (Q_zip_path, open(Q_zip_path,"rb"), "application/zip"),
    # "test_file": (test_input_path, open(test_input_path,"rb"), "text/plain"),
    # }
    # r = requests.post("https://my-codegen-api2.onrender.com/test", files=files, timeout=120)
    # r.raise_for_status()
    # j = r.json()
    # print(j)
    # # Save solution and test output locally
    # if j.get("solution"):
    #     open("coding_solution_saved.py","w",encoding="utf-8").write(j["solution"])
    # if j.get("test_stdout"):
    #     open("test_output_bt.txt","w",encoding="utf-8").write(j["test_stdout"])


    # # C

    # code_text = open("coding_solution_gen.py","r",encoding="utf-8").read()
    # files = {"test_file": (test_input_path, open(test_input_path,"rb"), "text/plain")}
    # data = {"solution": code_text}
    # r = requests.post("https://my-codegen-api2.onrender.com/test", data=data, files=files, timeout=120)
    # j = r.json()
    # print(j)
    # # Save output
    # open("test_output_ct.txt","w",encoding="utf-8").write(j.get("test_stdout",""))
