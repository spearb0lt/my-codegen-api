#A
#curl.exe -s -X POST "https://my-codegen-api2.onrender.com/test" -F "solution_id=THE_SOLUTION_ID" -F "test_file=@test_input.txt" | Tee-Object -FilePath test_response.json
#Get-Content test_response.json -Raw | ConvertFrom-Json | Format-List

import requests, json
# files = {"test_file": ("test_input.txt", open("test_input.txt","rb"), "text/plain")}
# data = {"solution_id": "7976324c-bc1e-4331-8a6f-2b2e24ac57ab"}
# r = requests.post("https://my-codegen-api2.onrender.com/test", data=data, files=files, timeout=120)
# r.raise_for_status()
# j = r.json()
# print(json.dumps(j, indent=2))
# # Save test output locally
# if j.get("test_stdout"):
#     open("test_output_at.txt","w",encoding="utf-8").write(j["test_stdout"])

# # B — Test using local generated .py + problem ZIP + test input in a single request
# # curl.exe -s -X POST "https://my-codegen-api2.onrender.com/test" `
# #   -F "solution_file=@coding_solution.py" `
# #   -F "problem_zip=@MyQ.zip" `
# #   -F "test_file=@test_input.txt" | Tee-Object -FilePath test_response.json
# # Get-Content test_response.json -Raw | ConvertFrom-Json | Format-List
# files = {
#   "solution_file": ("coding_solution_gen.py", open("coding_solution_gen.py","rb"), "text/x-python"),
#   "problem_zip": ("MyQ.zip", open("MyQ.zip","rb"), "application/zip"),
#   "test_file": ("test_input.txt", open("test_input.txt","rb"), "text/plain"),
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


# C — Test using solution text (send the code as form field) + test file
# curl.exe -s -X POST "https://my-codegen-api2.onrender.com/test" -F "solution=@coding_solution.py" -F "test_file=@test_input.txt" | Tee-Object -FilePath test_response.json
# Get-Content test_response.json -Raw | ConvertFrom-Json | Format-List



code_text = open("coding_solution_gen.py","r",encoding="utf-8").read()
files = {"test_file": ("test_input.txt", open("test_input.txt","rb"), "text/plain")}
data = {"solution": code_text}
r = requests.post("https://my-codegen-api2.onrender.com/test", data=data, files=files, timeout=120)
j = r.json()
print(j)
# Save output
open("test_output_ct.txt","w",encoding="utf-8").write(j.get("test_stdout",""))
