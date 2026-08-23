from huggingface_hub import HfApi
api = HfApi()
files = api.list_repo_files("liamdugan/raid", repo_type="dataset")
print("TOTAL FILES:", len(files))
for f in files:
    print(f)
