# save as convert_cookies.py in the project root
import json

with open("perplexity_cookies.json", "r") as f:
    raw = json.load(f)

# If it's Cookie-Editor format (list of objects)
if isinstance(raw, list):
    converted = {c["name"]: c["value"] for c in raw}
    with open("perplexity_cookies.json", "w") as f:
        json.dump(converted, f, indent=2)
    print("Converted! Saved as perplexity_cookies.json")
else:
    print("Already in correct format.")
