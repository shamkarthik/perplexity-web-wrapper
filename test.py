from lib.perplexity import Client
import json

with open("perplexity_cookies.json") as f:
    cookies = json.load(f)

client = Client(cookies)
result = client.search("What is Perplexity AI?", mode="auto")
print(result)