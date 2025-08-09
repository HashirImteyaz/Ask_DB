import requests
import json

# Test the API response structure
response = requests.post("http://localhost:8000/chat", json={
    "query": "What are the names of recipes?",
    "session_id": "debug_test"
})

print("Status Code:", response.status_code)
print("Response Headers:", dict(response.headers))
print("Response Content:")
print(json.dumps(response.json(), indent=2))
