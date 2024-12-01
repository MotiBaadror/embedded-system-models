import requests

# URL of the Flask app endpoint
url = "http://localhost:8000/predict"

# Data to send in the POST request
data = {
    'example_batch':{'text':['Your bank account is locked.Click here to unlock. ']}
}

# Send the POST request
response = requests.post(url, json=data)

# Print the response from the server
print(f"Status Code: {response.status_code}")
print(f"Response JSON: {response.text}")
