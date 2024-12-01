import requests

# URL of the Flask app endpoint
url = "http://localhost:8000/predict"

# Data to send in the POST request
# data =
    # 'example_batch':{'text':['Your bank account is locked.Click here to unlock. ']}
data = {'text': 'WINNER!! As a valued network customer you have been selected to receivea '}


# Send the POST request
response = requests.post(url, json=data)

# Print the response from the server
print(f"Status Code: {response.status_code}")
print(f"Response JSON: {response.text}")
# curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"text": "WINNER!! As a valued network customer you have been selected to receivea "}'

