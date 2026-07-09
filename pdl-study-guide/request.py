import requests
import warnings

# Suppress the insecure request warning for clean output
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

url = "https://it.lab.nycu.edu.tw/"

try:
    # verify=False is the key to bypassing the revoked cert
    response = requests.get(url, verify=False)
    print(response.text)
except Exception as e:
    print(f"An error occurred: {e}")