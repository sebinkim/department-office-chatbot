import requests
from requests_toolbelt import MultipartEncoder

URL = "https://inference.friendli.ai/stable-diffusion-v1-5/v1/text-to-image"

def generate_image(prompt: str):
    payload = {
        'model': 'stable-diffusion-v1-5',
        'prompt': prompt,
        "negative_prompt": "black",
        "response_format": "url",
        "num_steps": "10",
    }
    m = MultipartEncoder(fields=payload)
    headers = {
    'Content-Type': m.content_type,
    'Accept': 'application/json',
    'Authorization': 'Bearer flp_lYqxZZHbY9f2DJp6B7Fn1Y5wTG2OyTHhxhJuMecI4az872',
    }

    response = requests.request("POST", URL, headers=headers, data=m)

    return response.content