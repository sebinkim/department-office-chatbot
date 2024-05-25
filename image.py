import requests
import json
from requests_toolbelt import MultipartEncoder

URL = "https://inference.friendli.ai/stable-diffusion-v1-5/v1/text-to-image"

PROMPT_LIST = [
    "Draw a university office staff member. She is beaming with joy, smiling widely, and looks very happy because she hasn't received any inquiries so far today. She is asian, long-haired, 30 years old, black hair. She looks nice, kind, and friendly. Intricate, elegant, highly detailed, centered, digital painting, ArtStation, concept art, smooth, sharp focus, illustration, Artgerm.",
    "Draw a university office staff member. She is smiling pleasantly and looks happy because she hasn't received many inquiries so far today. She is asian, long-haired, 30 years old, black hair. She looks nice, kind, and friendly. Intricate, elegant, highly detailed, centered, digital painting, ArtStation, concept art, smooth, sharp focus, illustration, Artgerm.",
    "Draw a university office staff member. She has a calm and composed expression, looking neutral as she handles the regular inquiries today. She is asian, long-haired, 30 years old, black hair. She looks nice, kind, and friendly. Intricate, elegant, highly detailed, centered, digital painting, ArtStation, concept art, smooth, sharp focus, illustration, Artgerm.",
    "Draw a university office staff member. She looks slightly irritated and frustrated because she has received a lot of inquiries so far today. She is asian, long-haired, 30 years old, black hair. She looks nice, kind, and friendly. Intricate, elegant, highly detailed, centered, digital painting, ArtStation, concept art, smooth, sharp focus, illustration, Artgerm.",
    "Draw a university office staff member. She is very angry because she received too many inquiries so far today. She is asian, long-haired, 30 years old, black hair. She looks nice, kind, and friendly. Intricate, elegant, highly detailed, centered, digital painting, ArtStation, concept art, smooth, sharp focus, illustration, Artgerm."
]

def generate_image(n: int):
    prompt = PROMPT_LIST[min(n, 4)]
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

    obj = json.loads(response.content)

    return obj["data"][0]["url"]