import requests
from requests_toolbelt import MultipartEncoder

url = "https://inference.friendli.ai/stable-diffusion-v1-5/v1/text-to-image"

prompt_list = [
    "Draw a university office staff member. She is beaming with joy, smiling widely, and looks very happy because she hasn't received any inquiries so far today. She is asian, long-haired, 30 years old, black hair. She looks nice, kind, and friendly. Intricate, elegant, highly detailed, centered, digital painting, ArtStation, concept art, smooth, sharp focus, illustration, Artgerm.",
    "Draw a university office staff member. She is smiling pleasantly and looks happy because she hasn't received many inquiries so far today. She is asian, long-haired, 30 years old, black hair. She looks nice, kind, and friendly. Intricate, elegant, highly detailed, centered, digital painting, ArtStation, concept art, smooth, sharp focus, illustration, Artgerm.",
    "Draw a university office staff member. She has a calm and composed expression, looking neutral as she handles the regular inquiries today. She is asian, long-haired, 30 years old, black hair. She looks nice, kind, and friendly. Intricate, elegant, highly detailed, centered, digital painting, ArtStation, concept art, smooth, sharp focus, illustration, Artgerm.",
    "Draw a university office staff member. She looks slightly irritated and frustrated because she has received a lot of inquiries so far today. She is asian, long-haired, 30 years old, black hair. She looks nice, kind, and friendly. Intricate, elegant, highly detailed, centered, digital painting, ArtStation, concept art, smooth, sharp focus, illustration, Artgerm.",
    "Draw a university office staff member. She is very angry because she received too many inquiries so far today. She is asian, long-haired, 30 years old, black hair. She looks nice, kind, and friendly. Intricate, elegant, highly detailed, centered, digital painting, ArtStation, concept art, smooth, sharp focus, illustration, Artgerm."
]

payload={
    'model': 'stable-diffusion-v1-5',
    'prompt': "Draw angry chatbot",
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

response = requests.request("POST", url, headers=headers, data=m)

print(response.status_code)
print(response.content)