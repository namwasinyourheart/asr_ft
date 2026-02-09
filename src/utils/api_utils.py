
import requests
from consts import TEXT_POSTPROCESSING_URL_ENDPOINT


def postprocess_text_via_api(text):
    url = TEXT_POSTPROCESSING_URL_ENDPOINT
    headers = {
        "accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"text": text}
    
    response = requests.post(url, headers=headers, data=data)
    response.raise_for_status()  # Raises an error if the request failed
    postprocessed_text = response.json()['text']
    return postprocessed_text