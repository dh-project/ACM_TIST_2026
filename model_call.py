import json
import requests

# Print:
from colorama import Fore, Style

# Parameters:
EFFORT = 'low'
MAX_TOKENS = 10000
TEMPERATURE = 0
STREAM = False

# API Keys:
OPENROUTER_API_KEY = 'API KEY'


# OpenRouter:
def generation_openrouter(prompt, model, provider, reasoning_parameter):
    
    reasoning_config = {
        "exclude": True
    }

    if reasoning_parameter == 'Effort':
        reasoning_config["effort"] = EFFORT
    else:
        reasoning_config["max_tokens"] = MAX_TOKENS

    try:

        response = requests.post(

            url="https://openrouter.ai/api/v1/chat/completions",

            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },

            data=json.dumps({
                "model": model, 
                "provider": {
                    "order": [provider],
                    "allow_fallbacks": False
                },    
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "reasoning": reasoning_config,
                "usage": {
                    "include": True
                },
                "temperature": TEMPERATURE,
                "stream": STREAM
            })

        )

        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'], response.json()['usage']['cost']

    except requests.exceptions.HTTPError as http_error:
        print(Fore.YELLOW, http_error, Style.RESET_ALL, sep='')
        print(response.json().get('error').get('message'))
        return None
    
    except json.JSONDecodeError as json_error:
        print(Fore.YELLOW, json_error, Style.RESET_ALL, sep='')
        return None
