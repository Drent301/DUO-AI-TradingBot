# core/gpt_reflector.py
import os
import json
import logging
import re
from typing import Optional, Dict, Any, List, Tuple, Union
import asyncio # Added for asyncio.to_thread

import requests
import dotenv # Nodig voor het laden van .env in de testfunctie

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o')

ENDPOINT = 'https://api.openai.com/v1/chat/completions'

class GPTReflector:
    """
    Voert GPT-reflectie uit door te communiceren met de OpenAI API.
    Vertaald van gptReflector.js.
    """

    def __init__(self):
        if not OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY is niet ingesteld. Kan geen verbinding maken met OpenAI API.")
            raise ValueError("OPENAI_API_KEY is niet ingesteld in de omgevingsvariabelen.")
        self.api_key = OPENAI_API_KEY
        self.model = OPENAI_MODEL
        logger.info(f"GPTReflector geïnitialiseerd met model: {self.model}")

    def _parse_ai_response(self, text: str) -> Dict[str, Any]:
        """
        Parseert de AI-uitvoer naar een reflectie-object.
        Vertaald van parseAIResponse in gptReflector.js.
        """
        confidence: Optional[float] = None
        intentie: Optional[str] = None
        emotie: Optional[str] = None
        reflectie: str = text.strip()

        conf_match = re.search(r"confidence[:=]\s*(\d+(\.\d+)?)", text, re.IGNORECASE)
        if conf_match:
            confidence = float(conf_match.group(1))

        intentie_match = re.search(r"intentie[:=]\s*(.*)", text, re.IGNORECASE)
        if intentie_match:
            intentie = intentie_match.group(1).strip()

        emotie_match = re.search(r"emotie[:=]\s*(.*)", text, re.IGNORECASE)
        if emotie_match:
            emotie = emotie_match.group(1).strip()

        return {
            "reflectie": reflectie,
            "confidence": confidence,
            "intentie": intentie,
            "emotie": emotie
        }

    async def ask_ai(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Voert GPT-reflectie uit met de gegeven prompt en context.
        Vertaald van askAI in gptReflector.js.
        """
        if context is None:
            context = {}

        messages = [
            {"role": "system", "content": "Je bent een reflectieve AI die beslissingen analyseert, evalueert en verbetert op basis van vertrouwen, emotie en intentie."},
            {"role": "user", "content": prompt}
        ]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.6,
            "max_tokens": 512
        }

        try:
            # Gebruik requests, wat sync is. Voor async applicaties kan httpx worden gebruikt.
            # Hier gebruiken we asyncio.to_thread om blocking requests in een async context te draaien.
            response = await asyncio.to_thread(requests.post, ENDPOINT, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            response_data = response.json()

            if not response_data or not response_data.get('choices'):
                logger.warning(f"Geen geldige respons ontvangen van OpenAI API: {response_data}")
                return {
                    "prompt": prompt,
                    "reflectie": "(leeg of ongeldig antwoord van GPT)",
                    "confidence": None,
                    "intentie": None,
                    "emotie": None,
                    "model": self.model,
                    "context": context,
                    "raw": response_data
                }

            reply_content = response_data['choices'][0]['message']['content']
            parsed_data = self._parse_ai_response(reply_content)

            return {
                "prompt": prompt,
                **parsed_data,
                "model": self.model,
                "raw": reply_content,
                "context": context
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"[GPT Reflector] ❌ Fout bij aanroep OpenAI API: {e}")
            return {
                "prompt": prompt,
                "reflectie": "(fout bij aanroep GPT)",
                "confidence": None,
                "intentie": None,
                "emotie": None,
                "model": self.model,
                "context": context,
                "error": str(e)
            }
        except json.JSONDecodeError as e:
            logger.error(f"[GPT Reflector] ❌ Fout bij parseren JSON respons: {e}. Raw response: {response.text if 'response' in locals() else 'N/A'}")
            return {
                "prompt": prompt,
                "reflectie": "(fout bij parseren respons)",
                "confidence": None,
                "intentie": None,
                "emotie": None,
                "model": self.model,
                "context": context,
                "error": str(e)
            }
        except Exception as e:
            logger.error(f"[GPT Reflector] ❌ Onverwachte fout: {e}")
            return {
                "prompt": prompt,
                "reflectie": "(onverwachte fout bij aanroep GPT)",
                "confidence": None,
                "intentie": None,
                "emotie": None,
                "model": self.model,
                "context": context,
                "error": str(e)
            }

# Voorbeeld van hoe je het zou kunnen gebruiken (voor testen)
if __name__ == "__main__":
    import asyncio
    dotenv.load_dotenv() # Zorg dat .env bestand geladen wordt

    async def run_test_gpt_reflector():
        try:
            reflector = GPTReflector()
            test_prompt = "Analyseer de recente marktvolatiliteit van ETH/USDT. Moet ik long of short gaan? Leg uit waarom. Format: confidence: 0.8, intentie: LONG, emotie: optimistisch. Reflectie: <jouw analyse>"
            response = await reflector.ask_ai(test_prompt)
            print("\n--- GPT Reflectie Resultaat ---")
            print(f"Prompt: {response['prompt']}")
            print(f"Reflectie: {response['reflectie']}")
            print(f"Confidence: {response['confidence']}")
            print(f"Intentie: {response['intentie']}")
            print(f"Emotie: {response['emotie']}")
            print(f"Model: {response['model']}")
            if 'error' in response:
                print(f"Foutdetails: {response['error']}")
            # print(f"Raw Response: {response['raw']}") # Uncomment voor debugging

        except ValueError as e:
            print(f"Fout: {e}")
        except Exception as e:
            print(f"Algemene fout: {e}")

    asyncio.run(run_test_gpt_reflector())
