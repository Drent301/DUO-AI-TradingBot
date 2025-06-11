# core/grok_reflector.py
import os
import json
import logging
import re
from typing import Optional, Dict, Any
import asyncio # Nodig voor de async main-test

import requests
import dotenv # Nodig voor het laden van .env in de testfunctie

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

GROK_API_KEY = os.getenv('GROK_API_KEY')
GROK_MODEL = os.getenv('GROK_MODEL', 'grok-1')
# Corrected ENDPOINT definition to remove markdown link
ENDPOINT = os.getenv('GROK_API_URL', 'https://api.x.ai/v1/chat/completions') # Voorbeeldendpoint

class GrokReflector:
    """
    Voert Grok-reflectie uit door te communiceren met de Grok API.
    Vertaald van grokReflector.js.
    """

    def __init__(self):
        if not GROK_API_KEY:
            logger.error("GROK_API_KEY is niet ingesteld. Kan geen verbinding maken met Grok API.")
            raise ValueError("GROK_API_KEY is niet ingesteld in de omgevingsvariabelen.")
        self.api_key = GROK_API_KEY
        self.model = GROK_MODEL
        self.endpoint = ENDPOINT
        logger.info(f"GrokReflector geïnitialiseerd met model: {self.model}, endpoint: {self.endpoint}")

    def _parse_grok_response(self, text: str) -> Dict[str, Any]:
        """
        Parseert de AI-uitvoer van Grok naar een reflectie-object.
        Vertaald van parseGrokResponse in grokReflector.js.
        """
        confidence: Optional[float] = None
        intentie: Optional[str] = None
        emotie: Optional[Optional[str]] = None # Dubbele Optional door type hinting voor None
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

    async def ask_grok(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Voert Grok-reflectie uit met de gegeven prompt en context.
        Vertaald van askGrok in grokReflector.js.
        """
        if context is None:
            context = {}

        messages = [
            {"role": "system", "content": "Je bent een onafhankelijke AI (Grok) die beslissingen analyseert, intuïtieve reflectie toepast en zoekt naar strategische optimalisatie."},
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
            response = await asyncio.to_thread(requests.post, self.endpoint, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            response_data = response.json()

            if not response_data or not response_data.get('choices'):
                logger.warning(f"Geen geldige respons ontvangen van Grok API: {response_data}")
                return {
                    "prompt": prompt,
                    "reflectie": "(leeg of ongeldig antwoord van Grok)",
                    "confidence": None,
                    "intentie": None,
                    "emotie": None,
                    "model": self.model,
                    "context": context,
                    "raw": response_data
                }

            reply_content = response_data['choices'][0]['message']['content']
            parsed_data = self._parse_grok_response(reply_content)

            return {
                "prompt": prompt,
                **parsed_data,
                "model": self.model,
                "raw": reply_content,
                "context": context
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"[Grok Reflector] ❌ Fout bij aanroep Grok API: {e}")
            return {
                "prompt": prompt,
                "reflectie": "(fout bij aanroep Grok)",
                "confidence": None,
                "intentie": None,
                "emotie": None,
                "model": self.model,
                "context": context,
                "error": str(e)
            }
        except json.JSONDecodeError as e:
            logger.error(f"[Grok Reflector] ❌ Fout bij parseren JSON respons: {e}. Raw response: {response.text if 'response' in locals() else 'N/A'}")
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
            logger.error(f"[Grok Reflector] ❌ Onverwachte fout: {e}")
            return {
                "prompt": prompt,
                "reflectie": "(onverwachte fout bij aanroep Grok)",
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
    dotenv.load_dotenv()

    async def run_test_grok_reflector():
        try:
            grok_reflector = GrokReflector()
            test_prompt = "Wat is het sentiment over de recente ontwikkelingen in de crypto markt volgens sociale media? Geef een intuïtieve inschatting. Format: confidence: 0.7, intentie: NEUTRAAL, emotie: voorzichtig. Reflectie: <jouw analyse>"
            response = await grok_reflector.ask_grok(test_prompt)
            print("\n--- Grok Reflectie Resultaat ---")
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

    asyncio.run(run_test_grok_reflector())
