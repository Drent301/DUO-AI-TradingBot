# core/grok_sentiment_fetcher.py
import os
import json
import logging
from typing import Dict, Any, List, Optional
import requests
from datetime import datetime, timedelta
import asyncio # Nodig voor de async main-test
import dotenv # Nodig voor het laden van .env in de testfunctie

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

GROK_API_KEY = os.getenv('GROK_API_KEY')
LIVE_SEARCH_API_URL = os.getenv('GROK_LIVE_SEARCH_API_URL', 'https://api.x.ai/v1/live-search') # TODO: Update with official endpoint when available. This is a hypothetical endpoint.

# Padconfiguratie voor cache
# __file__ is de huidige bestandsnaam. os.path.dirname haalt de map op.
# os.path.abspath maakt er een absoluut pad van.
# '..' gaat één map omhoog (van 'core' naar de projectroot).
# os.path.join voegt dan 'data', 'cache' en 'live_search_cache.json' toe.
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'cache')
CACHE_FILE = os.path.join(CACHE_DIR, 'live_search_cache.json')
CACHE_VALID_MINUTES = 5

os.makedirs(CACHE_DIR, exist_ok=True) # Zorg dat de cache map bestaat

class GrokSentimentFetcher:
    """
    Haalt real-time sentiment- en nieuwsdata op via een hypothetische Grok Live Search API.
    Vertaald en geoptimaliseerd van functionaliteit genoemd in reflectieLus.js.
    Focus op de 5 meest recente en relevante berichten per token.
    """

    def __init__(self):
        if not GROK_API_KEY:
            logger.error("GROK_API_KEY is niet ingesteld. Kan geen verbinding maken met Grok Live Search API.")
            raise ValueError("GROK_API_KEY is niet ingesteld in de omgevingsvariabelen.")
        self.api_key = GROK_API_KEY
        self.live_search_url = LIVE_SEARCH_API_URL
        logger.info(f"GrokSentimentFetcher geïnitialiseerd. Live Search URL: {self.live_search_url}")

    def _load_cache(self) -> Dict[str, Any]:
        """Laad de cache vanuit een JSON-bestand."""
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_cache(self, cache_data: Dict[str, Any]):
        """Sla de cache op naar een JSON-bestand."""
        try:
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
        except IOError as e:
            logger.error(f"Fout bij opslaan cache bestand: {e}")

    async def fetch_live_search_data(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Haalt real-time data op via de hypothetische Live Search API.
        Optimaliseert voor de 5 meest recente berichten die direct over de coin gaan.
        Implementeert caching om API-limieten te respecteren.
        """
        # Controleer cache
        cache = self._load_cache()
        cached_entry = cache.get(symbol)
        if cached_entry:
            if datetime.now() - datetime.fromisoformat(cached_entry['timestamp']) < timedelta(minutes=CACHE_VALID_MINUTES):
                logger.debug(f"[GrokSentimentFetcher] Cache hit voor: {symbol}")
                return cached_entry['data']

        query = f"{symbol} sentiment OR economic news"
        logger.debug(f"[GrokSentimentFetcher] Zoeken naar: {query} via Live Search API")

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        params = {
            'query': query,
            'limit': 10, # Haal iets meer op om te filteren
            'sortBy': 'date'
        }

        try:
            # Gebruik asyncio.to_thread om blocking requests in een async context te draaien.
            response = await asyncio.to_thread(requests.get, self.live_search_url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            relevant_results_py = []
            for result in data.get('results', []):
                content = result.get('content', '')
                if content and symbol.lower() in content.lower(): # Check if symbol is in content
                    relevant_results_py.append(result)

            # Sort by timestamp (newest first) and take top 5
            relevant_results_py.sort(key=lambda x: datetime.fromisoformat(x.get('timestamp', datetime.min.isoformat())), reverse=True)
            relevant_results_py = relevant_results_py[:5]


            logger.debug(f"[GrokSentimentFetcher] Data opgehaald: {relevant_results_py}")

            # Sla op in cache
            cache[symbol] = {
                'data': relevant_results_py,
                'timestamp': datetime.now().isoformat()
            }
            self._save_cache(cache)

            return relevant_results_py

        except requests.exceptions.RequestException as e:
            logger.error(f"[GrokSentimentFetcher] ❌ Fout bij aanroep Grok Live Search API: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"[GrokSentimentFetcher] ❌ Fout bij parseren JSON respons: {e}. Raw response: {response.text if 'response' in locals() else 'N/A'}")
            return []
        except Exception as e:
            logger.error(f"[GrokSentimentFetcher] ❌ Onverwachte fout: {e}")
            return []

# Voorbeeld van hoe je het zou kunnen gebruiken (voor testen)
if __name__ == "__main__":
    # Corrected path for .env when running this script directly
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    dotenv.load_dotenv(dotenv_path)

    async def run_test_grok_sentiment_fetcher():
        try:
            fetcher = GrokSentimentFetcher()
            test_symbol = 'ETH' # Gebruik een echt token om te testen

            print(f"\n--- Test GrokSentimentFetcher voor {test_symbol} ---")
            # Create dummy cache dir for testing if it does not exist
            if not os.path.exists(CACHE_DIR):
                os.makedirs(CACHE_DIR)
                print(f"Created cache directory at {CACHE_DIR} for testing.")

            sentiment_data = await fetcher.fetch_live_search_data(test_symbol)

            if sentiment_data:
                print(f"Opgehaalde sentiment data voor {test_symbol} ({len(sentiment_data)} items):")
                for item in sentiment_data:
                    print(f"- Titel: {item.get('title', 'N/A')}")
                    print(f"  Bron: {item.get('source', 'N/A')}")
                    print(f"  Inhoud (eerste 100 chars): {item.get('content', 'N/A')[:100]}...")
                    print(f"  Tijd: {item.get('timestamp', 'N/A')}")
                    print("-" * 20)
            else:
                print(f"Geen sentiment data opgehaald voor {test_symbol}.")

            # Cleanup dummy cache file if created by test
            # if os.path.exists(CACHE_FILE):
            #     os.remove(CACHE_FILE)
            #     print(f"Test cache file {CACHE_FILE} removed.")

        except ValueError as e:
            print(f"Fout: {e}")
        except Exception as e:
            print(f"Algemene fout tijdens test: {e}")

    asyncio.run(run_test_grok_sentiment_fetcher())
