# core/grok_sentiment_fetcher.py
# This module is dependent on GROK_API_URL for fetching live social data.
# Grok's 'Live Feat' is expected to provide access to platforms like X (Twitter)
# and potentially TradingView for sentiment analysis.
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

    async def fetch_live_search_data(self, symbol: str, perform_fetch: bool = True) -> List[Dict[str, Any]]:
        """
        Haalt real-time data op via de hypothetische Live Search API.
        Optimaliseert voor de 5 meest recente berichten die direct over de coin gaan.
        Implementeert caching om API-limieten te respecteren.
        """
        if not perform_fetch:
            logger.info(f"[GrokSentimentFetcher] Fetch skipped for {symbol} due to perform_fetch=False.")
            return []

        # Controleer cache
        cache = self._load_cache()
        cached_entry = cache.get(symbol)
        if cached_entry:
            if datetime.now() - datetime.fromisoformat(cached_entry['timestamp']) < timedelta(minutes=CACHE_VALID_MINUTES):
                logger.debug(f"[GrokSentimentFetcher] Cache hit voor: {symbol}")
                # Ensure what's returned from cache matches the new structure.
                # If cache stores raw data, it would need reprocessing.
                # The modification below stores processed_results, so this should be fine.
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

            processed_results = []
            # Filter for relevance (symbol in content) first
            relevant_api_results = []
            for result in data.get('results', []):
                content_check_text = result.get('content', '') # Text to check for symbol presence
                if symbol.lower() in content_check_text.lower(): # Sticking to original check for now
                    relevant_api_results.append(result)

            # Sort by timestamp (newest first)
            relevant_api_results.sort(key=lambda x: datetime.fromisoformat(x.get('timestamp', datetime.min.isoformat())), reverse=True)

            # Take top 5
            top_5_results = relevant_api_results[:5]

            # Transform the top 5 results into the desired output structure
            for item in top_5_results:
                source = item.get('source', 'Unknown Source')
                text_content = item.get('content', '')
                if not text_content: # Fallback to description if content is empty
                    text_content = item.get('description', 'N/A')

                sentiment_value = item.get('sentiment', 'neutral') # Assume API provides this

                processed_results.append({
                    'source': source,
                    'text': text_content,
                    'sentiment': sentiment_value
                    # Optional: include timestamp if needed downstream, though not explicitly requested for this structure
                    # 'timestamp': item.get('timestamp')
                })

            logger.debug(f"[GrokSentimentFetcher] Processed data: {processed_results}")

            # Sla op in cache - cache should store the processed_results
            cache[symbol] = {
                'data': processed_results, # Store the new structure
                'timestamp': datetime.now().isoformat()
            }
            self._save_cache(cache)

            return processed_results

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

            sentiment_data = await fetcher.fetch_live_search_data(test_symbol, perform_fetch=True)

            if sentiment_data:
                print(f"Opgehaalde sentiment data voor {test_symbol} ({len(sentiment_data)} items) with perform_fetch=True:")
                for item in sentiment_data:
                    print(f"- Source: {item.get('source', 'N/A')}")
                    print(f"  Text (eerste 100 chars): {item.get('text', 'N/A')[:100]}...")
                    print(f"  Sentiment: {item.get('sentiment', 'N/A')}")
                    print("-" * 20)
            else:
                print(f"Geen sentiment data opgehaald voor {test_symbol} with perform_fetch=True.")

            print(f"\n--- Test GrokSentimentFetcher voor {test_symbol} with perform_fetch=False ---")
            sentiment_data_conditional_skip = await fetcher.fetch_live_search_data(test_symbol, perform_fetch=False)
            if not sentiment_data_conditional_skip:
                print(f"Data fetching skipped as expected for {test_symbol} when perform_fetch=False. Result: {sentiment_data_conditional_skip}")
            else:
                print(f"ERROR: Data was fetched for {test_symbol} even when perform_fetch=False. Result: {sentiment_data_conditional_skip}")

            # Cleanup dummy cache file if created by test
            # if os.path.exists(CACHE_FILE):
            #     os.remove(CACHE_FILE)
            #     print(f"Test cache file {CACHE_FILE} removed.")

        except ValueError as e:
            print(f"Fout: {e}")
        except Exception as e:
            print(f"Algemene fout tijdens test: {e}")

    asyncio.run(run_test_grok_sentiment_fetcher())
