# core/bitvavo_executor.py
import ccxt
import os
import logging
from typing import Dict, Any, Optional, List
import asyncio # Nodig voor de async main-test
import dotenv # Toegevoegd voor de __main__ sectie

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class BitvavoExecutor:
    """
    Voert handelsoperaties uit op de Bitvavo exchange via CCXT.
    Deze module is noodzakelijk omdat Freqtrade Bitvavo nog niet native ondersteunt.
    """

    def __init__(self):
        self.api_key = os.getenv('BITVAVO_API_KEY')
        self.secret_key = os.getenv('BITVAVO_SECRET_KEY')

        if not self.api_key or not self.secret_key:
            logger.error("BITVAVO_API_KEY of BITVAVO_SECRET_KEY is niet ingesteld in de omgevingsvariabelen.")
            raise ValueError("Bitvavo API-sleutels ontbreken. Stel deze in je .env bestand in.")

        self.exchange = ccxt.bitvavo({
            'apiKey': self.api_key,
            'secret': self.secret_key,
            'enableRateLimit': True, # Zeer belangrijk om rate limits te respecteren
            'options': {
                'adjustForTimeDifference': True, # Kan helpen bij time-sync problemen
                'recvWindow': 10000 # Aanbevolen voor sommige exchanges
            }
        })
        logger.info("BitvavoExecutor geïnitialiseerd via CCXT.")

    async def fetch_balance(self, currency: Optional[str] = None) -> Dict[str, Any]:
        """
        Haalt het huidige saldo op van de Bitvavo-rekening.
        :param currency: Optioneel, specifieke valuta om op te vragen (bijv. 'ETH', 'EUR').
                         Als None, haalt het alle valuta op.
        :return: Een dictionary met saldo-informatie.
        """
        try:
            balance = await self.exchange.fetch_balance()
            if currency:
                return balance.get(currency, {'free': 0, 'used': 0, 'total': 0})
            return balance
        except ccxt.NetworkError as e:
            logger.error(f"Netwerkfout bij ophalen Bitvavo saldo: {e}")
            return {}
        except ccxt.ExchangeError as e:
            logger.error(f"Exchangefout bij ophalen Bitvavo saldo: {e}")
            return {}
        except Exception as e:
            logger.error(f"Onverwachte fout bij ophalen Bitvavo saldo: {e}")
            return {}

    async def create_market_buy_order(self, symbol: str, amount: float) -> Optional[Dict[str, Any]]:
        """
        Plaatst een market kooporder.
        :param symbol: Het handelspaar (bijv. 'ETH/EUR').
        :param amount: De hoeveelheid van de basisvaluta om te kopen (bijv. 0.01 ETH).
        :return: De orderinformatie of None bij fout.
        """
        try:
            order = await self.exchange.create_market_buy_order(symbol, amount)
            logger.info(f"Market BUY order geplaatst voor {amount} {symbol}: {order}")
            return order
        except ccxt.InsufficientFunds as e:
            logger.warning(f"Onvoldoende saldo voor kooporder {symbol}: {e}")
            return None
        except ccxt.InvalidOrder as e:
            logger.error(f"Ongeldige kooporder {symbol}: {e}")
            return None
        except ccxt.NetworkError as e:
            logger.error(f"Netwerkfout bij plaatsen kooporder {symbol}: {e}")
            return None
        except ccxt.ExchangeError as e:
            logger.error(f"Exchangefout bij plaatsen kooporder {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Onverwachte fout bij plaatsen kooporder {symbol}: {e}")
            return None

    async def create_market_sell_order(self, symbol: str, amount: float) -> Optional[Dict[str, Any]]:
        """
        Plaatst een market verkooporder.
        :param symbol: Het handelspaar (bijv. 'ETH/EUR').
        :param amount: De hoeveelheid van de basisvaluta om te verkopen.
        :return: De orderinformatie of None bij fout.
        """
        try:
            order = await self.exchange.create_market_sell_order(symbol, amount)
            logger.info(f"Market SELL order geplaatst voor {amount} {symbol}: {order}")
            return order
        except ccxt.InsufficientFunds as e:
            logger.warning(f"Onvoldoende saldo voor verkooporder {symbol}: {e}")
            return None
        except ccxt.InvalidOrder as e:
            logger.error(f"Ongeldige verkooporder {symbol}: {e}")
            return None
        except ccxt.NetworkError as e:
            logger.error(f"Netwerkfout bij plaatsen verkooporder {symbol}: {e}")
            return None
        except ccxt.ExchangeError as e:
            logger.error(f"Exchangefout bij plaatsen verkooporder {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Onverwachte fout bij plaatsen verkooporder {symbol}: {e}")
            return None

    async def fetch_ohlcv(self, symbol: str, timeframe: str = '5m', limit: int = 100) -> List[List]:
        """
        Haalt OHLCV (Open, High, Low, Close, Volume) data op.
        :param symbol: Het handelspaar (bijv. 'ETH/EUR').
        :param timeframe: Het interval (bijv. '5m', '1h').
        :param limit: Maximaal aantal candles om op te halen.
        :return: Een lijst van OHLCV lijsten.
        """
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            logger.debug(f"OHLCV data opgehaald voor {symbol} ({timeframe}, {limit} candles).")
            return ohlcv
        except Exception as e:
            logger.error(f"Fout bij ophalen OHLCV voor {symbol}: {e}")
            return []

# Voorbeeld van hoe je het zou kunnen gebruiken (voor testen)
if __name__ == "__main__":
    # Zorg dat .env bestand geladen wordt voor dit testscript
    dotenv.load_dotenv()

    async def run_test_executor():
        try:
            executor = BitvavoExecutor()

            print("\n--- Test BitvavoExecutor ---")

            # Test saldo ophalen
            print("Saldo ophalen...")
            balance_eur = await executor.fetch_balance('EUR')
            print(f"Beschikbaar EUR: {balance_eur.get('free', 'N/A')}")
            balance_eth = await executor.fetch_balance('ETH')
            print(f"Beschikbaar ETH: {balance_eth.get('free', 'N/A')}")

            # Test OHLCV data ophalen
            print("\nOHLCV data ophalen voor ETH/EUR (5m, 10 candles)...")
            ohlcv_data = await executor.fetch_ohlcv('ETH/EUR', '5m', 10)
            if ohlcv_data:
                print(f"Laatste candle: {ohlcv_data[-1]}")
            else:
                print("Geen OHLCV data opgehaald.")

            # WAARSCHUWING: DE VOLGENDE REGELS PLAATSEN ECHTE (OF DRY-RUN) ORDERS.
            # DEZE EXECUTOR HOUDT GEEN REKENING MET FREQTRADE'S DRY_RUN MODE.
            # DEZE MOETEN UITGECOMMENTAARD ZIJN VOOR PRODUCTIE TENZIJ JE ZE BEWUST WIL GEBRUIKEN.
            # print("\nProbeer een kleine kooporder te plaatsen (dry run)...")
            # buy_order = await executor.create_market_buy_order('ETH/EUR', 0.001)
            # if buy_order:
            #     print(f"Kooporder resultaat: {buy_order}")
            # else:
            #     print("Kooporder mislukt.")

            # print("\nProbeer een kleine verkooporder te plaatsen (dry run)...")
            # sell_order = await executor.create_market_sell_order('ETH/EUR', 0.001)
            # if sell_order:
            #     print(f"Verkooporder resultaat: {sell_order}")
            # else:
            #     print("Verkooporder mislukt.")

        except ValueError as e:
            print(f"Fout: {e}")
        except Exception as e:
            print(f"Algemene fout tijdens test: {e}")

    asyncio.run(run_test_executor())
