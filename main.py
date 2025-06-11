# main.py
import os
import sys
from pathlib import Path
import logging
import dotenv

# Laad omgevingsvariabelen uit .env bestand
dotenv.load_dotenv()

# Configureer basis logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

# Voeg de projectroot en core map toe aan PYTHONPATH
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'core'))
sys.path.append(str(project_root / 'strategies'))

logger = logging.getLogger(__name__)

def main():
    """
    Hoofd entrypoint voor de DUO-AI Trading Bot.
    Coördineert het opstarten van Freqtrade en AI-componenten.
    """
    logger.info("Welkom bij de DUO-AI Trading Bot!")
    logger.info("Dit is de start van je zelflerende trading systeem.")

    # Voorbeeld van hoe je Freqtrade kunt aanroepen via de command line interface
    # Dit zal later worden geïntegreerd in de reflectiecyclus of handmatig via terminal
    # Voor testen:
    # command = ["freqtrade", "backtesting", "--strategy", "DUOAI_Strategy", "--config", "config/config.json"]
    # logger.info(f"Probeer Freqtrade backtest te starten met: {' '.join(command)}")
    # # Je zou subprocess.run(command) kunnen gebruiken, maar voor nu leggen we het hier uit
    # # De gebruiker zal dit commando handmatig uitvoeren via de terminal.

    logger.info("\nGebruik 'freqtrade' commando's voor backtesting en live trading na configuratie.")
    logger.info("Voorbeeld Backtest: freqtrade backtesting --strategy DUOAI_Strategy -c config/config.json")
    logger.info("Voorbeeld Dry-Run: freqtrade trade --config config/config.json --strategy DUOAI_Strategy")
    logger.info("Zorg ervoor dat je virtual environment geactiveerd is en Freqtrade geïnstalleerd is.")

    # Hieronder komen de AI-coördinatie-logica die periodiek wordt aangeroepen
    # door een scheduler, of getriggerd door Freqtrade events.
    # Bijvoorbeeld:
    # from core.reflectie_lus import ReflectieLus
    # reflectie_lus = ReflectieLus()
    # asyncio.run(reflectie_lus.start_reflectie_lus(symbol='ETH/USDT', interval_ms=5 * 60 * 1000))
    # Echter, de reflectiecyclus wordt initieel niet direct vanuit main.py gestart in de Freqtrade context,
    # maar getriggerd door trade-events of een aparte daemon/scheduler.

if __name__ == "__main__":
    main()
