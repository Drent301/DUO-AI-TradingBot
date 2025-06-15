# main.py
import os
import sys
from pathlib import Path
import logging
# import dotenv # dotenv wordt geladen binnen load_and_validate_config

# Importeer de nieuwe validatiefunctie
from utils.config_validator import load_and_validate_config
from freqtrade.main import main as freqtrade_main


# Definieer paden aan het begin voor duidelijkheid
BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
ENV_FILE_PATH = BASE_DIR / ".env"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.json" # Standaard config path
LOG_DIR_RELATIVE = "user_data/logs" # Relatief aan user_data_dir uit config

# Logger wordt geconfigureerd na het laden van de config
logger = logging.getLogger(__name__)

def setup_logging(log_file_path: Path):
    """Configureert de basis logging."""
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path), # Log to a file
            logging.StreamHandler(sys.stdout) # Log also to console
        ]
    )

def main():
    """
    Hoofd entrypoint voor de DUO-AI Trading Bot.
    Coördineert het opstarten van Freqtrade en AI-componenten.
    """
    try:
        # Laad en valideer configuratie eerst
        # Gebruik paden relatief tot de locatie van main.py
        config_path_obj = DEFAULT_CONFIG_PATH
        env_path_obj = ENV_FILE_PATH

        ft_config = load_and_validate_config(str(config_path_obj), str(env_path_obj))

        # Bepaal log directory gebaseerd op user_data_dir uit config
        # en maak het pad absoluut als user_data_dir relatief is aan BASE_DIR
        user_data_dir = Path(ft_config.get("user_data_dir", "user_data"))
        if not user_data_dir.is_absolute():
            user_data_dir = BASE_DIR / user_data_dir

        log_dir = user_data_dir / LOG_DIR_RELATIVE.split('/')[1] # Neem 'logs' deel
        log_file_name = "main.log" # Specifiek logbestand voor main
        log_file_path = log_dir / log_file_name

        # Configureer logging met het pad uit de config
        setup_logging(log_file_path)

        logger.info(f"Gevalideerde configuratie geladen. Logbestand ingesteld op: {log_file_path}")

    except ValueError as e:
        # Logging is mogelijk nog niet volledig geconfigureerd, print naar stderr
        print(f"Configuratiefout: {e}", file=sys.stderr)
        # Probeer toch te loggen als de basislogger al bestaat
        if logger:
            logger.error(f"Configuratiefout: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        print(f"Onverwachte fout tijdens initialisatie: {e}", file=sys.stderr)
        if logger:
            logger.error(f"Onverwachte fout tijdens initialisatie: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Welkom bij de DUO-AI Trading Bot!")
    logger.info("Dit is de start van je zelflerende trading systeem.")

    # Stel de command line arguments in voor Freqtrade
    # Gebruik het geconfigureerde pad voor config.json
    sys.argv = [
        "freqtrade",
        "trade",
        "--config", str(config_path_obj), # Gebruik het Path object, geconverteerd naar string
        "--strategy", "DUOAI_Strategy" # TODO: Strategienaam mogelijk ook uit config halen?
    ]

    logger.info(f"Freqtrade starten met argumenten: {' '.join(sys.argv)}")

    # Roep Freqtrade's main functie aan
    try:
        freqtrade_main()
    except SystemExit as e:
        # SystemExit wordt verwacht bij normaal afsluiten, dus info loggen.
        logger.info(f"Freqtrade afgesloten met status: {e.code}.")
        # Afhankelijk van de exit code, kan dit een fout zijn of niet.
        # Freqtrade gebruikt verschillende codes. 0 is meestal succes.
        # Her-raise de SystemExit exceptie om het gedrag van Freqtrade niet te verstoren.
        raise
    except Exception as e:
        # Vang alle andere exceptions die Freqtrade zou kunnen gooien
        logger.error(f"Algemene fout tijdens het uitvoeren van Freqtrade: {e}", exc_info=True)
        sys.exit(1)  # Sluit af met een error code


    # Voorbeeld van hoe je Freqtrade kunt aanroepen via de command line interface
    # Dit zal later worden geïntegreerd in de reflectiecyclus of handmatig via terminal
    # Voor testen:
    # command = ["freqtrade", "backtesting", "--strategy", "DUOAI_Strategy", "--config", "config/config.json"]
    # logger.info(f"Probeer Freqtrade backtest te starten met: {' '.join(command)}")
    # # Je zou subprocess.run(command) kunnen gebruiken, maar voor nu leggen we het hier uit
    # # De gebruiker zal dit commando handmatig uitvoeren via de terminal.

    # logger.info("\nGebruik 'freqtrade' commando's voor backtesting en live trading na configuratie.")
    # logger.info("Voorbeeld Backtest: freqtrade backtesting --strategy DUOAI_Strategy -c config/config.json")
    # logger.info("Voorbeeld Dry-Run: freqtrade trade --config config/config.json --strategy DUOAI_Strategy")
    # logger.info("Zorg ervoor dat je virtual environment geactiveerd is en Freqtrade geïnstalleerd is.")

    # Hieronder komen de AI-coördinatie-logica die periodiek wordt aangeroepen
    # door een scheduler, of getriggerd door Freqtrade events.
    # Voorbeeld:
    # from core.reflectie_lus import ReflectieLus # Pas pad aan indien nodig
    # reflectie_lus = ReflectieLus(ft_config) # Geef config mee indien nodig
    # import asyncio # Indien async functies gebruikt worden
    # asyncio.run(reflectie_lus.start_reflectie_lus(symbol='ETH/USDT', interval_ms=5 * 60 * 1000))
    # Echter, de reflectiecyclus wordt initieel niet direct vanuit main.py gestart in de Freqtrade context,
    # maar getriggerd door trade-events of een aparte daemon/scheduler.

if __name__ == "__main__":
    # Voordat main() wordt aangeroepen, is logging nog niet geconfigureerd.
    # Fouten hier (b.v. import errors) gaan naar stderr.
    main()
