# core/pre_trainer.py
import logging
import pandas as pd
from sqlalchemy import create_engine, text
from typing import List, Dict, Optional, Tuple
import asyncio
import os
from datetime import datetime, timedelta, timezone

# Importeer core componenten die nodig zijn voor pre-training of context
from core.bias_reflector import BiasReflector
from core.confidence_engine import ConfidenceEngine
from core.params_manager import ParamsManager
from core.trade_logger import TradeLogger
from core.interval_selector import IntervalSelector
from core.entry_decider import EntryDecider
from core.exit_optimizer import ExitOptimizer
# TODO: AI Reflectors (GPT, Grok) als ze direct gebruikt worden voor pre-training analyse

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO) # Wordt nu idealiter globaal beheerd

class PreTrainer:
    """
    Verantwoordelijk voor het pre-trainen van AI-modellen en het initialiseren van 'learned' states
    op basis van historische data uit de Freqtrade database.
    """
    def __init__(self, db_url: str, params_manager: Optional[ParamsManager] = None):
        self.db_url = db_url
        try:
            self.engine = create_engine(self.db_url)
        except Exception as e:
            logger.error(f"Kon geen verbinding maken met de database via {db_url}: {e}")
            self.engine = None

        # Initialiseer componenten die 'geleerde' data opslaan/gebruiken
        self.bias_reflector = BiasReflector()
        self.confidence_engine = ConfidenceEngine()
        self.params_manager = params_manager if params_manager else ParamsManager() # Gebruik meegegeven of maak nieuwe
        self.trade_logger = TradeLogger() # Voor het loggen van pre-train acties als 'trades'
        self.interval_selector = IntervalSelector()
        # Entry/Exit deciders kunnen nodig zijn als pre-training hun interne state wil opwarmen
        # of als hun logica wordt gesimuleerd tijdens pre-training.
        self.entry_decider = EntryDecider(params_manager=self.params_manager)
        self.exit_optimizer = ExitOptimizer(params_manager=self.params_manager) # ExitOptimizer ook ParamsManager geven

        logger.info(f"PreTrainer geïnitialiseerd met database URL: {db_url}")

    async def _fetch_historical_trades(self, days_history: int = 90) -> pd.DataFrame:
        """
        Haalt gesloten trades op uit de Freqtrade database van de laatste N dagen.
        """
        if not self.engine:
            logger.error("Database engine niet beschikbaar. Kan geen trades ophalen.")
            return pd.DataFrame()

        query = text(f"""
            SELECT *
            FROM trades
            WHERE close_date IS NOT NULL
            AND open_date >= :start_date
            ORDER BY close_date ASC
        """)
        start_date_param = datetime.now(timezone.utc) - timedelta(days=days_history)

        try:
            with self.engine.connect() as connection:
                trades_df = pd.read_sql_query(query, connection, params={"start_date": start_date_param.isoformat()})
            logger.info(f"{len(trades_df)} gesloten trades opgehaald van de laatste {days_history} dagen.")
            return trades_df
        except Exception as e:
            logger.error(f"Fout bij het ophalen van historische trades: {e}")
            return pd.DataFrame()

    async def _simulate_trade_for_learning(self, trade_row: pd.Series, strategy_id: str):
        """
        Simuleert een enkele trade om AI componenten (Bias, Confidence) te 'trainen'.
        Dit is een vereenvoudigde simulatie. Een echte implementatie zou de strategie
        en AI-beslissingen op historische candle-data moeten naspelen.
        """
        pair = trade_row['pair']
        profit_pct = trade_row['close_profit_ratio'] # close_profit_ratio is 0.01 voor 1%

        # Voor BiasReflector: update bias op basis van winst/verlies
        # [cite_start] Update bias based on trade outcome [cite: 89, 163, 232, 309, 380, 452, 522]
        # TODO: De `update_bias_score` methode in BiasReflector is nog niet gedefinieerd in de placeholder.
        # Voor nu simuleren we dat het bestaat en loggen we de actie.
        # self.bias_reflector.update_bias_score(pair, strategy_id, profit_pct > 0) # True als winst
        logger.debug(f"Simulating bias update for {pair}: profit_pct={profit_pct:.3f}")


        # Voor ConfidenceEngine: update confidence
        # [cite_start] Update confidence based on trade outcome [cite: 91, 165, 234, 311, 382, 455, 525]
        # TODO: De `update_confidence_score` methode in ConfidenceEngine is nog niet gedefinieerd.
        # Voor nu simuleren we dat het bestaat en loggen we de actie.
        # self.confidence_engine.update_confidence_score(pair, strategy_id, new_score=abs(profit_pct), trade_outcome="profit" if profit_pct >0 else "loss")
        logger.debug(f"Simulating confidence update for {pair}: new_score_proxy={abs(profit_pct):.3f}")


        # Log de gesimuleerde trade alsof het een echte AI-begeleide trade was
        # Dit helpt de TradeLogger te vullen met historische context.
        # We construeren een mock `trade_details` en `ai_decision_info`
        mock_trade_details = {
            "id": f"pretrain_{trade_row['id']}", "pair": pair,
            "open_rate": trade_row['open_rate'], "close_rate": trade_row['close_rate'],
            "profit_pct": profit_pct, "profit_abs": trade_row['close_profit_abs'],
            "stake_amount": trade_row['stake_amount'],
            "open_date": pd.to_datetime(trade_row['open_date']).isoformat() if pd.notnull(trade_row['open_date']) else None,
            "close_date": pd.to_datetime(trade_row['close_date']).isoformat() if pd.notnull(trade_row['close_date']) else None,
            "exit_reason": trade_row.get('exit_reason', 'pretrain_simulated')
        }
        # Voor pre-training is de AI decision info misschien niet volledig beschikbaar of relevant
        # tenzij we ook de AI besluitvorming op historische data simuleren.
        mock_ai_decision = {"pretrain_simulated": True, "profit_based_outcome": "win" if profit_pct > 0 else "loss"}

        await self.trade_logger.log_trade_event(
            event_type="pretrain_sim_exit",
            trade_details=mock_trade_details,
            ai_decision_info=mock_ai_decision,
            strategy_context={"strategy_id": strategy_id, "pretrain_days": trade_row.get('pretrain_days_history', 'N/A')}
        )

    async def run_pre_training(self, strategy_id: str, days_history: int = 90):
        """
        Voert het pre-trainingsproces uit.
        """
        logger.info(f"Start pre-training voor strategie '{strategy_id}' met {days_history} dagen historie...")
        if not self.engine:
            logger.error("Database engine niet geconfigureerd. Pre-training geannuleerd.")
            return

        historical_trades_df = await self._fetch_historical_trades(days_history)

        if historical_trades_df.empty:
            logger.info("Geen historische trades gevonden voor pre-training.")
            return

        for _, trade_row in historical_trades_df.iterrows():
            # Voeg pretrain_days_history toe aan de trade_row voor logging context
            trade_row_extended = trade_row.copy()
            trade_row_extended['pretrain_days_history'] = days_history
            await self._simulate_trade_for_learning(trade_row_extended, strategy_id)

        # TODO: Eventuele andere pre-training stappen:
        # - Optimaliseren van startparameters in ParamsManager op basis van globale trade statistieken.
        # - Vooraf bepalen van beste intervallen per pair via IntervalSelector.
        # - Opwarmen van AI-modellen zelf (indien van toepassing en mogelijk zonder live calls).

        logger.info(f"Pre-training voltooid voor strategie '{strategy_id}'. {len(historical_trades_df)} trades verwerkt.")


# Voorbeeld van hoe je het zou kunnen gebruiken (voor testen)
if __name__ == "__main__":
    import dotenv
    import sys

    # Setup basic logging for testing
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

    dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

    # Gebruik een tijdelijke SQLite DB voor testen
    TEST_DB_URL = "sqlite:///./memory/pretrain_test_freqtrade.sqlite"

    # Zet dummy API keys als ze niet in .env staan, voor testdoeleinden
    if not os.getenv("OPENAI_API_KEY"): os.environ["OPENAI_API_KEY"] = "dummy_openai_key_for_testing"
    if not os.getenv("GROK_API_KEY"): os.environ["GROK_API_KEY"] = "dummy_grok_key_for_testing"


    async def setup_test_db(engine):
        """Creëert een test tabel en voegt wat data toe."""
        if not engine: return

        # Verwijder oude test DB als die bestaat
        db_path = engine.url.database
        if db_path and os.path.exists(db_path):
            os.remove(db_path)
            logger.info(f"Oude test database '{db_path}' verwijderd.")

        # (Her)creeer de engine om zeker te zijn dat het naar een schone state gaat
        engine = create_engine(TEST_DB_URL)

        with engine.connect() as connection:
            try:
                # Drop table if it exists, then create
                connection.execute(text("DROP TABLE IF EXISTS trades"))
                connection.execute(text("""
                    CREATE TABLE trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pair TEXT NOT NULL,
                        stake_amount REAL NOT NULL,
                        open_date DATETIME NOT NULL,
                        close_date DATETIME,
                        open_rate REAL NOT NULL,
                        close_rate REAL,
                        close_profit_ratio REAL, /* 0.01 = 1% */
                        close_profit_abs REAL,
                        exit_reason TEXT,
                        exchange TEXT DEFAULT 'test_exchange'
                    )
                """))

                # Voeg mock trade data toe
                trades_data_tuples = [
                    ('ETH/USDT', 100, (datetime.now(timezone.utc) - timedelta(days=10)).isoformat(), (datetime.now(timezone.utc) - timedelta(days=9, hours=12)).isoformat(), 2000, 2020, 0.01, 2.0, 'roi'), # Winst
                    ('BTC/USDT', 500, (datetime.now(timezone.utc) - timedelta(days=5)).isoformat(), (datetime.now(timezone.utc) - timedelta(days=4, hours=20)).isoformat(), 30000, 29700, -0.01, -30.0, 'stop_loss'), # Verlies
                    ('ETH/USDT', 150, (datetime.now(timezone.utc) - timedelta(days=2)).isoformat(), (datetime.now(timezone.utc) - timedelta(days=1, hours=5)).isoformat(), 2050, 2152.5, 0.05, 7.625, 'signal'), # Winst
                    ('ADA/USDT', 200, (datetime.now(timezone.utc) - timedelta(days=60)).isoformat(), (datetime.now(timezone.utc) - timedelta(days=59)).isoformat(), 1.0, 1.05, 0.05, 10.0, 'custom_exit'), # Winst
                    ('BTC/USDT', 300, (datetime.now(timezone.utc) - timedelta(days=100)).isoformat(), (datetime.now(timezone.utc) - timedelta(days=99)).isoformat(), 28000, 27000, -0.0357, -10.71, 'timeout'), # Trade te oud
                ]
                # Convert list of tuples to list of dictionaries for named parameter binding
                keys = ["pair", "stake_amount", "open_date", "close_date", "open_rate", "close_rate", "close_profit_ratio", "close_profit_abs", "exit_reason"]
                trades_data_dicts = [dict(zip(keys, trade_tuple)) for trade_tuple in trades_data_tuples]

                connection.execute(text("""
                    INSERT INTO trades (pair, stake_amount, open_date, close_date, open_rate, close_rate, close_profit_ratio, close_profit_abs, exit_reason)
                    VALUES (:pair, :stake_amount, :open_date, :close_date, :open_rate, :close_rate, :close_profit_ratio, :close_profit_abs, :exit_reason)
                """), trades_data_dicts)
                connection.commit()
                logger.info("Test database en tabel 'trades' aangemaakt met mock data.")
            except Exception as e:
                logger.error(f"Fout bij opzetten test DB: {e}")
                connection.rollback() # Rollback bij fout

    async def run_pre_trainer_test():
        # Setup test DB
        test_engine = create_engine(TEST_DB_URL)
        await setup_test_db(test_engine) # Herbouw de DB elke keer voor een schone test

        # Initialiseer PreTrainer met de test DB URL
        pre_trainer = PreTrainer(db_url=TEST_DB_URL)
        if not pre_trainer.engine: # Check of engine succesvol is aangemaakt in __init__
             logger.error("PreTrainer engine niet succesvol geïnitialiseerd in test. Stoppen.")
             return

        test_strategy_id = "DUOAI_TestStrategy"

        # Draai pre-training voor 30 dagen historie
        await pre_trainer.run_pre_training(strategy_id=test_strategy_id, days_history=30)

        # Verificaties (simpel, kan uitgebreider)
        # Check of trade_log.json is gevuld met pretrain_sim_exit events
        trade_log_content = await trade_logger._read_json_async(trade_logger.TRADE_LOG_FILE) # Gebruik helper direct
        pretrain_sim_exits = [entry for entry in trade_log_content if entry['event_type'] == 'pretrain_sim_exit']

        # We verwachten 3 trades binnen de laatste 30 dagen
        # (10d, 5d, 2d geleden. 60d en 100d vallen buiten 30d historie)
        expected_sim_trades = 3
        logger.info(f"Aantal gesimuleerde pre-train exits gelogd: {len(pretrain_sim_exits)}")
        assert len(pretrain_sim_exits) >= expected_sim_trades, \
            f"Verwachtte minstens {expected_sim_trades} pre-train sim exits, vond {len(pretrain_sim_exits)}"

        # TODO: Controleer BiasReflector en ConfidenceEngine memory als hun update methodes volledig zijn.
        # Voorbeeld:
        # bias_eth = pre_trainer.bias_reflector.get_bias_score("ETH/USDT", test_strategy_id)
        # print(f"Bias voor ETH/USDT na pre-training: {bias_eth}")
        # assert bias_eth != 0.5 # Moet zijn veranderd van default

        # Clean up trade log (optioneel)
        # if os.path.exists(trade_logger.TRADE_LOG_FILE):
        #     os.remove(trade_logger.TRADE_LOG_FILE)
        # if os.path.exists(trade_logger.DECISION_LOG_FILE): # Ook decision log als die gebruikt wordt
        #     os.remove(trade_logger.DECISION_LOG_FILE)
        # if os.path.exists(PARAMS_MEMORY_PATH): # Params memory
        #     os.remove(PARAMS_MEMORY_PATH)
        # if os.path.exists(COOLDOWN_MEMORY_PATH): # Cooldown memory
        #     os.remove(COOLDOWN_MEMORY_PATH)
        # if os.path.exists(INTERVAL_MEMORY_PATH): # Interval memory
        #     os.remove(INTERVAL_MEMORY_PATH)


        logger.info("PreTrainer test voltooid.")

    # Importeer trade_logger lokaal voor de test om de file path te gebruiken
    from core import trade_logger
    # from core.interval_selector import MEMORY_PATH as INTERVAL_MEMORY_PATH # Als je interval memory ook wilt opschonen
    # from core.cooldown_tracker import COOLDOWN_MEMORY_PATH # Als je cooldown memory ook wilt opschonen


    asyncio.run(run_pre_trainer_test())
