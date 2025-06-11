# core/trade_logger.py
import logging
import json
import os
from datetime import datetime, timedelta # Import timedelta
from typing import Dict, Any, Optional, List, Union
import asyncio # For async file operations

# Configure logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Default, can be overridden by global config

MEMORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'memory')
TRADE_LOG_FILE = os.path.join(MEMORY_DIR, 'trade_log.json')
DECISION_LOG_FILE = os.path.join(MEMORY_DIR, 'decision_log.json')

os.makedirs(MEMORY_DIR, exist_ok=True) # Ensure memory directory exists

# Helper async functions from IntervalSelector, adapted for general use
async def _read_json_async(filepath: str) -> List[Dict[str, Any]]: # Expecting a list of logs
    def read_file_sync():
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return [] # Return empty list if file doesn't exist or is empty
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                content = json.load(f)
                # Ensure content is a list; if not, it might be an old/corrupt file
                return content if isinstance(content, list) else []
            except json.JSONDecodeError:
                logger.warning(f"JSON file {filepath} is corrupt or empty. Returning empty list.")
                return [] # Return empty list if corrupt
    try:
        return await asyncio.to_thread(read_file_sync)
    except FileNotFoundError:
        return []

async def _append_json_list_async(filepath: str, data_to_append: Dict[str, Any]):
    # Reads a list, appends to it, and writes back the list.
    # This is not the most performant for very large logs but simple for moderate use.
    current_data = await _read_json_async(filepath)
    current_data.append(data_to_append)

    def write_file_sync():
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(current_data, f, indent=2)
    try:
        await asyncio.to_thread(write_file_sync)
    except Exception as e:
        logger.error(f"Error writing to JSON file {filepath}: {e}")

class TradeLogger:
    """
    Logs trade details and AI decision-making processes to JSON files for analysis and learning.
    Vertaald van concepten in tradeLogger.js en uitgebreid.
    """

    def __init__(self):
        logger.info("TradeLogger geïnitialiseerd.")

    async def log_trade_event(
        self,
        event_type: str, # e.g., 'entry', 'exit', 'sl_update', 'roi_exit', 'stoploss_exit'
        trade_details: Dict[str, Any],
        ai_decision_info: Optional[Dict[str, Any]] = None,
        strategy_context: Optional[Dict[str, Any]] = None
    ):
        """
        Logs een belangrijke trade event (entry, exit, etc.) met relevante data.
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "trade_id": trade_details.get("id", trade_details.get("trade_id")), # Freqtrade trade id
            "pair": trade_details.get("pair"),
            "open_rate": trade_details.get("open_rate"),
            "close_rate": trade_details.get("close_rate"),
            "profit_pct": trade_details.get("profit_pct"),
            "profit_abs": trade_details.get("profit_abs"),
            "stake_amount": trade_details.get("stake_amount"),
            "open_date": trade_details.get("open_date").isoformat() if isinstance(trade_details.get("open_date"), datetime) else trade_details.get("open_date"),
            "close_date": trade_details.get("close_date").isoformat() if isinstance(trade_details.get("close_date"), datetime) else trade_details.get("close_date"),
            "exit_reason": trade_details.get("exit_reason", trade_details.get("sell_reason")),
            "current_rate": trade_details.get("current_rate"), # For SL updates or current status
            "ai_decision": ai_decision_info if ai_decision_info else {},
            "strategy_context": strategy_context if strategy_context else {}
        }

        try:
            await _append_json_list_async(TRADE_LOG_FILE, log_entry)
            logger.info(f"Trade event '{event_type}' gelogd voor pair {log_entry['pair']}, trade ID {log_entry['trade_id']}.")
        except Exception as e:
            logger.error(f"Kon trade event niet loggen: {e}")

    async def log_ai_decision(
        self,
        decision_type: str, # e.g., 'entry_evaluation', 'exit_evaluation', 'sl_optimization'
        pair: str,
        strategy_id: str,
        input_data_summary: Dict[str, Any], # Samenvatting van data gebruikt voor AI (indicatoren, context)
        ai_responses: List[Dict[str, Any]], # List of responses from different AIs (GPT, Grok)
        final_decision: Dict[str, Any], # Het uiteindelijke besluit en de reden
        trade_context: Optional[Dict[str, Any]] = None # Optionele trade context
    ):
        """
        Logs de details van een AI-besluitvormingsproces.
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "decision_type": decision_type,
            "pair": pair,
            "strategy_id": strategy_id,
            "input_data_summary": input_data_summary,
            "ai_responses": ai_responses,
            "final_decision": final_decision,
            "trade_context": trade_context if trade_context else {}
        }

        try:
            await _append_json_list_async(DECISION_LOG_FILE, log_entry)
            logger.info(f"AI decision '{decision_type}' gelogd voor {pair} (Strategie: {strategy_id}).")
        except Exception as e:
            logger.error(f"Kon AI decision niet loggen: {e}")

# Voorbeeld van hoe je het zou kunnen gebruiken (voor testen)
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG, # Zet op DEBUG voor testen
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

    async def run_test_trade_logger():
        logger_instance = TradeLogger()

        # Voorbeeld trade details
        mock_trade_entry = {
            "id": "TEST_TRADE_001", "trade_id": "TEST_TRADE_001", "pair": "ETH/USDT",
            "open_rate": 2000.0, "stake_amount": 100.0,
            "open_date": datetime.now() - timedelta(hours=1),
            "profit_pct": 0.0, "profit_abs": 0.0, "current_rate": 2000.0
        }
        mock_ai_entry_decision = {
            "gpt_response": {"intentie": "LONG", "confidence": 0.8, "reflectie": "Looks good"},
            "grok_response": {"intentie": "LONG", "confidence": 0.7, "reflectie": "Strong signal"},
            "consensus_intentie": "LONG", "combined_confidence": 0.75,
            "reason": "AI_CONSENSUS_LONG"
        }
        mock_strategy_context_entry = {"timeframe": "5m", "active_bias": 0.6}

        await logger_instance.log_trade_event(
            event_type="entry",
            trade_details=mock_trade_entry,
            ai_decision_info=mock_ai_entry_decision,
            strategy_context=mock_strategy_context_entry
        )

        # Voorbeeld AI decision logging
        mock_input_summary = {"rsi": 55, "macd_hist": 0.001, "current_profit": "-0.01"}
        mock_final_decision_sl = {"new_stoploss_pct": -0.05, "reason": "volatility_spike"}

        await logger_instance.log_ai_decision(
            decision_type="sl_optimization",
            pair="ETH/USDT",
            strategy_id="DUOAI_Strategy_v1",
            input_data_summary=mock_input_summary,
            ai_responses=[{"model": "gpt", "advice": "tighten SL to 5%"}, {"model": "grok", "advice": "SL at 4.8%"}],
            final_decision=mock_final_decision_sl,
            trade_context={"trade_id": "TEST_TRADE_001", "current_profit_pct": -0.01}
        )


        mock_trade_exit = {
            **mock_trade_entry, # Base details
            "close_rate": 2100.0, "profit_pct": 0.05, "profit_abs": 5.0,
            "close_date": datetime.now(), "exit_reason": "roi"
        }
        mock_ai_exit_decision = {
            "gpt_response": {"intentie": "SELL", "confidence": 0.9, "reflectie": "Peak reached"},
            "grok_response": {"intentie": "SELL", "confidence": 0.85, "reflectie": "Exit now"},
            "consensus_intentie": "SELL", "combined_confidence": 0.875,
            "reason": "AI_SELL_CONFIRMED"
        }
        await logger_instance.log_trade_event(
            event_type="exit",
            trade_details=mock_trade_exit,
            ai_decision_info=mock_ai_exit_decision
        )

        # Check if files were created and have content (simple check)
        trade_log_content = await _read_json_async(TRADE_LOG_FILE)
        decision_log_content = await _read_json_async(DECISION_LOG_FILE)

        assert len(trade_log_content) >= 2, "Trade log should have at least two entries."
        assert len(decision_log_content) >= 1, "Decision log should have at least one entry."

        print(f"\nTrade log content sample (last entry): \n{json.dumps(trade_log_content[-1], indent=2)}")
        print(f"\nDecision log content sample (last entry): \n{json.dumps(decision_log_content[-1], indent=2)}")

        # Clean up dummy log files (optional)
        # os.remove(TRADE_LOG_FILE)
        # os.remove(DECISION_LOG_FILE)

    asyncio.run(run_test_trade_logger())
