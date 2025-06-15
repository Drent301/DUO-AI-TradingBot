# core/bias_reflector.py
import json
import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio
import dotenv # Added for the __main__ section

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO) # Logging level configured by application

from pathlib import Path # Import Path

# Padconfiguratie met pathlib
CORE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_DIR = CORE_DIR.parent # Assumes 'core' is directly under project root
MEMORY_DIR = (PROJECT_ROOT_DIR / 'user_data' / 'memory').resolve() # Ensure absolute path
BIAS_MEMORY_FILE = (MEMORY_DIR / 'bias_memory.json').resolve()
BIAS_COOLDOWN_SECONDS = 3600 # 1 uur cooldown voor aanpassing

MEMORY_DIR.mkdir(parents=True, exist_ok=True)

class BiasReflector:
    """
    Beheert de lerende bias/voorkeur per strategie/token.
    Vertaald en geoptimaliseerd van biasReflector.js / biasReflectorCooldown.js.
    """

    def __init__(self):
        self.bias_memory = self._load_bias_memory()
        logger.info(f"BiasReflector geïnitialiseerd. Geheugen: {BIAS_MEMORY_FILE}")

    def _load_bias_memory(self) -> Dict[str, Any]:
        try:
            if BIAS_MEMORY_FILE.exists() and BIAS_MEMORY_FILE.stat().st_size > 0:
                with BIAS_MEMORY_FILE.open('r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.info(f"Bias geheugenbestand {BIAS_MEMORY_FILE} niet gevonden of leeg. Start met leeg geheugen.")
                return {}
        except FileNotFoundError:
            logger.warning(f"Bias geheugenbestand {BIAS_MEMORY_FILE} niet gevonden. Start met leeg geheugen.", exc_info=True)
            return {}
        except json.JSONDecodeError:
            logger.warning(f"Fout bij decoderen JSON uit {BIAS_MEMORY_FILE}. Start met leeg geheugen.", exc_info=True)
            return {}
        except Exception as e:
            logger.error(f"Onverwachte fout bij laden bias geheugen {BIAS_MEMORY_FILE}: {e}", exc_info=True)
            return {}


    def _save_bias_memory(self):
        try:
            with BIAS_MEMORY_FILE.open('w', encoding='utf-8') as f:
                json.dump(self.bias_memory, f, indent=2)
        except IOError as e:
            logger.error(f"Fout bij opslaan bias geheugen naar {BIAS_MEMORY_FILE}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Onverwachte algemene fout bij opslaan bias geheugen naar {BIAS_MEMORY_FILE}: {e}", exc_info=True)

    def get_bias_score(self, token: str, strategy_id: str) -> float:
        """
        Haalt de huidige bias score op voor een specifieke token/strategie.
        Retourneert een standaardwaarde als niet gevonden (0.5 is neutraal).
        """
        return self.bias_memory.get(token, {}).get(strategy_id, {}).get('bias', 0.5)

    async def update_bias(self, token: str, strategy_id: str, new_ai_bias: float, confidence: float,
                          trade_result_pct: Optional[float] = None,
                          sentiment_data_status: Optional[str] = None,
                          first_sentiment_observed: Optional[str] = None,
                          trade_direction: Optional[str] = None):
        """
        Werkt de bias score bij op basis van AI-reflectie en trade resultaat.
        Implementeert een cooldown om te snelle aanpassingen te voorkomen.
        'new_ai_bias' is de bias voorgesteld door de AI (0-1).
        'confidence' is de AI's confidence in die new_ai_bias (0-1).
        'trade_result_pct' is het winst/verlies percentage van een trade (e.g., 0.05 for +5%).
        """
        if token not in self.bias_memory:
            self.bias_memory[token] = {}
        if strategy_id not in self.bias_memory[token]:
            self.bias_memory[token][strategy_id] = {'bias': 0.5, 'last_update': None, 'trade_count': 0, 'total_profit_pct': 0.0}

        current_data = self.bias_memory[token][strategy_id]
        current_bias = current_data['bias']
        last_update_str = current_data.get('last_update')
        last_update_time = datetime.fromisoformat(last_update_str) if last_update_str else None

        # Implementeer cooldown
        if last_update_time and (datetime.now() - last_update_time).total_seconds() < BIAS_COOLDOWN_SECONDS:
            logger.debug(f"[BiasReflector] Cooldown actief voor {token}/{strategy_id}. Bias wordt niet bijgewerkt.")
            return

        # Start met de huidige bias
        calculated_new_bias = current_bias

        # Factor voor hoe sterk een trade resultaat de bias beïnvloedt.
        # Dit kan afhangen van de 'confidence' in de AI en de magnitude van het trade resultaat.
        trade_influence_factor = confidence * 0.1 # Basis invloed, verhoogd door confidence

        if trade_result_pct is not None:
            if trade_result_pct > 0: # Winst
                # Beweeg bias meer richting de AI's voorgestelde bias als die ook positief is,
                # of versterk de huidige bias als de AI neutraal/negatief was maar de trade won.
                # De aanpassing is sterker als de winst groot is.
                # Als AI positief was (new_ai_bias > 0.5), beweeg naar new_ai_bias.
                # Als AI neutraal/negatief was, verhoog de huidige bias een beetje.
                target_bias = new_ai_bias if new_ai_bias > current_bias else current_bias + 0.05 # Kleine boost als AI niet al positiever was
                adjustment = (target_bias - current_bias) * trade_influence_factor * (1 + min(trade_result_pct * 10, 1)) # Max 2x invloed
                calculated_new_bias += adjustment
            else: # Verlies
                # Beweeg bias meer richting de AI's voorgestelde bias als die ook negatief is,
                # of verzwak de huidige bias als de AI neutraal/positief was maar de trade verloor.
                target_bias = new_ai_bias if new_ai_bias < current_bias else current_bias - 0.05 # Kleine verzwakking
                adjustment = (target_bias - current_bias) * trade_influence_factor * (1 + min(abs(trade_result_pct) * 10, 1))
                calculated_new_bias += adjustment
        else:
            # Als er geen trade resultaat is, pas de bias aan richting de AI's suggestie, gewogen door confidence.
            # Dit is een standaard gewogen gemiddelde update.
            calculated_new_bias = (current_bias * (1 - confidence) + new_ai_bias * confidence)

        # Sentiment-Based Adjustment (only if trade_result_pct is present)
        if trade_result_pct is not None and sentiment_data_status == "present_and_not_empty" and first_sentiment_observed and trade_direction:
            sentiment_influence_weight = 0.05  # Weight for sentiment influence
            sentiment_aligned_with_profit = False
            sentiment_misaligned_with_profit = False

            if trade_direction == "long":
                if first_sentiment_observed == "positive" and trade_result_pct > 0: sentiment_aligned_with_profit = True
                elif first_sentiment_observed == "negative" and trade_result_pct <= 0: sentiment_aligned_with_profit = True
                elif first_sentiment_observed == "positive" and trade_result_pct <= 0: sentiment_misaligned_with_profit = True
                elif first_sentiment_observed == "negative" and trade_result_pct > 0: sentiment_misaligned_with_profit = True
            elif trade_direction == "short":
                if first_sentiment_observed == "negative" and trade_result_pct > 0: sentiment_aligned_with_profit = True
                elif first_sentiment_observed == "positive" and trade_result_pct <= 0: sentiment_aligned_with_profit = True
                elif first_sentiment_observed == "negative" and trade_result_pct <= 0: sentiment_misaligned_with_profit = True # Negative sentiment, but short lost (profit <=0)
                elif first_sentiment_observed == "positive" and trade_result_pct > 0: sentiment_misaligned_with_profit = True # Positive sentiment, but short won (profit > 0)

            sentiment_nudge = 0.0
            if sentiment_aligned_with_profit:
                adjustment_direction = 0
                if first_sentiment_observed == "positive": adjustment_direction = 1
                elif first_sentiment_observed == "negative": adjustment_direction = -1
                sentiment_nudge = sentiment_influence_weight * confidence * adjustment_direction
                calculated_new_bias += sentiment_nudge
                logger.info(f"[BiasReflector] Sentiment ALIGNED ({first_sentiment_observed} for {trade_direction}, outcome {trade_result_pct:.2%}). Nudging bias by {sentiment_nudge:.4f}.")
            elif sentiment_misaligned_with_profit:
                adjustment_direction = 0
                if first_sentiment_observed == "positive": adjustment_direction = -1 # Positive sentiment was wrong
                elif first_sentiment_observed == "negative": adjustment_direction = 1  # Negative sentiment was wrong
                sentiment_nudge = sentiment_influence_weight * confidence * adjustment_direction
                calculated_new_bias += sentiment_nudge
                logger.info(f"[BiasReflector] Sentiment MISALIGNED ({first_sentiment_observed} for {trade_direction}, outcome {trade_result_pct:.2%}). Nudging bias by {sentiment_nudge:.4f}.")

        # Zorg dat bias tussen 0 en 1 blijft
        calculated_new_bias = max(0.0, min(1.0, calculated_new_bias))

        current_data['bias'] = calculated_new_bias
        current_data['last_update'] = datetime.now().isoformat()
        if trade_result_pct is not None: # Alleen trade_count en profit updaten als er een trade was
            current_data['trade_count'] += 1
            current_data['total_profit_pct'] += trade_result_pct

        await asyncio.to_thread(self._save_bias_memory)
        logger.info(f"[BiasReflector] Bias voor {token}/{strategy_id} bijgewerkt naar {calculated_new_bias:.3f} (was {current_bias:.3f}).")


# Test sectie
async def run_test_bias_reflector():
    # Corrected path for .env when running this script directly
    # CORE_DIR and PROJECT_ROOT_DIR are defined at the top of the file
    dotenv_path = PROJECT_ROOT_DIR / '.env'
    if dotenv_path.exists():
        dotenv.load_dotenv(dotenv_path)
    else:
        logger.warning(f".env file not found at {dotenv_path} for __main__ test run.")

    # Setup basic logging for the test
    import sys # sys import for StreamHandler
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

    reflector = BiasReflector()

    # Clean up memory file before test if it exists
    if BIAS_MEMORY_FILE.exists():
        BIAS_MEMORY_FILE.unlink() # Use unlink for Path objects
        reflector = BiasReflector() # Re-initialize to ensure clean state

    # Test 1: Initial bias
    initial_bias = reflector.get_bias_score("BTC", "strategy1")
    logger.info(f"Test 1: Initial bias for BTC/strategy1: {initial_bias}")
    assert initial_bias == 0.5

    # Test 2: Update bias with no trade result (AI suggestion only)
    logger.info("\n--- Test 2: Update bias (AI suggestion only) ---")
    await reflector.update_bias("BTC", "strategy1", new_ai_bias=0.7, confidence=0.8)
    updated_bias_no_trade = reflector.get_bias_score("BTC", "strategy1")
    # Expected: (0.5 * 0.2) + (0.7 * 0.8) = 0.1 + 0.56 = 0.66
    logger.info(f"Test 2: Bias after AI suggestion (0.7 conf 0.8): {updated_bias_no_trade}")
    assert abs(updated_bias_no_trade - 0.66) < 0.001

    # Test 3: Update bias with profitable trade
    logger.info("\n--- Test 3: Update bias (Profitable Trade) ---")
    # Current bias is 0.66. AI suggests 0.7 (new_ai_bias). Confidence 0.8. Trade profit 5%.
    # trade_influence_factor = 0.8 * 0.1 = 0.08
    # target_bias = new_ai_bias (0.7) because 0.7 > 0.66
    # adjustment = (0.7 - 0.66) * 0.08 * (1 + min(0.05 * 10, 1)) = 0.04 * 0.08 * 1.5 = 0.0048
    # calculated_new_bias = 0.66 + 0.0048 = 0.6648
    await reflector.update_bias("BTC", "strategy1", new_ai_bias=0.7, confidence=0.8, trade_result_pct=0.05)
    updated_bias_profit = reflector.get_bias_score("BTC", "strategy1")
    logger.info(f"Test 3: Bias after profitable trade (+5%): {updated_bias_profit}")
    assert abs(updated_bias_profit - 0.6648) < 0.001


    # Test 4: Update bias with losing trade
    logger.info("\n--- Test 4: Update bias (Losing Trade) ---")
    # Current bias is 0.6648. AI suggests 0.3 (new_ai_bias). Confidence 0.6. Trade loss -3%.
    # trade_influence_factor = 0.6 * 0.1 = 0.06
    # target_bias = new_ai_bias (0.3) because 0.3 < 0.6648
    # adjustment = (0.3 - 0.6648) * 0.06 * (1 + min(abs(-0.03) * 10, 1)) = -0.3648 * 0.06 * 1.3 = -0.0284544
    # calculated_new_bias = 0.6648 - 0.0284544 = 0.6363456
    await reflector.update_bias("BTC", "strategy1", new_ai_bias=0.3, confidence=0.6, trade_result_pct=-0.03)
    updated_bias_loss = reflector.get_bias_score("BTC", "strategy1")
    logger.info(f"Test 4: Bias after losing trade (-3%): {updated_bias_loss}")
    assert abs(updated_bias_loss - 0.6363456) < 0.001

    # Test 5: Cooldown
    logger.info("\n--- Test 5: Cooldown Check ---")
    # Bias is 0.6363456. Last update was just now.
    await reflector.update_bias("BTC", "strategy1", new_ai_bias=0.9, confidence=0.9, trade_result_pct=0.10) # This should be blocked by cooldown
    bias_after_cooldown_attempt = reflector.get_bias_score("BTC", "strategy1")
    logger.info(f"Test 5: Bias after cooldown attempt: {bias_after_cooldown_attempt}")
    assert abs(bias_after_cooldown_attempt - updated_bias_loss) < 0.001 # Bias should not have changed

    # --- Sentiment-based Adjustment Tests ---
    # Reset memory for these specific tests for clarity, or use a new token/strategy
    if BIAS_MEMORY_FILE.exists():
        BIAS_MEMORY_FILE.unlink()
    reflector = BiasReflector() # Re-initialize

    logger.info("\n--- Test 6: Profitable LONG, POSITIVE sentiment (Aligned) ---")
    # Initial bias = 0.5. AI suggests 0.7 (bullish). Confidence 0.8. Profit 5%.
    # Standard adjustment:
    # trade_influence_factor = 0.8 * 0.1 = 0.08
    # target_bias = new_ai_bias (0.7) because 0.7 > 0.5
    # adjustment = (0.7 - 0.5) * 0.08 * (1 + min(0.05*10,1)) = 0.2 * 0.08 * 1.5 = 0.024
    # calculated_new_bias (before sentiment) = 0.5 + 0.024 = 0.524
    # Sentiment nudge:
    # sentiment_influence_weight = 0.05. first_sentiment_observed="positive" -> adjustment_direction = 1
    # sentiment_nudge = 0.05 * 0.8 * 1 = 0.04
    # final_calculated_new_bias = 0.524 + 0.04 = 0.564
    await reflector.update_bias("SENT", "long_positive_profit", new_ai_bias=0.7, confidence=0.8, trade_result_pct=0.05,
                                sentiment_data_status="present_and_not_empty",
                                first_sentiment_observed="positive",
                                trade_direction="long")
    final_bias_t6 = reflector.get_bias_score("SENT", "long_positive_profit")
    logger.info(f"Test 6: Bias = {final_bias_t6:.4f}")
    assert abs(final_bias_t6 - 0.564) < 0.0001

    logger.info("\n--- Test 7: Unprofitable LONG, POSITIVE sentiment (Misaligned) ---")
    # Initial bias = 0.5. AI suggests 0.7 (bullish). Confidence 0.8. Loss -3%.
    # Standard adjustment:
    # trade_influence_factor = 0.8 * 0.1 = 0.08
    # target_bias = new_ai_bias (0.7) because 0.7 > 0.5
    # adjustment = (0.7 - 0.5) * 0.08 * (1 + min(0.03*10,1)) = 0.2 * 0.08 * 1.3 = 0.0208
    # calculated_new_bias (before sentiment) = 0.5 + 0.0208 = 0.5208 -> This is wrong logic for loss, let's re-evaluate.
    # If loss: target_bias = new_ai_bias (0.7) if new_ai_bias < current_bias (0.5) -> False. So, target_bias = 0.5 - 0.05 = 0.45
    # adjustment = (0.45 - 0.5) * 0.08 * 1.3 = -0.05 * 0.08 * 1.3 = -0.0052
    # calculated_new_bias (before sentiment) = 0.5 - 0.0052 = 0.4948
    # Sentiment nudge (misaligned):
    # first_sentiment_observed="positive" -> adjustment_direction = -1
    # sentiment_nudge = 0.05 * 0.8 * (-1) = -0.04
    # final_calculated_new_bias = 0.4948 - 0.04 = 0.4548
    await reflector.update_bias("SENT", "long_positive_loss", new_ai_bias=0.7, confidence=0.8, trade_result_pct=-0.03,
                                sentiment_data_status="present_and_not_empty",
                                first_sentiment_observed="positive",
                                trade_direction="long")
    final_bias_t7 = reflector.get_bias_score("SENT", "long_positive_loss")
    logger.info(f"Test 7: Bias = {final_bias_t7:.4f}")
    assert abs(final_bias_t7 - 0.4548) < 0.0001


    logger.info("\n--- Test 8: Profitable SHORT, NEGATIVE sentiment (Aligned) ---")
    # Initial bias = 0.5. AI suggests 0.3 (bearish). Confidence 0.8. Profit 5% (for short).
    # Standard adjustment (profit):
    # trade_influence_factor = 0.8 * 0.1 = 0.08
    # target_bias = new_ai_bias (0.3) because 0.3 < 0.5
    # adjustment = (0.3 - 0.5) * 0.08 * (1 + min(0.05*10,1)) = -0.2 * 0.08 * 1.5 = -0.024
    # calculated_new_bias (before sentiment) = 0.5 - 0.024 = 0.476
    # Sentiment nudge (aligned):
    # first_sentiment_observed="negative" -> adjustment_direction = -1
    # sentiment_nudge = 0.05 * 0.8 * (-1) = -0.04
    # final_calculated_new_bias = 0.476 - 0.04 = 0.436
    await reflector.update_bias("SENT", "short_negative_profit", new_ai_bias=0.3, confidence=0.8, trade_result_pct=0.05,
                                sentiment_data_status="present_and_not_empty",
                                first_sentiment_observed="negative",
                                trade_direction="short")
    final_bias_t8 = reflector.get_bias_score("SENT", "short_negative_profit")
    logger.info(f"Test 8: Bias = {final_bias_t8:.4f}")
    assert abs(final_bias_t8 - 0.436) < 0.0001

    logger.info("\n--- Test 9: Unprofitable SHORT (loss for short means price went up), NEGATIVE sentiment (Misaligned) ---")
    # Initial bias = 0.5. AI suggests 0.3 (bearish). Confidence 0.8. Loss -3% (for short, means price increased).
    # Standard adjustment (loss):
    # trade_influence_factor = 0.8 * 0.1 = 0.08
    # target_bias = new_ai_bias (0.3) if new_ai_bias < current_bias (0.5) -> True. So target_bias = 0.3
    # adjustment = (0.3 - 0.5) * 0.08 * (1 + min(0.03*10,1)) = -0.2 * 0.08 * 1.3 = -0.0208
    # calculated_new_bias (before sentiment) = 0.5 - 0.0208 = 0.4792
    # Sentiment nudge (misaligned):
    # first_sentiment_observed="negative" -> adjustment_direction = 1 (negative sentiment was wrong for a short that lost value)
    # sentiment_nudge = 0.05 * 0.8 * 1 = 0.04
    # final_calculated_new_bias = 0.4792 + 0.04 = 0.5192
    await reflector.update_bias("SENT", "short_negative_loss", new_ai_bias=0.3, confidence=0.8, trade_result_pct=-0.03,
                                sentiment_data_status="present_and_not_empty",
                                first_sentiment_observed="negative",
                                trade_direction="short")
    final_bias_t9 = reflector.get_bias_score("SENT", "short_negative_loss")
    logger.info(f"Test 9: Bias = {final_bias_t9:.4f}")
    assert abs(final_bias_t9 - 0.5192) < 0.0001


    logger.info("\n--- Test 10: Profitable LONG, NEUTRAL sentiment ---")
    # Initial bias = 0.5. AI suggests 0.7 (bullish). Confidence 0.8. Profit 5%.
    # Standard adjustment: 0.524 (same as Test 6 before sentiment)
    # Sentiment nudge (neutral):
    # first_sentiment_observed="neutral" -> adjustment_direction = 0
    # sentiment_nudge = 0.05 * 0.8 * 0 = 0.0
    # final_calculated_new_bias = 0.524 + 0.0 = 0.524
    await reflector.update_bias("SENT", "long_neutral_profit", new_ai_bias=0.7, confidence=0.8, trade_result_pct=0.05,
                                sentiment_data_status="present_and_not_empty",
                                first_sentiment_observed="neutral",
                                trade_direction="long")
    final_bias_t10 = reflector.get_bias_score("SENT", "long_neutral_profit")
    logger.info(f"Test 10: Bias = {final_bias_t10:.4f}")
    assert abs(final_bias_t10 - 0.524) < 0.0001

    logger.info("\n--- Test 11: Profitable LONG, sentiment data ABSENT ---")
    # Initial bias = 0.5. AI suggests 0.7 (bullish). Confidence 0.8. Profit 5%.
    # Standard adjustment: 0.524 (same as Test 6 before sentiment)
    # Sentiment nudge: Should not occur.
    # final_calculated_new_bias = 0.524
    await reflector.update_bias("SENT", "long_no_sentiment_profit", new_ai_bias=0.7, confidence=0.8, trade_result_pct=0.05,
                                sentiment_data_status="absent", # Crucial part
                                first_sentiment_observed=None,
                                trade_direction="long")
    final_bias_t11 = reflector.get_bias_score("SENT", "long_no_sentiment_profit")
    logger.info(f"Test 11: Bias = {final_bias_t11:.4f}")
    assert abs(final_bias_t11 - 0.524) < 0.0001


    logger.info("\nAll BiasReflector tests passed.")
    # Clean up memory file after test
    if BIAS_MEMORY_FILE.exists():
        BIAS_MEMORY_FILE.unlink()

if __name__ == '__main__':
    asyncio.run(run_test_bias_reflector())
