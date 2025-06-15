import json
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

COMMON_TIMEFRAMES = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w']

def is_valid_timeframe(timeframe_str: str) -> bool:
    """
    Validates if the given string is a common Freqtrade timeframe.
    """
    return timeframe_str in COMMON_TIMEFRAMES

def load_and_validate_config(config_path: str, env_path: str) -> dict:
    """
    Loads and validates the configuration from config.json and .env files.

    Args:
        config_path: Path to the config.json file.
        env_path: Path to the .env file.

    Returns:
        The loaded and validated Freqtrade configuration dictionary.

    Raises:
        ValueError: If any validation fails.
    """
    logger.info(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}", exc_info=True)
        raise ValueError(f"Configuration file not found at {config_path}")
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {config_path}", exc_info=True)
        raise ValueError(f"Error decoding JSON from {config_path}")

    logger.info(f"Loading environment variables from: {env_path}")
    if not Path(env_path).exists():
        logger.error(f"Environment file not found at {env_path}", exc_info=True)
        raise ValueError(f"Environment file not found at {env_path}")
    load_dotenv(dotenv_path=env_path)

    # Validate config.json contents
    logger.debug("Validating config.json contents...")
    if not isinstance(config.get('exchange', {}).get('name'), str) or not config.get('exchange', {}).get('name'):
        raise ValueError("`exchange.name` must be a non-empty string in config.json")
    logger.debug("exchange.name validation passed.")

    pair_whitelist = config.get('exchange', {}).get('pair_whitelist')
    if not isinstance(pair_whitelist, list) or not pair_whitelist or not all(isinstance(p, str) for p in pair_whitelist):
        raise ValueError("`exchange.pair_whitelist` must be a non-empty list of strings in config.json")
    logger.debug("exchange.pair_whitelist validation passed.")

    timeframe = config.get('timeframe')
    if not isinstance(timeframe, str) or not timeframe:
        raise ValueError("`timeframe` must be a non-empty string in config.json")
    if not is_valid_timeframe(timeframe):
        raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of {COMMON_TIMEFRAMES}")
    logger.debug("timeframe validation passed.")

    user_data_dir = config.get('user_data_dir')
    if not isinstance(user_data_dir, str) or not user_data_dir:
        raise ValueError("`user_data_dir` must be a non-empty string in config.json")
    logger.debug("user_data_dir validation passed.")

    data_dir = config.get('data_dir')
    if not isinstance(data_dir, str) or not data_dir:
        raise ValueError("`data_dir` must be a non-empty string in config.json")
    # Typically user_data_dir + '/data', but can be different
    # No direct validation against user_data_dir here, depends on setup.
    logger.debug("data_dir validation passed.")

    # Optional: logfile configuration
    if 'logfile' in config or 'logging' in config: # Freqtrade might have general logging config
        logger.debug("Logfile configuration found in config.json.")
    else:
        logger.warning("No specific 'logfile' or 'logging' configuration found in config.json. Using default logging setup.")


    # Validate .env contents
    logger.debug("Validating .env contents...")
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("`OPENAI_API_KEY` not found or empty in .env file")
    logger.debug("OPENAI_API_KEY validation passed.")

    grok_api_key = os.getenv('GROK_API_KEY')
    if not grok_api_key:
        raise ValueError("`GROK_API_KEY` not found or empty in .env file")
    logger.debug("GROK_API_KEY validation passed.")

    # Conditional validation for Bitvavo keys
    if config.get('exchange', {}).get('name', '').lower() == 'bitvavo':
        bitvavo_api_key = os.getenv('BITVAVO_API_KEY')
        bitvavo_secret_key = os.getenv('BITVAVO_SECRET_KEY')
        if not bitvavo_api_key:
            raise ValueError("`BITVAVO_API_KEY` not found or empty in .env file (Bitvavo exchange configured)")
        if not bitvavo_secret_key:
            raise ValueError("`BITVAVO_SECRET_KEY` not found or empty in .env file (Bitvavo exchange configured)")
        logger.debug("Bitvavo API keys validation passed (Bitvavo exchange configured).")
    else:
        # Check for existence if not Bitvavo, but don't fail if they're not there
        # as they might be used for other purposes or simply present.
        if not os.getenv('BITVAVO_API_KEY'):
            logger.warning("`BITVAVO_API_KEY` not found in .env file. This is okay if not using Bitvavo.")
        if not os.getenv('BITVAVO_SECRET_KEY'):
            logger.warning("`BITVAVO_SECRET_KEY` not found in .env file. This is okay if not using Bitvavo.")


    logger.info("Configuration and environment variables loaded and validated successfully.")
    return config
