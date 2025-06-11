# scripts/fetch_market_data.py
import argparse
import logging
import os
import sys
from datetime import datetime

# Adjust the path to import from the core directory
# This assumes 'scripts' is at the same level as 'core' and 'config'
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

try:
    from core.market_data_fetcher import BinanceDataFetcher
except ImportError:
    print("Error: Could not import BinanceDataFetcher. Ensure you are running from the project root"
          " or that the paths are set up correctly.")
    sys.exit(1)

# Setup basic logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Fetch historical market data from Binance.")
    parser.add_argument(
        "--symbols",
        type=str,
        required=True,
        help="Comma-separated list of trading symbols (e.g., BTCUSDT,ETHUSDT)."
    )
    parser.add_argument(
        "--intervals",
        type=str,
        required=True,
        help="Comma-separated list of K-line intervals (e.g., 1m,5m,1h,1d)."
    )
    parser.add_argument(
        "--start_date",
        type=str,
        required=True,
        help="Start date for data fetching (format: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)."
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=None, # Handled by fetch_and_save_historical_data if None
        help="Optional: End date for data fetching (format: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS). Defaults to current time if not provided."
    )

    args = parser.parse_args()

    symbols_list = [s.strip().upper() for s in args.symbols.split(',')]
    intervals_list = [i.strip() for i in args.intervals.split(',')]

    try:
        fetcher = BinanceDataFetcher()
    except ValueError as e:
        logger.error(f"Failed to initialize BinanceDataFetcher: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during fetcher initialization: {e}")
        sys.exit(1)


    logger.info(f"Starting data fetching process...")
    logger.info(f"Symbols: {symbols_list}")
    logger.info(f"Intervals: {intervals_list}")
    logger.info(f"Start Date: {args.start_date}")
    logger.info(f"End Date: {args.end_date if args.end_date else 'Now'}")

    for symbol in symbols_list:
        for interval in intervals_list:
            logger.info(f"--- Processing: Symbol={symbol}, Interval={interval} ---")
            try:
                fetcher.fetch_and_save_historical_data(
                    symbol=symbol,
                    interval=interval,
                    start_date_str=args.start_date,
                    end_date_str=args.end_date
                )
                logger.info(f"Successfully processed Symbol={symbol}, Interval={interval}")
            except Exception as e:
                logger.error(f"Error processing Symbol={symbol}, Interval={interval}: {e}", exc_info=True)
            logger.info(f"--- Finished: Symbol={symbol}, Interval={interval} ---")

    logger.info("All fetching tasks complete.")

if __name__ == "__main__":
    main()
