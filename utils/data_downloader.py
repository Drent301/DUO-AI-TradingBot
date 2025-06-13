import subprocess
import os
import pathlib

def download_data(pairs: list[str], timeframes: list[str], exchange: str, data_dir: str = "user_data/data/binance", days: int = None, since: str = None):
    """
    Downloads data using Freqtrade.
    """
    # Ensure data_dir is a Path object for easier manipulation
    data_dir_path = pathlib.Path(data_dir)

    # Freqtrade's --datadir is the parent of the 'data' folder.
    # e.g., if data_dir is /some/path/user_data/data/exchange_name
    # then freqtrade_datadir should be /some/path/user_data
    try:
        # Find the 'data' directory and get its parent
        # This assumes a structure like .../user_data/data/exchange
        parts = list(data_dir_path.parts)
        data_index = parts.index('data')
        freqtrade_datadir = str(pathlib.Path(*parts[:data_index]))
        if not freqtrade_datadir and len(parts) > data_index +1: # Handle cases like 'user_data/data/binance'
             freqtrade_datadir = str(pathlib.Path(*parts[:data_index+1])) # then it should be user_data
             # Correction: if parts = ['user_data', 'data', 'binance'], data_index = 1. parts[:1] = ['user_data']
             # So freqtrade_datadir should be user_data if the structure is user_data/data/exchange
             # Let's re-evaluate the logic for freqtrade_datadir.
             # If data_dir_path = "user_data/data/binance", we want "user_data"
             # If data_dir_path = "/abs/path/to/user_data/data/binance", we want "/abs/path/to/user_data"

        # A more robust way to find the parent 'user_data' or equivalent directory for --datadir
        current_path = data_dir_path.resolve() # Get absolute path
        freqtrade_datadir_path = None
        # Iterate upwards from data_dir_path until a directory containing 'data' is found,
        # then go one level up from that 'data' directory's parent if possible,
        # or assume data_dir_path's parent if 'data' is not found.
        # This logic is tricky. A simpler assumption: freqtrade --datadir is the directory *containing* the 'data' directory.
        # So if data_dir = 'user_data/data/binance', then user_data/data is where freqtrade looks,
        # and --datadir should be 'user_data'.

        # Revised logic for freqtrade_datadir:
        # We expect data_dir to be like ".../some_base_dir/data/exchange_name"
        # Freqtrade's --datadir should then be ".../some_base_dir"

        # Let's assume data_dir is '.../parent_of_data_folder/data/exchange_name'
        # Then freqtrade_datadir is '.../parent_of_data_folder'
        # Example: data_dir = "user_data/data/binance" -> freqtrade_datadir = "user_data"
        # Example: data_dir = "/opt/freqtrade/user_data/data/kraken" -> freqtrade_datadir = "/opt/freqtrade/user_data"

        # We need to find the parent of the directory named 'data' in the path.
        path_parts = data_dir_path.parts
        try:
            data_folder_index = path_parts.index("data")
            # The directory for --datadir is the one *before* 'data' in the path
            # e.g. ('user_data', 'data', 'binance'), data_folder_index is 1. We need path_parts[0] -> 'user_data'
            if data_folder_index > 0:
                freqtrade_datadir = str(pathlib.Path(*path_parts[:data_folder_index]))
            else: # 'data' is the first component, or not present in the way expected.
                  # This case is ambiguous. Let's default to data_dir_path.parent if 'data' is top level or not found.
                  # However, freqtrade expects --datadir to be the root of user_data.
                  # A common convention is that data_dir is "path_to_freqtrade/user_data/data/exchange".
                  # So, data_dir.parent.parent should be "path_to_freqtrade/user_data".
                freqtrade_datadir = str(data_dir_path.parent.parent)

        except ValueError:
            # If 'data' is not in the path, this indicates an unexpected data_dir structure.
            # For example, if data_dir is just 'my_data/binance'.
            # In this scenario, freqtrade might expect --datadir to be 'my_data'.
            # This is a fallback, might need adjustment based on actual Freqtrade behavior for non-standard dirs.
            print(f"Warning: 'data' directory not found in path {data_dir_path}. Using {data_dir_path.parent} as Freqtrade datadir. This might be incorrect.")
            freqtrade_datadir = str(data_dir_path.parent)


    except IndexError:
        print(f"Error: Could not determine Freqtrade datadir from {data_dir_path}. Please ensure it follows a structure like '.../user_data/data/exchange_name'")
        return
    except ValueError: # Handles cases where 'data' is not in parts
        print(f"Error: The 'data' directory was not found in the path '{data_dir_path}'. Cannot determine appropriate --datadir for Freqtrade.")
        return


    print(f"Using Freqtrade --datadir: {freqtrade_datadir}")

    # Create the specific data directory if it doesn't exist
    try:
        os.makedirs(data_dir_path, exist_ok=True)
        print(f"Ensured data directory exists: {data_dir_path}")
    except OSError as e:
        print(f"Error creating directory {data_dir_path}: {e}")
        return

    for pair in pairs:
        for timeframe in timeframes:
            command = [
                "freqtrade", "download-data",
                "--exchange", exchange.lower(),
                "-p", pair,
                "-t", timeframe,
                "--datadir", freqtrade_datadir
            ]

            if days is not None:
                command.extend(["--days", str(days)])
            elif since is not None:
                command.extend(["--since", since])
            else:
                # Default to a small period if neither days nor since is provided, e.g., 7 days
                # Or raise an error, or make one of them mandatory in the function signature.
                # For now, let's assume one will be provided by the caller logic,
                # or Freqtrade has its own default if not specified.
                # The problem description implies 'days' will be used for the 5-year requirement.
                pass

            print(f"Executing command: {' '.join(command)}")
            try:
                process = subprocess.run(command, check=True, capture_output=True, text=True)
                print(f"Successfully downloaded data for {pair} - {timeframe}")
                if process.stdout:
                    print("Stdout:")
                    print(process.stdout)
                if process.stderr:
                    print("Stderr:")
                    print(process.stderr)
            except subprocess.CalledProcessError as e:
                print(f"Error downloading data for {pair} - {timeframe} on {exchange}.")
                print(f"Command failed: {' '.join(e.cmd)}")
                print(f"Return code: {e.returncode}")
                if e.stdout:
                    print("Stdout:")
                    print(e.stdout)
                if e.stderr:
                    print("Stderr:")
                    print(e.stderr)
            except FileNotFoundError:
                print("Error: 'freqtrade' command not found. Make sure Freqtrade is installed and in your PATH.")
                return # Stop further processing if freqtrade is not found

if __name__ == "__main__":
    # As per requirements for the main block
    sample_pairs = ["ZEN/BTC", "LSK/BTC", "ETH/BTC", "ETH/EUR"]
    sample_timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
    sample_exchange = "binance"
    # Freqtrade stores data by default in user_data/data/<exchange_name>
    # So, if we want data in user_data/data/binance,
    # data_dir parameter should be "user_data/data/binance"
    sample_data_dir = "user_data/data/binance"
    five_years_in_days = 5 * 365

    print(f"Starting data download for {sample_exchange}...")
    download_data(
        pairs=sample_pairs,
        timeframes=sample_timeframes,
        exchange=sample_exchange,
        data_dir=sample_data_dir,
        days=five_years_in_days
    )
    print("Data download process finished.")
