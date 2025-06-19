FROM freqtradeorg/freqtrade:stable

WORKDIR /freqtrade

COPY requirements.txt /freqtrade/requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --force-reinstall -r /freqtrade/requirements.txt --break-system-packages

COPY ./user_data /freqtrade/user_data/
COPY ./core /freqtrade/core/
COPY ./strategies /freqtrade/user_data/strategies/
COPY ./utils /freqtrade/utils/
COPY ./config /freqtrade/config/
