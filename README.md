# DUO-AI Trading Bot

## Doel
Een zelflerende, AI-gestuurde crypto trading bot, gebouwd bovenop Freqtrade, die gebruik maakt van GPT- en Grok-reflecties, technische indicatoren, patroonherkenning, en live Bitvavo-executie. Het doel is om ETH-stacks te vergroten via slimme entries en exits.

## Projectstructuur
-   `/core/`: Bevat alle eigen AI-modules (reflectie, bias, confidence, besluitvorming, patroonherkenning, etc.).
-   `/strategies/`: Bevat de Freqtrade strategie-scripts, zoals `DUOAI_Strategy.py`.
-   `/data/`: Voor trainingsdata en cachebestanden (bijv. Grok Live Search cache). Bevat nu ook `/data/models/` voor getrainde CNN-modellen.
-   `/memory/`: Opslag voor de geleerde parameters van de AI-modules (bias, confidence, strategie-parameters, reflectielogs).
-   `/notebooks/`: Bestemd voor AI-modelontwikkeling, training, prompt-experimenten en diepere data-analyse.
-   `/tests/`: Voor unit tests en integratietests (nog op te zetten).
-   `/config/`: Bevat de Freqtrade configuratie (`config.json`).
-   `main.py`: Het hoofdentriepunt voor de AI-coördinatie.
-   `requirements.txt`: Specificeert alle benodigde Python-pakketten.
-   `.env`: Bevat gevoelige API-sleutels. Wordt niet gecommit naar Git.

## Installatie

1.  **Kloon de repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)<jouw-github-gebruikersnaam>/DUO-AI-TradingBot.git
    cd DUO-AI-TradingBot
    ```

2.  **Maak een virtuele omgeving aan (aanbevolen):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Op Linux/macOS
    # of `.\venv\Scripts\activate` op Windows
    ```

3.  **Installeer Freqtrade en alle Python-afhankelijkheden:**
    ```bash
    pip install -r requirements.txt
    ```
    Freqtrade wordt nu geïnstalleerd als een Python-pakket. De eerder handmatig aangemaakte `/freqtrade/` map is verwijderd, aangezien deze niet nodig is voor de Freqtrade core code.

## Configuratie

1.  **API-sleutels (`.env`):**
    Creëer een bestand genaamd `.env` in de hoofdmap van het project. Vul dit bestand met je API-sleutels voor OpenAI, Grok en Bitvavo. **Dit bestand mag NOOIT naar GitHub worden gecommit.**
    ```
    OPENAI_API_KEY="sk-YOUR_OPENAI_API_KEY"
    OPENAI_MODEL="gpt-4o"
    GROK_API_KEY="grok-YOUR_GROK_API_KEY"
    GROK_MODEL="grok-1"
    GROK_LIVE_SEARCH_API_URL="[https://api.x.ai/v1/live-search](https://api.x.ai/v1/live-search)" # Pas dit aan naar het officiële Grok Live Search API endpoint zodra bekend
    BITVAVO_API_KEY="YOUR_BITVAVO_API_KEY"
    BITVAVO_SECRET_KEY="YOUR_BITVAVO_SECRET_KEY"
    ```

2.  **Freqtrade Configuratie (`config/config.json`):**
    De basisconfiguratie is aanwezig. De `pair_whitelist` in `config/config.json` bevat de volgende actieve handelsparen: `"ETH/EUR", "BTC/EUR", "ZEN/EUR", "WETH/USDT", "USDC/USDT", "WBTC/USDT", "LINK/USDT", "UNI/USDT", "ZEN/BTC", "LSK/BTC", "ETH/BTC"`. Dit is de definitieve lijst die momenteel door de bot wordt overwogen. De API-sleutels in `config.json` zijn placeholders en zullen door Freqtrade via de omgevingsvariabelen (geladen via `.env`) worden overschreven in live-modus.
    **Belangrijk:** Controleer de beschikbaarheid van paren zoals `LSK/BTC` op Bitvavo, aangezien niet alle exchanges alle cross-paren ondersteunen.

## Gebruik

1.  **Start de bot (backtest of dry-run):**
    Activeer eerst je virtuele omgeving.

    * **Backtesting (aanbevolen voor testen en leren):**
        ```bash
        freqtrade backtesting --strategy DUOAI_Strategy -c config/config.json --timerange=20240101-20250101 --export-filename user_data/backtest_results/DUOAI_Strategy_backtest.json
        ```
        De AI-reflectie- en leermechanismen zullen actief zijn tijdens backtesting en de interne AI-parameters bijwerken.

    * **Dry Run (simulatie van live trading):**
        ```bash
        freqtrade trade --config config/config.json --strategy DUOAI_Strategy --dry-run
        ```
        Dit simuleert live trading, inclusief AI-besluitvorming en -reflectie, zonder echt kapitaal te riskeren.

    * **Live Trading (UITERST VOORZICHTIG):**
        ```bash
        freqtrade trade --config config/config.json --strategy DUOAI_Strategy
        ```
        **WAARSCHUWING:** Zet `dry_run: false` in `config.json` en begrijp de risico's van live trading voordat je deze modus gebruikt.

## Modules Overzicht

De bot is opgebouwd uit de volgende kern-AI-modules in de `/core/` map:

-   `gpt_reflector.py`: Communiceert met de OpenAI GPT API voor reflectie.
-   `grok_reflector.py`: Communiceert met de Grok API voor reflectie.
-   `grok_sentiment_fetcher.py`: Haalt live nieuws- en sentimentdata op via Grok Live Search (hypothetisch API).
-   `cnn_patterns.py`: Detecteert candlestick- en grafiekpatronen. Deze module combineert traditionele, regelgebaseerde patroonherkenning met geavanceerde Deep Learning (CNN) voorspellingen. Voor de CNN-voorspellingen worden per patroontype specifieke, vooraf getrainde modellen en bijbehorende scalers geladen (indien beschikbaar in `/data/models/`). De module kan hierdoor numerieke scores (waarschijnlijkheden) genereren voor gedetecteerde patronen. **Belangrijk:** De effectiviteit van de CNN-gebaseerde voorspellingen is direct afhankelijk van de aanwezigheid en kwaliteit van de getrainde modellen en de labels die tijdens de training zijn gebruikt. Het ontwikkelen en onderhouden van hoogwaardige modellen is een essentieel en voortdurend proces.
-   `prompt_builder.py`: Genereert gedetailleerde AI-prompts met marktgegevens, patronen en sentiment.
-   `reflectie_lus.py`: De centrale AI-reflectiecyclus, coördineert promptgeneratie, AI-aanroepen en het loggen van reflecties.
-   `reflectie_analyser.py`: Analyseert reflectielogs en haalt prestatiegegevens direct uit Freqtrade's database om bias-scores en mutatievoorstellen te genereren.
-   `bias_reflector.py`: Beheert de leerbare voorkeur/bias van de strategie per token/strategie.
-   `confidence_engine.py`: Beheert de leerbare confidence score en past `maxTradeRiskPct` dynamisch aan.
-   `cooldown_tracker.py`: Beheert AI-specifieke cooldown-periodes per token/strategie, aanvullend op Freqtrade's ingebouwde cooldowns.
-   `entry_decider.py`: Neemt AI-gestuurde entry-besluiten op basis van consensus, confidence, bias en patronen, inclusief AI-specifieke cooldowns en `timeOfDayEffectiveness`.
-   `exit_optimizer.py`: Neemt AI-gestuurde exit-besluiten en optimaliseert dynamisch de trailing stop loss.
-   `pre_trainer.py`: Module voor pre-learning en data-voorbereiding. Haalt historische data en trade-outcomes direct uit Freqtrade's database. Kan ook een basis CNN-model trainen en opslaan.
-   `strategy_manager.py`: Beheert strategieparameters, prestaties (haalt uit Freqtrade DB) en mutatievoorstellen.
-   `ai_activation_engine.py`: Trigger-engine voor AI-reflectie bij specifieke gebeurtenissen.
-   `interval_selector.py`: Detecteert en beheert de beste timeframe/interval voor AI-analyse.
-   `params_manager.py`: Centrale manager voor alle dynamisch lerende variabelen.
-   `trade_logger.py`: Deze module is **verwijderd** uit de kern AI-workflow. Alle AI-analyses en modeltraining baseren zich **exclusief op Freqtrade's interne database** voor trade-historie en -data. Het bestand kan nog aanwezig zijn in de repository voor historische doeleinden of sporadische, handmatige data-export, maar speelt geen actieve rol meer in de geautomatiseerde botlogica.

## Belangrijke Operationele Aspecten en Huidige Status

### CNN Modellen: Functionaliteit en Doorlopende Ontwikkeling
De `cnn_patterns.py` module kan, zoals eerder genoemd, zowel regelgebaseerde als Deep Learning CNN-voorspellingen uitvoeren. Hoewel `pre_trainer.py` een basis CNN kan trainen en `cnn_patterns.py` modellen kan laden, is de **volledige pijplijn van modelontwikkeling, uitgebreide training, validatie en optimalisatie essentieel** en een voortdurende taak. De huidige CNN-integratie is een fundament; de daadwerkelijke voorspellende kracht hangt af van robuust getrainde modellen. De `cnn_patterns.py` retourneert numerieke scores, maar de actieve toepassing van `cnnPatternWeight` als een leerbare multiplier in `entry_decider.py` en `exit_optimizer.py` om de CNN-voorspellingen te wegen, is een toekomstige verbetering.

### Dynamische Configuratie en Runtime Aanpassingen
De AI-modules binnen dit project kunnen adviezen genereren voor diverse strategieparameters, die veelal intern gebruikt worden. Echter, voor Freqtrade's kernconfiguratie (`config/config.json`) geldt het volgende:
*   **Herstart Vereist voor `config.json` Wijzigingen:** Aanpassingen aan fundamentele Freqtrade-instellingen (zoals `pair_whitelist`, `stake_amount`, `exchange` details, etc.) worden **niet dynamisch tijdens runtime door Freqtrade overgenomen.** Indien de AI advies genereert voor dergelijke parameters, of indien u deze handmatig wijzigt, is een **volledige herstart van de Freqtrade bot noodzakelijk**.
*   **Aanpak voor Gevoelige Parameters:** Voor kritische, door Freqtrade beheerde parameters, zoals `slippageTolerancePct` (indien in `config.json`), blijft het AI-advies een aanbeveling voor handmatige aanpassing en herstart.

### `preferredPairs` en Dynamisch Pair Management
*   **Leerlogica:** De logica voor het dynamisch identificeren van `preferredPairs` is geïmplementeerd in `core/ai_optimizer.py`.
*   **Koppeling met `pair_whitelist`:** Freqtrade laadt zijn actieve handelsparen uit `config/config.json` bij het opstarten. Een door `ai_optimizer.py` gegenereerde lijst van `preferredPairs` wordt **niet automatisch tijdens runtime gesynchroniseerd**.
*   **Herstart Vereist:** Om geleerde `preferredPairs` actief te maken, dient de `pair_whitelist` in `config/config.json` handmatig bijgewerkt te worden, gevolgd door een **herstart van de Freqtrade bot**.

### Data Logging en Analyse
Voor alle AI-gestuurde analyses, prestatie-evaluaties en het trainen van modellen wordt nu **exclusief gebruik gemaakt van Freqtrade's interne database**. Dit waarborgt een consistente en betrouwbare databron.

### Teststructuur en Kwaliteitsborging
Hoewel de `if __name__ == "__main__":` blokken basale tests per module bieden, is de ontwikkeling van een **formele, geautomatiseerde testsuite (bijvoorbeeld met `pytest`) cruciaal** voor de robuustheid en betrouwbaarheid van de bot. Dit staat hoog op de prioriteitenlijst voor toekomstige ontwikkeling. De `/tests/` map is hiervoor gereserveerd.

## Toekomstige Ontwikkeling
-   Verdere verfijning en training van CNN-modellen voor specifieke patronen.
-   Implementatie van `cnnPatternWeight` voor het wegen van CNN-voorspellingen in de besluitvormingslogica.
-   Ontwikkeling van de formele `pytest` testsuite voor uitgebreide kwaliteitsborging.
-   Onderzoek naar geavanceerdere methoden voor dynamisch pairlist management binnen Freqtrade.
-   Uitbreiding van monitoring en visualisatie (bijvoorbeeld via een AI-gestuurde GUI).
