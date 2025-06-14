# DUO-AI Trading Bot

## Doel
Een zelflerende, AI-gestuurde crypto trading bot, gebouwd bovenop Freqtrade, die gebruik maakt van GPT- en Grok-reflecties, technische indicatoren, patroonherkenning, en live Bitvavo-executie. Het doel is om ETH-stacks te vergroten via slimme entries en exits.

## Projectstructuur
-   `/core/`: Bevat alle eigen AI-modules (reflectie, bias, confidence, besluitvorming, patroonherkenning, etc.).
-   `/strategies/`: Bevat de Freqtrade strategie-scripts, zoals `DUOAI_Strategy.py`.
-   `/data/`: Voor trainingsdata en cachebestanden (bijv. Grok Live Search cache). Bevat nu ook `/data/models/` voor getrainde CNN-modellen.
-   `/memory/`: Opslag voor de geleerde parameters van de AI-modules (bias, confidence, strategie-parameters, reflectielogs).
-   `/notebooks/`: Bestemd voor AI-modelontwikkeling, training, prompt-experimenten en diepere data-analyse.
-   `/tests/`: Voor unit tests en integratietests (nu met uitgebreide Pytest tests).
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
    De basisconfiguratie is aangepast voor gebruik met **Binance** als de primaire exchange. De volgende `pair_whitelist` in `config/config.json` is een voorbeeld van een geconfigureerde lijst:
    ```json
    [
        "LSK/BTC",
        "ETH/BTC",
        "ZEN/BTC"
    ]
    ```
    De API-sleutels in `config.json` (`exchange.key` en `exchange.secret`) zijn ingesteld op placeholders zoals `"ENV_BINANCE_API_KEY"` en `"ENV_BINANCE_SECRET_KEY"`. Deze worden door Freqtrade automatisch geladen uit de corresponderende omgevingsvariabelen (die je in je `.env` bestand of direct in je omgeving kunt instellen) wanneer de bot in live- of dry-run modus draait.
    **Belangrijk:** Zorg ervoor dat de paren in de `pair_whitelist` daadwerkelijk beschikbaar zijn op Binance en dat je API-sleutels correct zijn ingesteld in je omgeving voor live trading.

3.  **Historische Data Downloaden**:
    Voor het backtesten en trainen van de AI modellen is het essentieel om voldoende historische marktdata te hebben.
    Het script `utils/data_downloader.py` is geconfigureerd om 5 jaar aan historische data te downloaden voor de in `config/config.json` gespecificeerde `pair_whitelist` en de volgende timeframes: `1m, 5m, 15m, 1h, 4h, 1d`.
    De data wordt opgeslagen in de `user_data/data/<exchange_naam>` map (bijv. `user_data/data/binance`).

    **Vereisten:**
    *   Zorg ervoor dat Freqtrade geïnstalleerd is en correct geconfigureerd in je `PATH`.
    *   Het `config/config.json` bestand moet de juiste `exchange` (Binance) en `pair_whitelist` bevatten.

    **Commando (uitvoeren vanuit de project root directory):**
    ```bash
    python utils/data_downloader.py
    ```

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

    * **Starten van de AI Reflectie Lus (onafhankelijk proces):**
        De AI reflectie lus kan als een apart, continu proces worden gestart om periodiek AI-reflecties uit te voeren. Dit is nuttig voor het continu leren en aanpassen van de AI-modellen, los van directe Freqtrade trade-executie.
        ```bash
        python run_pipeline.py --start-reflection-loop --reflection-symbols ETH/USDT,BTC/USDT --reflection-interval 60
        ```
        Pas `--reflection-symbols` aan met de gewenste handelsparen (komma-gescheiden) en `--reflection-interval` met het gewenste interval in minuten.

## Modules Overzicht

De bot is opgebouwd uit de volgende kern-AI-modules in de `/core/` map:

-   `gpt_reflector.py`: Communiceert met de OpenAI GPT API voor reflectie.
-   `grok_reflector.py`: Communiceert met de Grok API voor reflectie.
-   `grok_sentiment_fetcher.py`: Haalt live nieuws- en sentimentdata op via Grok 3's live data/search functionaliteit (hypothetisch API), die vervolgens wordt opgenomen in de AI prompts.
-   `cnn_patterns.py`: Voert patroonherkenning uit. Deze module **combineert regelgebaseerde detectie (`_get_price_action_features`) met Deep Learning CNN-voorspellingen** (indien een getraind model en bijbehorende scaler beschikbaar zijn in `/data/models/`). Voor de CNN-features maakt het gebruik van basisindicatoren en candlestickpatronen die door `DUOAI_Strategy.py` worden aangeleverd. Hierdoor kan de module numerieke scores (bijv. waarschijnlijkheden) voor patronen genereren. De `detect_candlestick_patterns` method binnen deze class (gebruikt door `detect_patterns_multi_timeframe`) leest nu ook de voorgecalculeerde candlestick kolommen die door de strategie zijn aangeleverd. **Cruciaal:** De **kwaliteit en representativiteit van de getrainde modellen en de gebruikte labels zijn absoluut essentieel** voor betrouwbare CNN-voorspellingen. Het ontwikkelen, valideren en onderhouden van deze modellen is een belangrijke en **voortdurende taak**.
-   `prompt_builder.py`: Genereert gedetailleerde AI-prompts. Het integreert marktgegevens, sentiment, en een uitgebreide set aan technische features. Deze features, inclusief standaard indicatoren (RSI, MACD, Bollinger Bands, EMA's, Volume Means) en candlestickpatronen, worden nu centraal berekend in `DUOAI_Strategy.py` (middels `pandas-ta`, custom Pandas functies, of `qtpylib`) en aangeleverd aan de `PromptBuilder`. CNN patroon scores van `cnn_patterns.py` worden ook meegenomen.
-   `reflectie_lus.py`: De centrale AI-reflectiecyclus, coördineert promptgeneratie, AI-aanroepen en het loggen van reflecties.
-   `reflectie_analyser.py`: Analyseert reflectielogs en haalt prestatiegegevens direct uit Freqtrade's database om bias-scores en mutatievoorstellen te genereren.
-   `bias_reflector.py`: Beheert de leerbare voorkeur/bias van de strategie per token/strategie.
-   `confidence_engine.py`: Beheert de leerbare confidence score en past `maxTradeRiskPct` dynamisch aan.
-   `cooldown_tracker.py`: Beheert AI-specifieke cooldown-periodes per token/strategie, aanvullend op Freqtrade's ingebouwde cooldowns.
-   `entry_decider.py`: Neemt AI-gestuurde entry-besluiten op basis van consensus, confidence, bias en patronen, inclusief AI-specifieke cooldowns en `timeOfDayEffectiveness`.
-   `exit_optimizer.py`: Neemt AI-gestuurde exit-besluiten en optimaliseert dynamisch de trailing stop loss.
-   `backtester.py`: Biedt een custom backtesting-omgeving, los van Freqtrade's ingebouwde backtest-functionaliteit. Het haalt OHLCV data op en gebruikt de `DUOAI_Strategy.py` (via `advise_all_indicators`) om alle benodigde indicatoren te berekenen en toe te voegen voor het simuleren van trades.
-   `pre_trainer.py`: Module voor data-voorbereiding en het pre-trainen van AI-modellen. Haalt historische OHLCV data op en verrijkt deze met technische indicatoren en candlestickpatronen door gebruik te maken van de logica in `DUOAI_Strategy.py` (via `advise_all_indicators`). Deze verrijkte data wordt vervolgens gebruikt voor het trainen van modellen, zoals de CNNs.
-   `strategy_manager.py`: Beheert strategieparameters, prestaties (haalt uit Freqtrade DB) en mutatievoorstellen. Kan nu proberen Freqtrade actief te informeren over parameterwijzigingen via een API-aanroep. Hiervoor dienen de omgevingsvariabelen `FREQTRADE_API_URL` (bijv. `http://localhost:8080`) en optioneel `FREQTRADE_API_TOKEN` (voor authenticatie) ingesteld te zijn. De specifieke Freqtrade API endpoint (standaard geprobeerd: `/api/v1/reloadconfig`) moet overeenkomen met de actieve Freqtrade-configuratie.
-   `bitvavo_executor.py`: Module voor interactie met de Bitvavo exchange (balans, orders, data), en de integratie is verder geconsolideerd voor robuuste live-executie.
-   `ai_optimizer.py`: Orchestreert AI-gedreven optimalisatiecycli. Maakt nu gebruik van `core/market_data_provider.py` om recente marktdata op te halen, die nodig is voor het genereren van contextuele mutatievoorstellen. Deze provider kan data lokaal laden of, indien nodig, downloaden via Freqtrade. De afhankelijkheden (zoals `PreTrainer` en `StrategyManager`) worden nu via dependency injection beheerd voor verbeterde modulariteit en testbaarheid.
-   `ai_activation_engine.py`: Trigger-engine voor AI-reflectie bij specifieke gebeurtenissen.
-   `interval_selector.py`: Detecteert en beheert de beste timeframe/interval voor AI-analyse.
-   `params_manager.py`: Centrale manager voor alle dynamisch lerende variabelen.
-   `trade_logger.py`: De actieve functionaliteit van deze module is **verwijderd** ten gunste van Freqtrade's databank. Alle AI-analyses en modeltraining baseren zich nu **exclusief op Freqtrade's interne database** voor trade-historie en -data. Het bestand `trade_logger.py` zelf kan nog aanwezig zijn voor historische referentie of specifieke handmatige exporttaken, maar het speelt geen rol meer in de geautomatiseerde AI-workflow.

## Learnable Dynamische Patroongewichten (Learnable Dynamic Pattern Weights)

Een belangrijke recente ontwikkeling is de implementatie van een systeem voor het dynamisch leren en aanpassen van de gewichten van verschillende CNN-patronen (bv. `cnn_bullFlag_weight`, `cnn_bearishEngulfing_weight`). Dit stelt de bot in staat om adaptief meer belang te hechten aan patronen die historisch succesvoller zijn gebleken.

-   **Patroonbijdrage Logging:** `EntryDecider` identificeert en `DUOAI_Strategy` logt nu welke specifieke patronen (zowel CNN als regelgebaseerd) hebben bijgedragen aan een entry-beslissing. Deze gegevens worden opgeslagen in `user_data/logs/pattern_performance_log.json`.
-   **Prestatieanalyse:** De nieuwe `PatternPerformanceAnalyzer` module (`core/pattern_performance_analyzer.py`) analyseert deze logs in combinatie met gesloten trades uit de Freqtrade database. Het berekent succescriteria zoals winstpercentage en gemiddelde winst voor elk uniek patroon.
-   **Gewichtsoptimalisatie:** De `PatternWeightOptimizer` module (`core/pattern_weight_optimizer.py`) gebruikt deze prestatiemetrieken om de individuele gewichten van CNN-patronen aan te passen. Gewichten van succesvolle patronen worden verhoogd, terwijl die van minder succesvolle patronen worden verlaagd (binnen configureerbare minimum- en maximumgrenzen).
-   **Continue Integratie:** Dit leerproces is geïntegreerd in de periodieke cyclus van de `AIOptimizer`, wat zorgt voor continue adaptatie en optimalisatie van de patroongewichten.
-   **Configuratie:** Alle relevante parameters voor dit systeem (zoals minimum/maximum gewicht, leersnelheid, prestatiedrempels) zijn configureerbaar via `params.json`.

## Belangrijke Operationele Aspecten en Huidige Status
Deze sectie belicht enkele belangrijke punten met betrekking tot de huidige werking, de status van bepaalde functionaliteiten en eventuele beperkingen of aandachtspunten.

### CNN Modellen: Functionaliteit en Doorlopende Ontwikkeling
De `cnn_patterns.py` module kan, zoals eerder genoemd, zowel regelgebaseerde als Deep Learning CNN-voorspellingen uitvoeren. Hoewel `pre_trainer.py` een basis CNN kan trainen en `cnn_patterns.py` modellen kan laden, is de **volledige pijplijn van modelontwikkeling, uitgebreide training, validatie en optimalisatie essentieel** en een voortdurende taak. De huidige CNN-integratie is een fundament; de daadwerkelijke voorspellende kracht hangt af van robuust getrainde modellen. De `cnn_patterns.py` retourneert numerieke scores. EntryDecider past nu **dynamisch geleerde, individuele gewichten** toe op de scores van verschillende CNN-patronen (bv. cnn_bullFlag_weight, cnn_bearishEngulfing_weight) om hun invloed op entry-beslissingen te bepalen. Deze gewichten worden continu bijgesteld door de PatternWeightOptimizer.

### Dynamische Configuratie, AI-Advies en Runtime Aanpassingen
De AI-modules binnen dit project kunnen adviezen genereren voor diverse strategieparameters. Sommige hiervan worden intern door de AI gebruikt (bijvoorbeeld via aparte AI-specifieke filters of logica zoals `cooldown_tracker.py`). Echter, voor Freqtrade's kernconfiguratie (`config/config.json`) geldt:
*   **AI-Advies vs. Directe Aanpassing:** AI-gegenereerde suggesties voor parameters in `config.json` (zoals `slippageTolerancePct` of `stake_amount`) dienen als **advies**. Ze worden niet automatisch door de AI in `config.json` gewijzigd tijdens runtime.
*   **Herstart Vereist voor `config.json` Wijzigingen:** Alle aanpassingen aan fundamentele Freqtrade-instellingen in `config/config.json` (zoals `pair_whitelist`, `stake_amount`, `exchange` details, etc.), ongeacht of ze handmatig of op basis van AI-advies worden gedaan, worden **niet dynamisch tijdens runtime door Freqtrade overgenomen.** Een **volledige herstart van de Freqtrade bot is noodzakelijk** om deze wijzigingen actief te maken.
*   **Interne AI-Mechanismen:** Voor bepaalde parameters, zoals cooldowns, kan de AI gebruikmaken van eigen, parallelle mechanismen (zoals `cooldown_tracker.py`) die Freqtrade's instellingen aanvullen zonder `config.json` direct te wijzigen.

### `preferredPairs` en Dynamisch Pair Management
*   **Leerlogica Geïmplementeerd:** De leerlogica om dynamisch `preferredPairs` te identificeren is **nu geïmplementeerd** en wordt beheerd door `core/ai_optimizer.py`.
*   **Koppeling met `pair_whitelist`:** Freqtrade laadt zijn actieve handelsparen (de `pair_whitelist`) uit `config/config.json` bij het opstarten. Een door `core/ai_optimizer.py` gegenereerde lijst van `preferredPairs` wordt **niet automatisch tijdens runtime gesynchroniseerd** met deze `pair_whitelist`.
*   **Handmatige Update en Herstart Vereist:** Om de door de AI geleerde `preferredPairs` actief te maken voor trading, dient de `pair_whitelist` in `config/config.json` handmatig bijgewerkt te worden met de aanbevelingen van de AI. Vervolgens is een **volledige herstart van de Freqtrade bot noodzakelijk**.

### Data Logging en Analyse
Voor alle AI-gestuurde analyses, prestatie-evaluaties en het trainen van modellen wordt nu **exclusief gebruik gemaakt van Freqtrade's interne database**. Dit waarborgt een consistente en betrouwbare databron.

### Teststructuur en Kwaliteitsborging
Het project bevat nu een **uitgebreide, formele pytest testsuite** in de /tests/ map, die kernmodules zoals ParamsManager, EntryDecider, PatternPerformanceAnalyzer, PatternWeightOptimizer en hun interacties dekt. Hoewel de dekking significant is verbeterd, blijft continue uitbreiding en onderhoud van de testsuite een prioriteit om de robuustheid en betrouwbaarheid van de bot te waarborgen.

## Toekomstige Ontwikkeling

-   **Continue Verfijning van CNN Modellen:** Voortdurende training, validatie en optimalisatie van CNN-modellen voor diverse marktpatronen, inclusief het verkennen van nieuwe architecturen en feature sets.
-   **Uitbreiding van Leerbare Parameters:** Onderzoeken welke andere strategische parameters (naast CNN patroongewichten) dynamisch geleerd kunnen worden op basis van prestaties.
-   **Geavanceerd Dynamisch Pairlist Management:** Verder onderzoek naar en implementatie van meer geavanceerde, AI-gestuurde methoden voor het dynamisch selecteren en beheren van de Freqtrade `pair_whitelist`.
-   **Continue Verbetering Testsuite:** Voortdurende uitbreiding en onderhoud van de `pytest` testsuite om maximale codekwaliteit en betrouwbaarheid te garanderen.
-   **Uitgebreide Monitoring en Visualisatie:** Ontwikkeling van geavanceerdere tools voor monitoring van botprestaties en datavisualisatie, mogelijk via een AI-ondersteunde GUI.
-   **Integratie van Backtesting Resultaten in Leerprocessen:** Formeel integreren van Freqtrade's backtesting resultaten in de leerlus van de `AIOptimizer` en `PatternWeightOptimizer` voor snellere iteratie en parameteroptimalisatie.
