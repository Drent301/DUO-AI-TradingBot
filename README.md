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
    De basisconfiguratie is aanwezig. De `pair_whitelist` is bijgewerkt met de gevraagde handelsparen.
    De API-sleutels in `config.json` zijn placeholders en zullen door Freqtrade via de omgevingsvariabelen (geladen via `.env`) worden overschreven in live-modus.
    De momenteel opgenomen paren zijn: `"ETH/EUR", "BTC/EUR", "ZEN/EUR", "WETH/USDT", "USDC/USDT", "WBTC/USDT", "LINK/USDT", "UNI/USDT", "ZEN/BTC", "LSK/BTC", "ETH/BTC"`.
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
-   `cnn_patterns.py`: Bevat algoritmische patroonherkenning (candlestick, chart). **Deze module is voorbereid op integratie met getrainde CNN Deep Learning-modellen en kan nu getrainde modellen laden en gebruiken voor numerieke voorspellingen.** Momenteel gebruikt het echter nog primaire regelgebaseerde detectie zolang er geen getraind model beschikbaar is.
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
-   `trade_logger.py`: **Deze module functioneert nu als een placeholder voor geavanceerde export en specifieke debugging-behoeften.** De primaire trade-historie wordt beheerd door Freqtrade's interne database.

## Status & Ontbrekende Functionaliteit (Kritieke Punten)

De workflow is grotendeels geïmplementeerd op algoritmisch niveau en de AI-modules communiceren effectief. Echter, om een volledig operationele en geavanceerd zelflerende bot te realiseren zoals in de manifesten beschreven, zijn de volgende punten cruciaal:

-   **CNN Modellen - Training en Volledige Integratie:**
    * **Ontbrekend:** Hoewel `pre_trainer.py` nu een basis CNN kan trainen en `cnn_patterns.py` een model kan laden en voorspellingen kan doen, moet de **volledige pijplijn van modelontwikkeling, uitgebreide training en validatie** nog plaatsvinden. De huidige CNN in `cnn_patterns.py` is een proof-of-concept; deze moet geoptimaliseerd worden voor nauwkeurige patroonherkenning in een Deep Learning-context.
    * **Prioriteit:** **Hoog**. Dit is de belangrijkste "AI" ontwikkelingsfase.

-   **`cnnPatternWeight`:**
    * **Ontbrekend:** De `cnn_patterns.py` retourneert nu numerieke scores (wanneer het CNN-model geladen is), maar de **actieve toepassing van `cnnPatternWeight` als een leerbare multiplier** in `entry_decider.py` en `exit_optimizer.py` om de CNN-voorspellingen te wegen in het besluitvormingsalgoritme is nog een TODO. Momenteel wordt alleen de boolean check op `cnn_patterns.get_patterns` gebruikt.

-   **Dynamische Freqtrade Configuratie-aanpassing (Beperking):**
    * **Status:** De AI kan advies geven en opslaan in `params.json` voor variabelen zoals `slippageTolerancePct`. `cooldown_tracker.py` biedt een AI-specifieke cooldown.
    * **Beperking:** Freqtrade's architectuur staat **geen hot-swapping** toe van cruciale parameters in `config.json` tijdens runtime. AI-advies voor *deze specifieke Freqtrade-instellingen* zal **niet automatisch** worden toegepast zonder een bot-herstart.
    * **Aanpak:** De huidige aanpak (AI-specifieke cooldown als aanvulling, slippage als advies) is de meest praktische zonder diepe Freqtrade-core modificaties.

-   **`preferredPairs` (Actieve Koppeling met Freqtrade):**
    * **Status:** De leerlogica om dynamisch de "favoriete paren" te bepalen in `ai_optimizer.py` is geïmplementeerd.
    * **Ontbrekend/Beperking:** De **automatische koppeling van deze geleerde `preferredPairs` met Freqtrade's `pair_whitelist`** in `config.json` vereist een **herstart** van de bot.
    * **Optimalisatie:** Documenteer de beperking en adviseer periodieke handmatige synchronisatie of onderzoek geavanceerde Freqtrade extensies voor dynamisch pairlist management.

-   **Teststructuur (Formele Unit Tests):**
    * **Ontbrekend:** Hoewel de `if __name__ == "__main__":` blokken zijn toegevoegd voor elk Python-bestand, is een **formele, geautomatiseerde testsuite met `pytest` cruciaal** voor robuustheid en kwaliteitsborging. Voor meer informatie over deze frameworks, zie de [pytest documentatie](https://docs.pytest.org/) en de [unittest documentatie](https://docs.python.org/3/library/unittest.html).
    * **Prioriteit:** **Hoog**.

## Toekomstige Ontwikkeling

-   **Fase 1: CNN Model Training & Verfijning:** Dit omvat uitgebreide training, evaluatie en optimalisatie van de Deep Learning CNN-modellen.
-   **Fase 2: Verfijning van AI-Besluitvorming:** Actieve integratie van `cnnPatternWeight` met numerieke scores in `entry_decider.py` en `exit_optimizer.py`.
-   **Fase 3: Geavanceerd Pairlist Management:** Onderzoek en implementeer geavanceerde methoden voor het dynamisch beïnvloeden van Freqtrade's pairlist.
-   **Fase 4: Formele Testsuite:** Ontwikkel uitgebreide unit tests met `pytest` in de `/tests/` map.
-   **Fase 5: Monitoring & Visualisatie:** Overweeg een AI-GUI via een Socket.IO dashboard.
