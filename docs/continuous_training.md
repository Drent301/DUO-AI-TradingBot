# Continue Training en Model Hernieuwing

Dit document beschrijft de strategie en mechanismen voor het periodiek hertrainen van modellen met recente data om hun relevantie in veranderende marktomstandigheden te behouden.

## Overzicht

Het primaire doel van de continue trainingspijplijn is om de CNN-modellen periodiek te hertrainen met de meest recente beschikbare marktdata. Dit zorgt ervoor dat de modellen zich kunnen aanpassen aan nieuwe marktdynamieken, patronen en volatiliteit, waardoor hun voorspellende kracht en relevantie over tijd behouden blijft of zelfs verbetert.

## Uitvoeren van de Pijplijn: `run_pipeline.py`

Het centrale script voor het uitvoeren van de volledige data- en trainingspijplijn is `run_pipeline.py`, te vinden in de root directory van het project.

De belangrijkste stappen die dit script uitvoert zijn:
1.  **Initialisatie van Logging:** Configureert logging om output naar zowel een bestand (`pipeline_run.log`) als naar de console te sturen.
2.  **Laden Omgevingsvariabelen:** Laadt variabelen uit het `.env` bestand (bijv. API keys).
3.  **Initialisatie Core Componenten:**
    *   `ParamsManager`: Voor het beheren van alle configuratieparameters.
    *   `BitvavoExecutor`: Voor interactie met de Bitvavo API (data ophalen, order executie in de toekomst). Foutafhandeling is aanwezig voor ontbrekende API keys.
    *   `CNNPatterns`: Voor het laden van modeldefinities en het doen van voorspellingen (wordt gebruikt door `PreTrainer` en `Backtester`).
    *   `PreTrainer`: Het hoofdobject dat de gehele pre-trainingscyclus beheert.
4.  **Uitvoeren Pijplijn:** Roept `PreTrainer.run_pretraining_pipeline()` aan. Deze methode orkestreert het volledige proces:
    *   Ophalen van historische OHLCV data (zoals geconfigureerd in `ParamsManager`).
    *   Voorbereiden van de data (toevoegen indicatoren, labelen op basis van patronen en uitkomsten).
    *   Trainen van de CNN modellen (inclusief eventuele Cross-Validatie, indien geconfigureerd).
    *   Uitvoeren van backtesting op de nieuw getrainde (of geselecteerde) modellen (indien geconfigureerd).
5.  **Logging:** Alle belangrijke acties, configuraties, en resultaten worden gelogd naar `pipeline_run.log` en de console.

## Data Actualisatie en Ingestie

De strategie voor het actualiseren van data is als volgt:
*   De pijplijn, via `PreTrainer.fetch_historical_data`, haalt OHLCV data op voor de periode die is geconfigureerd in `ParamsManager`. Dit wordt primair bepaald door:
    *   `data_fetch_start_date_str`: De startdatum vanaf wanneer data wordt opgehaald.
    *   `data_fetch_end_date_str`: De einddatum tot wanneer data wordt opgehaald. Als deze leeg is, wordt data tot "nu" (meest recent beschikbaar) gehaald.
*   **Caching:** Ruwe OHLCV data wordt gecached in `data/raw_historical_data/` per symbool, timeframe, en specifieke periode (start-/eindtimestamp van de chunk).
    *   Als de `run_pipeline.py` opnieuw wordt gestart en de overall gevraagde periode (bijv. `data_fetch_start_date_str` tot `data_fetch_end_date_str`="nu") overlapt met reeds gecachede periodes, zal voor die specifieke, exacte periodes de cache worden gebruikt.
    *   Echter, als `data_fetch_end_date_str` impliciet "nu" is, dan zal elke nieuwe run potentieel nieuwe data voor de meest recente (nog niet volledig afgesloten) periode proberen te fetchen. De pagineringslogica in `_fetch_ohlcv_for_period` (en de onderliggende `BitvavoExecutor.fetch_ohlcv`) zal dan nieuwe candles aan het einde van de periode toevoegen. Als een nieuwe chunk wordt gehaald die deels overlapt met een oude, zal de cache voor die specifieke chunk (indien de periode grenzen exact matchen) gebruikt worden, of een nieuwe chunk wordt gehaald en gecached.
    *   De combinatie van een vaste `data_fetch_start_date_str` en een dynamische (lege) `data_fetch_end_date_str` zorgt er effectief voor dat bij elke run de dataset wordt uitgebreid met de meest recente candles. Oude, volledig afgesloten periodes blijven gecached.
*   **Marktregimes:** Indien `market_regimes.json` specifieke periodes definieert, worden die gebruikt. Als voor een symbool geen regimes zijn gedefinieerd, of als de regimes geen data opleveren, valt de pijplijn terug op de globale `data_fetch_start_date_str` en `data_fetch_end_date_str`.

## Model Deployment en Activering

De huidige strategie voor het activeren van nieuw getrainde modellen is een "nieuwste wint" benadering:

1.  **Model Opslag:** Modellen worden opgeslagen in de `data/models/{symbol}/{timeframe}/` directory. De bestandsnaam bevat het symbool, timeframe, patroontype, en de `current_cnn_architecture_key` (uit `ParamsManager`). Bijvoorbeeld: `ETH_EUR_1h_bullFlag_default_simple.pth`.
2.  **Overschrijven:** Wanneer `run_pipeline.py` een model hertraint met exact dezelfde configuratie (symbool, timeframe, patroon, architectuur), wordt het bestaande modelbestand (`.pth`) en het bijbehorende scaler-bestand (`.json`) overschreven met de nieuw getrainde versie.
3.  **Activering in `CNNPatterns`:**
    *   De `CNNPatterns` klasse laadt modellen dynamisch wanneer een voorspelling voor een specifieke combinatie van symbool, timeframe, patroontype en architectuur wordt gevraagd en het model nog niet in het geheugen is.
    *   De `current_cnn_architecture_key` uit `ParamsManager` speelt een rol bij het bepalen welk model geladen moet worden voor een bepaald patroon.
    *   Als een modelbestand op schijf is overschreven door een nieuwe trainingsronde, zal `CNNPatterns` bij de volgende keer dat het dit specifieke model nodig heeft (bijv. na een herstart van een applicatie die `CNNPatterns` gebruikt, of als `CNNPatterns` intern een mechanisme zou hebben om periodiek modellen te herladen) het nieuwste, overschreven model laden.

**Toekomstige Mogelijkheden (Geavanceerdere Strategieën):**
*   **Champion/Challenger Model:** Een nieuw getraind model (challenger) wordt vergeleken met het huidige actieve model (champion) op basis van recente performance of een specifieke validatieset. Alleen als de challenger significant beter presteert, wordt deze gepromoveerd tot champion.
*   **Model Registry:** Een systeem (bijv. MLflow Model Registry) om versies van modellen te beheren, hun levenscyclus te volgen (staging, production, archived), en het deployen van specifieke versies te faciliteren.
*   **Shadow Mode:** Nieuwe modellen draaien mee in een live-omgeving zonder daadwerkelijk te handelen, om hun prestaties te monitoren voordat ze volledig worden geactiveerd.

Voor de huidige opzet geldt dat het overschrijven van het modelbestand de manier is waarop een nieuw model "live" gaat voor componenten die het via `CNNPatterns` laden.

## Automatisering en Scheduling

Om de trainingspijplijn periodiek en automatisch uit te voeren, kan gebruik worden gemaakt van systeemplanningstools zoals `cron` op Linux/macOS of Windows Task Scheduler.

**Linux/macOS (`cron`):**

Een `crontab` entry kan worden aangemaakt om `run_pipeline.py` op gezette tijden uit te voeren.

1.  Open de crontab editor:
    ```bash
    crontab -e
    ```
2.  Voeg een regel toe voor de gewenste schedule. Voorbeelden:
    *   Elke dag om 03:00 's nachts:
        ```cron
        0 3 * * * /usr/bin/python3 /pad/naar/project/run_pipeline.py >> /pad/naar/project/cron_pipeline.log 2>&1
        ```
    *   Elke maandag om 05:00 's ochtends:
        ```cron
        0 5 * * 1 /usr/bin/python3 /pad/naar/project/run_pipeline.py >> /pad/naar/project/cron_pipeline.log 2>&1
        ```

    **Uitleg van de cron entry:**
    *   `0 3 * * *`: Minute (0), Hour (3), Day of Month (* = elke), Month (* = elke), Day of Week (* = elke).
    *   `/usr/bin/python3`: Het pad naar de Python interpreter (pas aan indien nodig, bijv. als je een virtual environment gebruikt).
    *   `/pad/naar/project/run_pipeline.py`: Het absolute pad naar het `run_pipeline.py` script.
    *   `>> /pad/naar/project/cron_pipeline.log 2>&1`: Stuurt zowel standaard output (stdout) als standaard error (stderr) naar een logbestand specifiek voor cronjobs. Dit is handig voor debugging. De `pipeline_run.log` wordt nog steeds aangemaakt door het script zelf.

**Windows (Task Scheduler):**

1.  Open Task Scheduler (Taakplanner).
2.  Klik op "Create Basic Task..." (Basistaak maken...).
3.  **Name:** Geef een naam, bijv. "Crypto ML Pipeline Training".
4.  **Trigger:** Stel in hoe vaak de taak moet draaien (bijv. "Daily", "Weekly"). Configureer de tijd.
5.  **Action:** Kies "Start a program" (Een programma starten).
    *   **Program/script:** Geef het pad naar de Python interpreter (bijv. `C:\Python39\python.exe` of het pad binnen een venv).
    *   **Add arguments (optional):** Geef het pad naar `run_pipeline.py` (bijv. `C:\pad\naar\project\run_pipeline.py`).
    *   **Start in (optional):** Stel dit in op de root directory van het project (bijv. `C:\pad\naar\project\`). Dit is belangrijk zodat relatieve paden in het script correct werken en het `.env` bestand wordt gevonden.
6.  Controleer de instellingen en voltooi de wizard.
7.  **Belangrijk:** Ga naar de eigenschappen van de aangemaakte taak:
    *   **General Tab:** Overweeg "Run whether user is logged on or not" en "Run with highest privileges" (indien nodig, maar wees voorzichtig).
    *   **Actions Tab:** Controleer of de "Start in" directory correct is ingesteld.

**Belangrijke Overwegingen voor Automatisering:**
*   **Virtual Environments:** Als je een virtual environment (venv) gebruikt, zorg er dan voor dat de Python interpreter van de venv wordt aangeroepen.
    *   Voor `cron`: ` /pad/naar/project/venv/bin/python3 /pad/naar/project/run_pipeline.py ...`
    *   Voor Task Scheduler: Stel het pad naar `python.exe` binnen de venv in.
*   **Padverwijzingen:** Gebruik absolute paden in de scheduler configuratie om problemen met de werkdirectory te voorkomen. Het script `run_pipeline.py` zelf is ontworpen om vanuit de project root te draaien.
*   **API Keys & Environment:** Zorg dat het `.env` bestand met API keys toegankelijk is voor de context waarin de geplande taak draait.
*   **Resource Management:** Langdurige trainingssessies kunnen veel resources (CPU, geheugen) verbruiken. Plan ze op momenten dat de server/machine minder belast is.
*   **Monitoring & Foutafhandeling:**
    *   Controleer regelmatig de logbestanden (`pipeline_run.log` en de cron/Task Scheduler specifieke logs).
    *   Implementeer eventueel extra notificaties (bijv. e-mail bij falen) als de betrouwbaarheid kritisch is.
*   **Concurrency:** Zorg ervoor dat niet meerdere instanties van de pijplijn tegelijkertijd draaien als dit tot conflicten kan leiden (bijv. met het schrijven naar dezelfde bestanden of API rate limits). Een lock-bestand mechanisme kan hierbij helpen.

## Relevante `ParamsManager` Configuraties voor Continue Training

De volgende parameters in `ParamsManager` zijn cruciaal voor het sturen van het gedrag van de continue training en data actualisatie:

*   **Data Gerelateerd:**
    *   `data_fetch_start_date_str`: Bepaalt hoe ver terug in de tijd data wordt gehaald. Voor continue training kan dit een vaste datum in het verleden zijn, of een dynamisch berekende datum (bijv. "1 jaar geleden").
    *   `data_fetch_end_date_str`: Indien leeg, wordt data tot het meest recente punt gehaald. Dit is typisch voor continue training.
    *   `data_fetch_pairs`: De lijst van handelsparen waarvoor data wordt verwerkt en modellen worden getraind.
    *   `data_fetch_timeframes`: De lijst van timeframes die worden gebruikt.
*   **Model Training:**
    *   `patterns_to_train`: Lijst van patroontypes (bijv. "bullFlag", "bearishEngulfing") waarvoor modellen worden getraind.
    *   `current_cnn_architecture_key`: Selecteert welke CNN architectuurconfiguratie uit `cnn_architecture_configs` wordt gebruikt voor training.
    *   `cnn_architecture_configs`: Bevat de gedetailleerde configuraties voor verschillende CNN architecturen.
    *   `sequence_length_cnn`, `num_epochs_cnn`, `batch_size_cnn`, `learning_rate_cnn`: Hyperparameters voor de CNN training.
    *   `pattern_labeling_configs`: Bevat de configuraties voor het labelen van data per patroontype (future N candles, profit/loss thresholds).
*   **Cross-Validatie:**
    *   `perform_cross_validation`: Boolean die aangeeft of cross-validatie moet worden uitgevoerd.
    *   `cv_num_splits`: Aantal splits voor `TimeSeriesSplit`.
*   **Backtesting:**
    *   `perform_backtesting`: Boolean die aangeeft of backtesting na modeltraining moet worden uitgevoerd.
    *   `backtest_start_date_str`: Startdatum voor de out-of-sample backtest periode. Moet idealiter na de trainingsdata liggen.
    *   Overige `backtest_*` parameters (bijv. `backtest_entry_threshold`, `backtest_take_profit_pct`, etc.) die de backtest simulatie sturen.
*   **Algemeen:**
    *   `default_strategy_id`: Wordt door `run_pipeline.py` gebruikt als er een strategy ID nodig is voor de `PreTrainer.run_pretraining_pipeline` methode.

Het periodiek aanpassen van deze parameters (met name de data-gerelateerde en backtesting startdatum) kan onderdeel zijn van de strategie voor continue modelverbetering.
