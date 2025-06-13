# Validatie en Backtesting Plan

Dit document beschrijft de strategieën en richtlijnen voor het valideren van de getrainde modellen en het backtesten van de handelsstrategieën binnen dit project.

## Cross-Validatie Strategie

**Doel:** Het evalueren van de generalisatiecapaciteit van de getrainde CNN-modellen en het voorkomen van overfitting op de trainingsdata.

**Methode:**
De gekozen methode voor cross-validatie is `TimeSeriesSplit` uit `sklearn.model_selection`. Deze methode is geschikt voor tijdreeksdata omdat het ervoor zorgt dat de validatieset altijd data bevat die later in de tijd komt dan de trainingsset, waardoor data leakage wordt voorkomen.

De cross-validatie wordt aangestuurd door de volgende parameters in `ParamsManager`:
*   `perform_cross_validation` (boolean): Bepaalt of cross-validatie wordt uitgevoerd.
*   `cv_num_splits` (int): Specificeert het aantal (k) folds dat gebruikt wordt in de `TimeSeriesSplit`.

**Proces:**
1.  De trainingsdata wordt opgesplitst in `k` opeenvolgende folds.
2.  Voor elke fold wordt een nieuw CNN-model geïnstantieerd en getraind op de trainingsdata van die fold.
3.  Het getrainde model wordt vervolgens gevalideerd op de validatiedata van die fold.
4.  De volgende model metrieken worden berekend en gelogd voor elke fold:
    *   Accuracy
    *   Precision (met `zero_division=0`)
    *   Recall (met `zero_division=0`)
    *   F1-Score (met `zero_division=0`)
    *   AUC-ROC Score (met een check voor het aantal klassen in `y_true`)
5.  Na het doorlopen van alle folds, worden de gemiddelde waarde en de standaarddeviatie van elk van deze metrieken berekend over alle folds. Deze geaggregeerde resultaten geven een indicatie van de algehele prestatie en stabiliteit van het model.
6.  Los van de cross-validatie wordt er ook een finaal model getraind op de volledige (gecombineerde) trainingsset. Dit model wordt vervolgens opgeslagen voor later gebruik, zoals in de backtesting fase.

**Resultaten Opslag:**
De gedetailleerde resultaten van de cross-validatie, inclusief de metrieken per fold en de geaggregeerde statistieken, worden opgeslagen in `memory/pre_train_log.json` onder de `cross_validation_results` sectie voor elke getrainde modelconfiguratie.

## Backtesting Plan op Out-of-Sample Data

**Doel:** Het evalueren van de prestaties van de getrainde modellen en de daarop gebaseerde handelsstrategie op een significante periode van recente, ongeziene data. Dit simuleert hoe de strategie zou kunnen presteren in een live trading omgeving.

**Data Selectie:**
Een recent deel van de beschikbare historische data wordt gereserveerd als een out-of-sample (OOS) testset. Deze data is niet gebruikt tijdens het trainen of cross-valideren van de modellen. De startdatum van deze OOS-periode wordt gespecificeerd via de `ParamsManager` parameter:
*   `backtest_start_date_str` (string, "YYYY-MM-DD"): Definieert het begin van de backtest periode. Data vanaf deze datum tot de meest recent beschikbare data wordt gebruikt.

**Simulatie Proces (via `core/backtester.py`):**
1.  **Laden van Modellen:** De relevante, eerder getrainde CNN-modellen worden geladen via de `CNNPatterns` module. Dit gebeurt op basis van het handelspaar, timeframe, patroontype en de gespecificeerde CNN-architectuur.
2.  **Iteratie door Data:** De `Backtester` module itereert candle-voor-candle (of bar-voor-bar) door de out-of-sample dataset.
3.  **Voorspellingen:** Voor elke candle (na een initiële `sequence_length` periode) wordt met het geladen CNN-model een voorspelling gedaan van de patroonscore (`predict_pattern_score`).
4.  **Handelsstrategie:** Een eenvoudige, op regels gebaseerde strategie wordt toegepast:
    *   **Entry:** Een long positie wordt geopend als de voorspelde patroonscore een gespecificeerde drempel overschrijdt.
    *   **Exit:** Een open positie wordt gesloten op basis van:
        *   Take Profit (TP): Als de prijs een bepaald percentage stijgt.
        *   Stop Loss (SL): Als de prijs een bepaald percentage daalt.
        *   Hold Duration: Als de positie een maximaal aantal candles open is geweest.
5.  **Parameters Handelsstrategie (uit `ParamsManager`):**
    *   `backtest_entry_threshold` (float): Minimale score voor entry.
    *   `backtest_take_profit_pct` (float): Percentage voor take profit.
    *   `backtest_stop_loss_pct` (float): Percentage voor stop loss.
    *   `backtest_hold_duration_candles` (int): Maximale houdtijd in candles.
    *   `backtest_initial_capital` (float): Startkapitaal voor de simulatie.
    *   `backtest_stake_pct_capital` (float): Percentage van het kapitaal dat per trade wordt ingezet.
6.  **Handelsparen:** De backtesting wordt uitgevoerd voor de handelsparen gespecificeerd in `data_fetch_pairs` (bijv. "LSK/BTC", "ZEN/BTC", "ETH/BTC", "ETH/EUR").

**Opslag Resultaten:**
Alle resultaten van de backtest, inclusief een lijst van uitgevoerde trades, de evolutie van het portfolio kapitaal over tijd, en de berekende financiële performance metrieken, worden opgeslagen in `memory/backtest_results.json`.

## Gebruikte Performance Metrieken

### Model Evaluatie Metrieken (Cross-Validatie / Validatieset)

Deze metrieken worden gebruikt om de voorspellende kracht van de CNN-modellen te beoordelen:

*   **Accuracy:** Het percentage correct geclassificeerde samples (zowel positieve als negatieve klasse).
    *   *Formule:* `(True Positives + True Negatives) / Total Samples`
*   **Precision:** Van alle samples die als positief zijn voorspeld, welk percentage daadwerkelijk positief was. Relevant om het aantal valse positieven te minimaliseren.
    *   *Formule:* `True Positives / (True Positives + False Positives)`
*   **Recall (Sensitivity):** Van alle daadwerkelijk positieve samples, welk percentage correct als positief is voorspeld. Relevant om te zorgen dat werkelijke patronen niet gemist worden.
    *   *Formule:* `True Positives / (True Positives + False Negatives)`
*   **F1-Score:** Het harmonisch gemiddelde van Precision en Recall, biedt een gebalanceerde maat.
    *   *Formule:* `2 * (Precision * Recall) / (Precision + Recall)`
*   **AUC-ROC (Area Under the Receiver Operating Characteristic Curve):** Een maat voor de algehele prestatie van een classificatiemodel om onderscheid te maken tussen positieve en negatieve klassen over alle drempelwaarden. Een waarde van 1.0 is perfect, 0.5 is willekeurig.

### Backtesting Financiële Metrieken

Deze metrieken worden gebruikt om de financiële prestaties en risico's van de gesimuleerde handelsstrategie te beoordelen:

*   **Total Return %:** Het totale procentuele rendement op het initiële kapitaal over de gehele backtest periode.
*   **Sharpe Ratio (Annualized):** Meet het voor risico gecorrigeerde rendement. Een hogere Sharpe Ratio indiceert een beter rendement per eenheid risico (volatiliteit). Berekend op basis van dagelijkse returns en geannualiseerd.
*   **Maximum Drawdown %:** Het grootste piek-tot-dal verlies gedurende de backtest periode, uitgedrukt als een percentage van het piek kapitaal. Geeft een indicatie van het potentiële neerwaartse risico.
*   **Win Rate %:** Het percentage winnende trades ten opzichte van het totale aantal trades.
*   **Profit Factor:** De verhouding tussen de totale winst van winnende trades en de totale absolute verlies van verliezende trades (`Gross Profit / Gross Loss`). Een waarde > 1 indiceert winstgevendheid.
*   **Number of Trades:** Het totale aantal uitgevoerde trades gedurende de backtest.
*   **Average Profit per Winning Trade:** Het gemiddelde P&L (Profit and Loss) bedrag voor alle winnende trades.
*   **Average Loss per Losing Trade:** Het gemiddelde P&L bedrag voor alle verliezende trades (zal een negatieve waarde zijn).

## Interpretatie van Resultaten

De gecombineerde resultaten van cross-validatie en backtesting worden gebruikt om de levensvatbaarheid van een model en strategie te beoordelen.

**Model Metrieken (Cross-Validatie):**
*   **Hoge Gemiddelde Scores (Accuracy, F1, AUC-ROC > 0.55-0.6):** Indiceert dat het model enige voorspellende waarde heeft op de validatie data-segmenten. Scores dichtbij 0.5 suggereren dat het model niet veel beter presteert dan willekeurig gokken.
*   **Lage Standaarddeviaties over Folds:** Wijst op stabiele prestaties en dat het model niet extreem gevoelig is voor specifieke data segmenten in de trainingsset. Hoge standaarddeviaties kunnen wijzen op overfitting op bepaalde data karakteristieken.
*   **Balans Precision/Recall:** Afhankelijk van de strategie kan de ene belangrijker zijn dan de andere. Voorzichtigheid is geboden bij het openen van trades (hoge precision), versus het niet missen van kansen (hoge recall). F1 biedt een balans.

**Financiële Metrieken (Backtesting):**
*   **Total Return %:** Moet positief en significant zijn om de strategie als potentieel winstgevend te beschouwen. Te vergelijken met een buy-and-hold benchmark.
*   **Sharpe Ratio:** Een geannualiseerde Sharpe Ratio > 1 wordt vaak als goed beschouwd, > 2 als zeer goed. Negatieve waarden zijn onwenselijk.
*   **Maximum Drawdown %:** Een kritische risicomaatstaf. Lagere waarden zijn beter. Een te hoge drawdown (bijv. > 20-30%) kan onacceptabel zijn, afhankelijk van de risicotolerantie.
*   **Win Rate %:** Hoewel niet de enige factor, geeft een indicatie van de consistentie. Een lage win rate kan psychologisch zwaar zijn, zelfs als de strategie winstgevend is door enkele grote winnaars.
*   **Profit Factor:** Idealiter > 1.5 - 2.0. Een waarde dichtbij 1 betekent dat winsten en verliezen elkaar bijna opheffen.
*   **Number of Trades:** Moet voldoende zijn om statistische relevantie te bieden. Te weinig trades (< 50-100 over een lange periode) maken de resultaten minder betrouwbaar.
*   **Avg Profit/Win vs. Avg Loss/Losing (Risk/Reward Ratio):** Een goede strategie heeft vaak winnaars die significant groter zijn dan de verliezers (bijv. een ratio > 1.5:1 of 2:1).

**Algemene Overwegingen:**
*   **Consistentie:** Resultaten moeten consistent zijn tussen cross-validatie en de out-of-sample backtest. Een model dat goed presteert in CV maar faalt in backtesting is waarschijnlijk overfit.
*   **Vergelijking:** Prestaties moeten worden vergeleken met een baseline (bijv. buy-and-hold van de onderliggende asset) en eventueel met andere strategieën.
*   **Parameter Gevoeligheid:** Onderzoek (indien tijd) hoe robuust de strategie is voor kleine variaties in de parameters (TP, SL, entry drempel).
*   **Realiteitscheck:** Houd rekening met factoren die niet in de backtest zijn meegenomen (slippage, fees, liquiditeit, data kwaliteit). De `BitvavoExecutor` houdt al rekening met fees bij het simuleren van trades, wat een pluspunt is.

Een positieve evaluatie vereist een model dat generaliseert (goede CV scores) en een strategie die winstgevendheid en gecontroleerd risico laat zien op ongeziene data (goede backtesting metrics). Iteratie op modelarchitectuur, feature engineering, en strategieparameters zal nodig zijn om tot een bevredigend resultaat te komen.
