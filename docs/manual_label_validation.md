# Manual Label Validation Process and Tooling Concept

## 1. Purpose of Manual Label Validation

Automated labeling, while efficient, can sometimes produce inaccurate or suboptimal labels for complex market patterns. This can be due to rigid rule definitions, noise in the data, or the inherent ambiguity of certain chart formations. Manual label validation by experienced traders or analysts serves several critical purposes:

*   **Accuracy Assessment:** To quantify the performance of the automated labeling algorithms (e.g., `bullFlag`, `bearishEngulfing` detectors combined with future price action checks).
*   **Algorithm Refinement:** To identify systematic errors or weaknesses in the automated labeling rules, providing insights for their improvement.
*   **Gold Standard Dataset Creation:** To build a high-quality, human-verified dataset of labeled patterns. This "gold standard" dataset can be used for:
    *   More robust model training and evaluation.
    *   Fine-tuning models that were initially pre-trained on automatically labeled data.
    *   Developing more sophisticated automated labeling techniques.
*   **Understanding Edge Cases:** To discover and document ambiguous or rare scenarios where automated rules may fail, helping to build a more nuanced understanding of pattern validity.

## 2. Proposed Workflow

The manual validation process will involve selecting relevant data points, presenting them to an expert through a dedicated tool, capturing their feedback, and storing this feedback in a structured format.

### 2.1. Data Selection for Validation

To make the validation process efficient and impactful, data points should be selected strategically:

*   **Source Data:** Candlestick data (OHLCV) for specific trading pairs and timeframes that have already been processed by the automated labeling pipeline (i.e., have a `{pattern_type}_label` and potentially a `form_pattern_detected` column).
*   **Selection Strategies:**
    *   **Random Sampling:** A baseline approach to get a general sense of accuracy across all detected patterns.
    *   **Uncertainty Sampling:** Prioritize samples where the automated model (if a preliminary model exists) has low confidence in its prediction.
    *   **Disagreement-Based Sampling:** If multiple automated labeling methods or models are used, select samples where they disagree.
    *   **Focus on Positives:** Initially, it might be more valuable to validate samples that the automated system has labeled as positive (i.e., pattern detected and outcome confirmed `_label = 1`), as these are often rarer and more critical to get right.
    *   **Focus on Form Detections:** Validate samples where `form_pattern_detected == True` to specifically assess the quality of the initial shape recognition before the future outcome is considered.

### 2.2. Tooling - Review Interface

A dedicated review interface (validation tool) is crucial for presenting the necessary information to the expert clearly and efficiently. The interface should display:

*   **Primary Chart:**
    *   Interactive candlestick chart (e.g., using Plotly, Bokeh, or a lightweight JS charting library) showing the OHLCV data for a window around the potential pattern.
    *   The specific candle where the pattern is detected should be highlighted (e.g., the engulfing candle, the breakout candle of a flag).
    *   The window size should be configurable (e.g., 50-100 candles before the pattern, 50-100 after).
*   **Contextual Information:**
    *   Trading Pair (e.g., ETH/EUR)
    *   Timeframe (e.g., 1h)
    *   Date/Time of the detected pattern's end candle.
*   **Pattern Information:**
    *   `pattern_type`: The type of pattern being validated (e.g., "bullFlag", "bearishEngulfing").
    *   Automated Label: The label assigned by the automated system (e.g., `bullFlag_label = 1`).
    *   (Optional) `form_pattern_detected`: The boolean output from the initial shape detection step.
*   **Future Price Action:**
    *   Clearly display the `future_N_candles` period on the chart that was used by the automated system for determining profit/loss.
    *   Visually indicate the profit and loss thresholds if possible.
*   **Relevant Indicators (Optional but Recommended):**
    *   Display common indicators like Moving Averages, RSI, MACD, Bollinger Bands on the chart or as subplots to provide more context for the decision. These should be the same indicators available during automated processing.

### 2.3. Expert Interaction & Validation Options

The expert should be able to provide feedback for each presented sample:

*   **Confirm Automated Label:** Agree with the label provided by the system.
*   **Correct Label:**
    *   **Correct to Positive (1):** If the system labeled it 0, but the expert deems it a valid and successful pattern.
    *   **Correct to Negative (0):** If the system labeled it 1, but the expert deems it invalid or unsuccessful.
*   **Mark as Unclear/Ambiguous:** For situations where the pattern is too noisy or doesn't clearly fit the definition.
*   **Add Notes (Optional):** A text field to provide qualitative feedback or reasons for their decision, especially for corrections or unclear cases. This is valuable for refining rules.

### 2.4. Saving Validated Labels

The validated labels need to be stored in a structured format for later use.

*   **Format:** CSV or JSON are suitable. CSV is often simpler for direct use with pandas.
*   **Key Columns to Include:**
    *   `original_candle_timestamp`: Timestamp of the candle where the pattern was detected (e.g., from the source DataFrame index).
    *   `symbol`: e.g., "ETH/EUR"
    *   `timeframe`: e.g., "1h"
    *   `pattern_type`: e.g., "bullFlag"
    *   `automated_label`: The original label from the system.
    *   `validated_label`: The label provided by the expert (0, 1, or perhaps a specific code for "unclear").
    *   `validator_id`: Identifier for the expert who performed the validation.
    *   `validation_timestamp`: When the validation was performed.
    *   `notes` (Optional): Any text notes from the expert.
    *   (Optional) `original_form_detected`: The boolean value from the `form_pattern_detected` step.

## 3. Iteration and Usage of Validated Data

The manually validated dataset is a valuable asset and can be used iteratively:

*   **Evaluate Automated Labeling Algorithm:** Calculate precision, recall, F1-score, and accuracy of the automated labeling against the validated set to understand its performance.
*   **Refine Automated Rules:** Analyze discrepancies and expert notes to identify weaknesses in the `prepare_training_data` logic (both form detection and future outcome rules) and refine them.
*   **Train "Gold Standard" Models:** Use the human-verified labels to train new models or fine-tune existing pre-trained models. These models are expected to be more robust.
*   **Active Learning:** The validation process can be integrated into an active learning loop, where the models being trained can request human validation for samples they are most uncertain about.

## 4. Technology Suggestions for Validation Tool

Several technologies can be used to build the review interface:

*   **Streamlit (Recommended to Start):**
    *   Pros: Pure Python, very fast development for data-centric apps, good support for interactive charts (e.g., Plotly), easy to integrate with pandas.
    *   Cons: Can be less flexible for highly custom UI/UX compared to web frameworks.
*   **Dash (Plotly):**
    *   Pros: Also Python-based, more customizable than Streamlit, built by Plotly so excellent charting integration.
    *   Cons: Steeper learning curve than Streamlit, can involve more callback boilerplate.
*   **Jupyter Notebook/Lab with `ipywidgets`:**
    *   Pros: Good for quick, iterative development and data exploration within the familiar Jupyter environment.
    *   Cons: Not ideal for a standalone, shareable tool for non-technical experts; can become unwieldy for complex UIs.
*   **Lightweight Web Framework (e.g., Flask, FastAPI) + JavaScript Charting Library:**
    *   Pros: Maximum flexibility in UI/UX design and functionality.
    *   Cons: Requires web development skills (Python backend, HTML/CSS/JS frontend), significantly more development effort.

**Recommendation:** Start with **Streamlit** due to its rapid development capabilities and suitability for data-focused applications. If more complex UI/UX requirements arise, Dash or a lightweight web framework could be considered.

## 5. Example Data Structure for Validated Labels

**File: `validated_labels.csv`**

```csv
original_candle_timestamp,symbol,timeframe,pattern_type,automated_label,validated_label,validator_id,validation_timestamp,notes,original_form_detected
1609459200000,ETH/EUR,1h,bullFlag,1,1,expert_A,2023-10-27T10:30:00Z,"Clear breakout and follow-through",True
1609545600000,ETH/EUR,1h,bullFlag,1,0,expert_A,2023-10-27T10:35:00Z,"Breakout failed, turned into a bull trap.",True
1609632000000,BTC/USDT,4h,bearishEngulfing,0,0,expert_B,2023-10-27T11:00:00Z,"Not a clear engulfing pattern.",False
1609718400000,BTC/USDT,4h,bearishEngulfing,1,1,expert_B,2023-10-27T11:05:00Z,"Confirmed bearish engulfing with good downside.",True
1609804800000,ETH/EUR,1h,bullFlag,0,-1,expert_A,2023-10-27T11:10:00Z,"Very choppy, unclear if flag or just consolidation.",False
```
*(Note: `validated_label = -1` could represent "unclear/ambiguous")*
*(Note: `original_form_detected` is optional but useful for assessing the shape detection step specifically)*
