
***

# S&P 500 ESG Risk Rating Predictor

---

# 1. Project Information

- **Project Title:** S&P 500 ESG Risk Rating Predictor
- **Group Members:**
  - Student 1 – Sarah JACQUEMIN
  - Student 2 – Yasmin OWEN
  - Student 3 – Romane WALLERAND 
- **Course Name:** AI In Finance
- **Instructor:** Nicolas De Roux & Mohamed EL FAKIR
- **Submission Date:** 20/04/2026

---

# 2. Project Description

Environmental, Social, and Governance (ESG) criteria are increasingly vital for investors who want to assess the sustainability and ethical impact of an investment in a company. This project addresses the challenge of automatically determining a company's ESG Risk Level by combining traditional structured corporate data, real-time financial metrics, and natural language processing (NLP) on company descriptions. By building a multimodal machine learning pipeline, this project helps automate risk profiling, which is highly beneficial for portfolio managers, sustainable investment funds, and financial analysts who need to process large universes of stocks efficiently.

---

# 3. Project Goal

This project aims to accurately classify S&P 500 companies into distinct ESG risk categories (Low, Medium, or High Risk). A successful solution will effectively synthesize structured categories (like Sector and Industry), text-based data (company descriptions), and financial fundamentals (like Market Cap and EBITDA) into a single predictive model. Success is measured by how well the final combined model outperforms a standard baseline that only relies on basic structured data.



---

# 4. Task Definition

- **Task Type:** Multi-class Classification
- **Input Variables:** - Structured data (Sector, Industry, Full-Time Employees)
  - Text data (Company Descriptions)
  - Financial data (Market Cap, EBITDA, P/E Ratio, Beta, Profit Margin, Debt-to-Equity)
- **Target Variable:** ESG Risk Level (Low, Medium, High)
- **Evaluation Metric(s):** F1-Macro score (primary) and Accuracy (secondary), alongside Confusion Matrices to evaluate class-wise performance.

---

# 5. Dataset Description

## Dataset Overview

This project uses a primary dataset from Kaggle combined dynamically with live financial data from Yahoo Finance.

- **Number of samples:** ~500 (representing S&P 500 companies, minus dropped rows with missing targets)
- **Number of features:** 12 primary features (3 structured, 2 engineered NLP flags, 1 raw text, 6 financial) + TF-IDF matrix
- **Target variable:** `ESG Risk Level`
- **Data source:** [S&P 500 ESG Risk Ratings on Kaggle](https://www.kaggle.com/datasets/pritish509/s-and-p-500-esg-risk-ratings) and Yahoo Finance API (`yfinance`).

---

## Feature Description

| Feature | Description | Type |
|------|------|------|
| Sector | Broad economic sector of the company | Categorical |
| Industry | Specific industry classification | Categorical |
| Full Time Employees | Total number of employees | Numerical |
| Description | Text summary of the company's business activities | Text |
| is_renewable | Flag indicating mention of renewable energy keywords | Engineered (Binary) |
| is_fossil | Flag indicating mention of fossil fuel keywords | Engineered (Binary) |
| Market Cap | Total market value of a company's outstanding shares | Numerical |
| EBITDA | Earnings before interest, taxes, depreciation, and amortization | Numerical |
| Trailing PE | Trailing Price-to-Earnings ratio | Numerical |
| Beta | Measure of a stock's volatility in relation to the market | Numerical |
| Profit Margin | Ratio of profitability | Numerical |
| Debt To Equity | Ratio of total shareholder equity to debt | Numerical |

---

## Target Variable

The model is trying to predict the **ESG Risk Level** of a given company. Originally, the dataset contains 5 levels of risk. To create a stronger, single source of truth and reduce overlapping ambiguities, we simplified the target variable into 3 distinct classes:
- **Low** (Merges "Negligible" and standard "Low")
- **Medium**
- **High** (Merges "Severe" and standard "High")

---

## Data Types

- **Numerical:** Employee counts and all Yahoo Finance metrics.
- **Categorical:** Sector, Industry.
- **Text:** The raw string company descriptions (vectorized later).
- **Binary/Indicator:** Engineered flags for energy exposure (`is_renewable`, `is_fossil`).

---

## Data Distribution

- The target variable was simplified to 3 classes to improve class balance. 
- Words within the text descriptions vary significantly by risk level (explored with WordClouds), with high-risk companies frequently utilizing terminology related to traditional energy, extraction, and heavy industry. 
- A stratified 80/20 train-test split was used to ensure the class distribution remained consistent across training and evaluation phases.

---

## Data Quality

Several data quality issues were addressed during preprocessing:
- **Missing Targets:** Rows without an ESG Risk Level were dropped.
- **Formatting Issues:** The "Full Time Employees" column contained string formatting (commas) which required cleaning and conversion to floats.
- **Missing Text:** Empty descriptions were filled with empty strings.
- **Missing Financials:** The Yahoo Finance API sometimes fails to fetch data or returns nulls for certain metrics; these missing values were handled using imputation pipelines.

---

# 6. Data Preprocessing

1. **Target Mapping:** We grouped the 5-class ESG risk into 3 classes to improve model stability.
2. **Feature Engineering (Regex):** We created `is_renewable` and `is_fossil` binary features by scanning the `Description` column for specific industry keywords. This acts as a strong prior for ESG risk.
3. **Data Cleaning:** We stripped commas from the `Full Time Employees` column and cast it to numerical types.
4. **Train/Test Split:** We performed a stratified split *before* any major transformations to prevent data leakage.
5. **Categorical Encoding:** We applied `OneHotEncoder` with `handle_unknown="ignore"` to safely encode `Sector` and `Industry` without leaking test-set categories.
6. **Numerical Scaling & Imputation:** We used a `SimpleImputer` (median strategy) to handle missing values (especially from yfinance) and `StandardScaler` to normalize numerical ranges.
7. **Text Vectorization:** We fit a `TfidfVectorizer` (max 300 features, English stop words removed) solely on the training descriptions, yielding a sparse matrix of text features.

---

# 7. Modeling Approach

## Chosen Models

1. **Random Forest Classifier** (Used for Baseline, NLP, and Final Models)
2. **BART-MNLI (Facebook)** (Used for Zero-Shot text classification)

---

## Modeling Strategy

- **Baseline Model:** A Random Forest trained exclusively on structured data (Sector, Industry, Employees) to establish a performance floor.
- **NLP Integration:** We combined the structured data with the TF-IDF text vectors using `scipy.sparse.hstack` to see if text descriptions improve predictive power.
- **Zero-Shot NLP Model:** We leveraged a pre-trained Hugging Face transformer (`facebook/bart-large-mnli`) purely on the test set. By providing candidate labels ("Low ESG Risk", etc.), we evaluated how well an LLM understands ESG risk without any dataset-specific training.
- **Final Combined Model:** We merged structured data, TF-IDF text features, and real-time financial metrics fetched via `yfinance` into a final Random Forest model. 
- **Validation & Tuning:** `StratifiedKFold` (5 splits) was used during training to estimate variance and ensure robustness. `class_weight="balanced"` was applied to the Random Forests to penalize misclassifications in minority classes.

---

## Evaluation Metrics

- **F1-Macro Score:** Chosen as the primary metric. Because class imbalances often exist in risk ratings, F1-macro ensures the model is evaluated fairly across all classes, not just the majority class.
- **Accuracy:** Used as a secondary, easy-to-interpret metric.
- **Confusion Matrix:** Plotted for all 4 modeling approaches to visually inspect where the models are making errors (e.g., confusing Medium risk with High risk).



---

# 8. Project Structure

```text
├── data/
│   └── (Downloaded kaggle dataset CSVs)
├── financial_data_cache_2.csv    # Cached financial data to prevent repeated API calls
├── rf_final.joblib               # Saved final Random Forest model
├── preprocessor_final.joblib     # Saved sklearn column transformer
├── tfidf.joblib                  # Saved TF-IDF vectorizer
├── wordclouds_esg.png            # Generated exploratory data analysis plot
├── confusion_matrices.png        # Generated evaluation plots
├── feature_importances.png       # Generated feature importance plots
├── main.py                       # Main pipeline code (or Jupyter Notebook)
└── README.md                     # Project documentation
```

---

# 9. Installation


```bash
pip install pandas numpy matplotlib scikit-learn transformers yfinance kagglehub joblib wordcloud scipy
```
