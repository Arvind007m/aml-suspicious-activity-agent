# AI-Powered Anti-Money-Laundering (AML) Detection Agent

An intelligent, autonomous Anti-Money-Laundering (AML) agent powered by an **LLM Dynamic Planner** (`Groq API`, model `llama-3.3-70b-versatile`) and modular tools. 

Unlike traditional fixed sequential pipelines, this system is a **true autonomous agent**—it evaluates natural language user queries, dynamically constructs a structured JSON execution plan, selects only the required tools, and deliberately skips unnecessary computation before executing analysis.

---

## 🌟 Key Highlights & Features

- **True Dynamic Agent Planning**: Uses Groq LLM to parse natural language queries into structured JSON execution plans before running tools.
- **Graceful Fallback Planner**: Includes a deterministic rule-engine fallback so the agent never crashes, even if API keys are missing or offline.
- **Modular Tool Architecture**:
  - `eda`: Exploratory data profiling & distribution stats.
  - `feature_engineering`: Transaction velocity, 24h rolling windows, structuring ($9k-$10k) counts, rapid cash-out indicators.
  - `anomaly_detection`: Unsupervised machine learning (`scikit-learn` `IsolationForest`).
  - `risk_classification`: Multi-factor risk scoring & Low/Medium/High categorization.
  - `explanation`: Plain-English narrative generation with single-request batched LLM prompts or templates, plus actionable escalation recommendations (`File SAR`, `EDD Review`, `Routine Monitoring`).
- **Ground Truth Evaluation Metrics**: Computes **Precision**, **Recall**, **F1-Score**, and **Detection Hit Rate** using dataset ground-truth labels (`is_laundering`).
- **Visual Chart Artifacts**: Automatically saves high-resolution summary charts (`charts/latest_analysis.png`).
- **Multiple Interfaces**: Command Line Interface (CLI), Master Demo Runner (`run_demo.py`), and Interactive Streamlit Dashboard (`app.py`).

---

## 🏗️ System Architecture

```
                       +-------------------------+
                       |    User Query (CLI/UI)  |
                       +------------+------------+
                                    |
                                    v
                       +-------------------------+
                       |   planner.py (Groq LLM) |
                       | Extracts intent, entity,|
                       | filters, aml_pattern &  |
                       | builds structured PLAN  |
                       | (Logs LLM vs Fallback)  |
                       +------------+------------+
                                    |
                                    v
                       +-------------------------+
                       |   registry.py & Tools   |
                       | Validates & executes    |
                       | target tools in sequence|
                       +------------+------------+
                                    |
            +-----------------------+-----------------------+
            |                       |                       |
            v                       v                       v
     [eda.py]            [feature_engineering.py]   [anomaly_detection.py]
  (Data profiling)       (Velocity, Structuring)    (IsolationForest ML)
            |                       |                       |
            +-----------------------+-----------------------+
                                    |
                                    v
                       +-------------------------+
                       | risk_classification.py  |
                       | Categorizes Low/Med/High|
                       +------------+------------+
                                    |
                                    v
                       +-------------------------+
                       |     explanation.py      |
                       | Batched/Templated LLM   |
                       | Plain-English reason &  |
                       | Escalation action       |
                       +------------+------------+
                                    |
                                    v
                       +-------------------------+
                       |     orchestrator.py     |
                       | Judge-Facing Summary    |
                       | + Ground Truth Metrics  |
                       | + PNG Chart Artifact    |
                       +-------------------------+
```

---

## 🔀 Dynamic Planning & 4 Demo Query Behaviors

The core graded criterion is **visible, verifiable agentic branching**. Below are the 4 demo queries and their distinct execution plans:

| # | User Query | Intent | Tools Executed | Deliberately Skipped Tools | Rationale / Behavior |
|---|---|---|---|---|---|
| **1** | `"Analyse this dataset for suspicious activity"` | `broad_analysis` | `eda`, `feature_engineering`, `anomaly_detection`, `risk_classification`, `explanation` | *None* | Full end-to-end dataset profiling, feature engineering, ML anomaly scoring, risk scoring, and explanations. |
| **2** | `"Find structuring patterns in the last 30 days"` | `pattern_detection` | `feature_engineering`, `anomaly_detection`, `risk_classification`, `explanation` | `eda` | Focuses on structuring indicators (multiple cash/wire txns under $10k threshold) within 30-day window. Skips exploratory data analysis. |
| **3** | `"Which customers made 10+ transactions under $10,000?"` | `threshold_query` | `feature_engineering`, `risk_classification`, `explanation` | `eda`, `anomaly_detection` | Pure aggregation and threshold filtering (**NO ML scoring**). Skips EDA and IsolationForest anomaly detection. |
| **4** | `"Is customer 4521 suspicious?"` | `single_entity` | `anomaly_detection`, `risk_classification`, `explanation` | `eda`, `feature_engineering` | Single-entity lookup for `customer_id: 4521`. Directly evaluates entity anomaly score and risk level without dataset-wide feature extraction or EDA. |

---

## 📊 Dataset Schema & Synthetic Generation Logic

The project includes a synthetic dataset generator (`generate_data.py`) modeled after IBM AML synthetic transaction specifications (~5,000 rows CSV):

| Column Name | Data Type | Description |
|---|---|---|
| `timestamp` | ISO 8601 String | Transaction timestamp (`YYYY-MM-DD HH:MM:SS`) |
| `transaction_id` | String | Unique transaction ID (e.g. `TXN_100042`) |
| `customer_id` | String | **Primary entity key** (e.g. `4521`, `1089`) |
| `sender_account` | String | Originating bank account |
| `receiver_account` | String | Destination bank account |
| `amount` | Float | Transaction amount in USD |
| `currency` | String | Transaction currency (`USD`, `EUR`, `GBP`) |
| `payment_format` | String | Payment channel (`Cash Deposit`, `Wire`, `ACH`, `Credit Card`) |
| `is_laundering` | Integer | Ground truth binary label (`1` = Laundering, `0` = Clean) |

### Baked-in Laundering Typologies:
1. **Structuring / Smurfing**: Customer `4521` and Customer `1089` execute 15 rapid cash deposits of $9,100–$9,950 (just below the $10,000 reporting threshold) within 24 hours.
2. **Rapid Cash-Out / Velocity Spike**: Customer `3310` receives a $180,000 wire deposit and immediately transfers out 98% in multiple split wires within 45 minutes.
3. **Extreme Amount Anomaly**: Customer `8802` with typical $20-$50 transactions suddenly executes a $495,000 wire transfer.

---

## 💻 Tech Stack

- **Language**: Python 3.11+
- **LLM Engine**: Groq API (`llama-3.3-70b-versatile`) via `groq` SDK
- **Data & ML**: `pandas`, `numpy`, `scikit-learn` (`IsolationForest`)
- **Visualization**: `matplotlib`
- **User Interfaces**: CLI (`orchestrator.py`), Demo Runner (`run_demo.py`), Streamlit UI (`app.py`)
- **Environment Management**: `python-dotenv`

---

## 🚀 Setup & Running Instructions

### 1. Clone & Environment Setup
```bash
git clone <repository_url>
cd aml-suspicious-activity-agent

# Create virtual environment (optional but recommended)
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Groq API Key (Optional)
Copy `.env.example` to `.env` and set your key:
```bash
cp .env.example .env
```
Inside `.env`:
```env
GROQ_API_KEY=your_actual_groq_api_key_here
```
> *Note: If `GROQ_API_KEY` is not set, the agent automatically falls back to the deterministic rule-engine planner without crashing.*

### 3. Generate Synthetic Dataset
```bash
python generate_data.py
```

---

## 🎬 Running the Agent & Demos

### Option A: Master Demo Runner (Recommended for Judges)
Runs all 4 queries sequentially to showcase dynamic agentic branching:
```bash
python run_demo.py
```

### Option B: Command Line Interface (CLI)
Run any natural language query directly:
```bash
python orchestrator.py "Find structuring patterns in the last 30 days"
python orchestrator.py "Is customer 4521 suspicious?"
```

### Option C: Interactive Streamlit Dashboard
Launch the web interface for visual query exploration:
```bash
streamlit run app.py
```

---

## 🤖 AI Tools Disclosure

In compliance with hackathon regulations, this project disclaims the use of AI tools during development:
- **Antigravity AI Agent** (Google DeepMind pair programmer) was used for initial architectural planning, skeleton code generation, and test suite verification.
- **Gemini 3.6 Flash** model was utilized during interactive pair programming sessions.
- **Groq API (`llama-3.3-70b-versatile`)** powers the runtime autonomous dynamic planner and explanation engine in the live application.