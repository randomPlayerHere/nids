# what should be in the config folder and how sould the main.py should look like

Perfect — let’s make this concrete.

You already have the *shape* of the project; now let’s fill in what exactly goes inside the **`config` folder** and how a **production-ready `main.py`** should look so it can orchestrate the full cleaning → feature selection → (later modeling) pipeline.

---

## 🗂️ `config/` Folder Structure and Purpose

```
src/config/
│
├── settings.yaml     # Parameters for cleaning, feature selection, modeling
├── logger.yaml       # Logging level, format, handlers
└── paths.yaml        # Centralized file paths (optional)
```

---

### 1️⃣ `settings.yaml`

Holds *tunable parameters* and thresholds.

```yaml
cleaning:
  shuffle: true
  corr_threshold: 0.9
  downcast: true

  attack_mapping:
    BENIGN: Normal Traffic
    DoS Hulk: DoS
    DDoS: DDoS
    PortScan: Port Scanning
    DoS GoldenEye: DoS
    FTP-Patator: Brute Force
    DoS slowloris: DoS
    DoS Slowhttptest: DoS
    SSH-Patator: Brute Force
    Bot: Bots
    Web Attack - Brute Force: Web Attacks
    Web Attack - XSS: Web Attacks
    Web Attack - Sql Injection: Web Attacks
    Infiltration: Infiltration
    Heartbleed: Miscellaneous

  drop_attack_types:
    - Infiltration
    - Miscellaneous

feature_engineering:
  pca_variance: 0.95
  variance_threshold: 0.01
  model_based_top_k: 30

modeling:
  test_size: 0.2
  random_state: 42
```

---

### 2️⃣ `logger.yaml`

Controls your logging setup (replace print statements).

```yaml
version: 1
formatters:
  simple:
    format: "[%(asctime)s] %(levelname)s: %(message)s"
handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: simple
    stream: ext://sys.stdout
loggers:
  cicids:
    level: INFO
    handlers: [console]
    propagate: no
root:
  level: INFO
  handlers: [console]
```

---

### 3️⃣ (Optional) `paths.yaml`

Centralizes input/output paths so you don’t hardcode file locations.

```yaml
data:
  raw: "data/raw/"
  interim: "data/interim/"
  processed: "data/processed/"
logs: "logs/"
```

---

## 🧠 `main.py` — Orchestration Script

Here’s a practical, minimal version that:

* Loads all config files,
* Initializes logging,
* Runs data cleaning,
* (Later) runs feature selection/engineering.

```python
import os
import yaml
import logging.config
import pandas as pd
from src.data_cleaning.cleaner import DataCleaner

# -------------- Helpers -----------------
def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(default_path="src/config/logger.yaml"):
    config = load_yaml(default_path)
    logging.config.dictConfig(config)

# -------------- Main Pipeline -----------------
def main():
    # 1. Setup logging
    setup_logging()
    logger = logging.getLogger("cicids")

    # 2. Load configuration
    settings = load_yaml("src/config/settings.yaml")
    cleaning_cfg = settings["cleaning"]

    # 3. Define paths (or load from paths.yaml)
    input_path = "data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    interim_path = "data/interim/cicids_clean.csv"
    os.makedirs(os.path.dirname(interim_path), exist_ok=True)

    # 4. Load data
    logger.info(f"Loading dataset: {input_path}")
    df = pd.read_csv(input_path)

    # 5. Clean data
    logger.info("Starting data cleaning...")
    cleaner = DataCleaner(cleaning_cfg)
    df_clean = cleaner.clean(df)
    logger.info(f"Cleaning complete. Rows: {len(df_clean)}, Columns: {len(df_clean.columns)}")

    # 6. Save cleaned data
    df_clean.to_csv(interim_path, index=False)
    logger.info(f"Cleaned data saved to {interim_path}")

    # 7. (Optional) Continue pipeline
    # from src.feature_engineering.selector import FeatureSelector
    # fs = FeatureSelector(settings['feature_engineering'])
    # df_selected = fs.select(df_clean)
    # df_selected.to_csv("data/processed/cicids_selected.csv", index=False)

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
```

---

## 🧾 Summary of What Goes Where

| File                       | Purpose                                                                  |
| -------------------------- | ------------------------------------------------------------------------ |
| **`settings.yaml`**        | Central hub for configurable parameters and thresholds.                  |
| **`logger.yaml`**          | Defines how logs are formatted and displayed.                            |
| **`paths.yaml`**           | (Optional) Keeps I/O paths consistent across modules.                    |
| **`main.py`**              | High-level controller: loads config, initializes logging, runs pipeline. |
| **`cleaner.py`**           | Implements actual cleaning logic.                                        |
| **`feature_engineering/`** | Later modules for PCA, correlation filters, etc.                         |

---

## 🧱 Scalability Tip

As your project grows, consider using **Hydra** or **OmegaConf** for configuration management — it lets you compose multiple YAML configs dynamically, which is ideal if you’ll experiment with variants (e.g., “PCA-only run”, “tree-based model run”, etc.).

---

Would you like me to extend this with the **feature engineering/selection** modules structured in the same modular way (with configs for PCA, correlation filtering, and model-based selection)? That would naturally follow from this setup.


