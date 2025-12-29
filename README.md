## Maternal Mortality Analytics 

### Project Concept

**Goal:** Build an end-to-end health analytics pipeline that:

-   Reproducible Ingestion and tidy transformation.

-   Exploratory analytics (trends, regional comparisons, missingness).

-   Fit interpretable statistical models (baseline).

-   ML extension for forecasting/predicting “next-year's MMR” and risk profiling.

**Outputs:**

-   A reproducible analysis report (Markdown).

-   Cleaned “tidy” dataset saved to /data/processed/

-   Model evaluation summary tables (baseline + ML).

-   A small set of charts in /reports/figs/

Repo Structure

```{bash}
maternal-mortality-analytics/
├── README.md
├── data/
│   ├── raw/
│   └── processed/
├── reports/
│   ├── figs/
│   └── report.md  
├── script/
│   ├── ingest.py    
│   ├── clean.py     
│   ├── eda.py       
│   └── model.py     
└── environment/
    ├── requirements.txt 
    └── Makefile 
```

### Python Scripts

Each Python script was tested in JupyterLab before importing into the shell script to run on WSL.
