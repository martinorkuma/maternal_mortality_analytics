# 

# Maternal Mortality Analytics: A Reproducible WSL Pipeline with Panel Modeling and Forecasting 
mkdir -p maternal-mortality-analytics/{data/raw,data/processed,reports/figs,script,environment} # -p creates the parent dir first
cd maternal-mortality-analytics

# Install the Python environ
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

pip install pandas numpy matplotlib scikit-learn statsmodels openpyxl pyarrow
pip freeze > environment/requirements.txt 

# Copy files to data/raw
cp Metadata_Country_API_SH.STA.MMRT_DS2.csv data/raw/
cp Metadata_Indicator_API_SH.STA.MMRT_DS2.csv data/raw/
cp API_SH.STA.MMRT_DS2.csv data/raw/
cp "2.14 Reproductive health_Metadata.xls" data/raw/

# Create the Pythons scripts
touch script/01_ingest_tidy.py script/02_eda.py script/03_model.py

# I created the content of eaxh .py script in JupyterLab and then ran them in WSL.
# Ensure pyarrow is installed in WSL: pip install pyarrow
python3 script/01_ingest_tidy.py
python3 script/02_eda.py
python3 script/03_model.py

# Test if script 03 worked
sed -n '1,35p' data/processed/panel_model_summary.txt 

nano .gitignore 
    .venv/
    __pycache__/
    *.pyc
    data/raw/
    data/processed/*.parquet
    data/processed/mmr_long.csv
    reports/figs/

cat .gitignore

# Commit to GitHub
git init
git branch -m main 
git add .
git commit -m "Initial pipeline: ingest, EDA, panel model, etc"

# Use SSH instead of https
git remote add origin git@github.com:martinorkuma/maternal_mortality_analytics.git
ssh -T git@github.com # Test GitHb SSH authentication
git push -u origin main # Push repo to GitHub

git status # Check status of push
