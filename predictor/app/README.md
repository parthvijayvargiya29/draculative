Draculative Dashboard

Run locally:

1. Create and activate a Python virtualenv (recommended):

   python3 -m venv .venv
   source .venv/bin/activate

2. Install requirements:

   pip install -r predictor/app/requirements.txt

3. Run Streamlit app:

   streamlit run predictor/app/dashboard.py

Notes:
- The app imports the predictor code from `predictor/src` so run it from the repo root.
- For production or multi-user deployment, consider using a proper web server and caching.
