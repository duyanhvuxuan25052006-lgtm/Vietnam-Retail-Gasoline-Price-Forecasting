import os
import sys
import time
import subprocess

subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
# Set encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8')

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_DIR, "src")

def run_script(script_name):
    print(f"\n==========================================")
    print(f" Running {script_name}...")
    print(f"==========================================")
    
    script_path = os.path.join(SRC_DIR, script_name)
    if not os.path.exists(script_path):
        print(f" ERROR: Missing script {script_path}.")
        sys.exit(1)

    start_time = time.time()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run([sys.executable, "-u", script_path], cwd=PROJECT_DIR, env=env, capture_output=False)
    
    elapsed = time.time() - start_time
    if result.returncode == 0:
        print(f" Finished {script_name} successfully in {elapsed:.2f} seconds.")
    else:
        print(f" ERROR: {script_name} failed with exit code {result.returncode}.")
        sys.exit(result.returncode)

def main():
    os.makedirs(os.path.join(PROJECT_DIR, "data"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_DIR, "output"), exist_ok=True)
    print("Starting Automated Price Forecasting Pipeline...")
    
    # 1. Fetch & Merge Data
    run_script("data_fetcher.py")
    
    # 2. Run Cross-Validation & Model Training
    run_script("models.py")
    
    # 3. Generate Sensitivity Analysis & Next Thursday Predictions
    run_script("analysis.py")
    
    # 4. Generate Jupyter Notebook
    run_script("create_notebook.py")
    
    print("\n==========================================")
    print(" Pipeline completed successfully!")
    print(" Output files generated in the 'output' directory:")
    print(" - output/forecast_this_thursday.csv (Next Thursday Forecasts & 95% CI)")
    print(" - output/model_comparison.csv (MAPE comparison across models)")
    print(" - output/policy_cycle_comparison.csv (7-day vs 10-day weighted MAPE)")
    print(" - output/sensitivity_matrix.csv (Brent vs BOG scenario grid)")
    print(" - output/sensitivity_heatmap.png (Brent vs BOG sensitivity matrix)")
    print(" - output/policy_analysis.png (7-day vs 10-day policy cycle comparison)")
    print("==========================================\n")

if __name__ == "__main__":
    main()
