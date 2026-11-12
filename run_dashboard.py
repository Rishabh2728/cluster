"""
Dashboard Runner Script for INT315 Cluster Computing Project
Predicting Insurance Claim Approval Using Apache Spark
Student: Rishabh Kumar | Roll No: 49 | Reg No: 12310491 | Section: K2P23PD
"""

import subprocess
import sys
from pathlib import Path

def main():
    dashboard_script = Path(__file__).parent / "dashboard" / "app.py"
    if not dashboard_script.exists():
        print(f"Error: Dashboard script not found at {dashboard_script}")
        sys.exit(1)

    print("===================================================================")
    print("  LAUNCHING STREAMLIT DASHBOARD — INT315 CLUSTER COMPUTING")
    print("  PREDICTING INSURANCE CLAIM APPROVAL USING APACHE SPARK")
    print("  Student: Rishabh Kumar (Reg No: 12310491 | Roll No: 49)")
    print("  URL: http://localhost:8501")
    print("===================================================================")

    cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard_script), "--server.port=8501"]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user.")

if __name__ == "__main__":
    main()
