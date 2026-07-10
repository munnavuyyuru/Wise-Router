import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.engine import RoutingEngine

print("RoutellM — Model Router / Cost Optimizer")
print("Starting Streamlit app...")
print("Run: streamlit run streamlit_app/Home.py")
