import logging
import sys

# Configure logging to see the instrumentation
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

print(f"Python executable: {sys.executable}")

print("Attempting to import gliner...")
try:
    from gliner2 import GLiNER2
    print("Success: gliner2 imported.")
except ImportError as e:
    print(f"Failed to import gliner2: {e}")
except Exception as e:
    print(f"Error during gliner2 import: {e}")

print("Attempting to import fastcoref...")
try:
    from fastcoref import FCoref
    print("Success: fastcoref imported.")
except ImportError as e:
    print(f"Failed to import fastcoref: {e}")
except Exception as e:
    print(f"Error during fastcoref import: {e}")

print("Attempting to import infinity_emb...")
try:
    from infinity_emb import EngineArgs, AsyncEmbeddingEngine
    print("Success: infinity_emb imported.")
except ImportError as e:
    print(f"Failed to import infinity_emb: {e}")
except Exception as e:
    print(f"Error during infinity_emb import: {e}")
