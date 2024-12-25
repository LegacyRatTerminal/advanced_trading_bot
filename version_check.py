import sys
import numpy
import tensorflow as tf
import platform

def print_versions():
    print("🖥️ System Information:")
    print(f"Python Version: {sys.version}")
    print(f"Operating System: {platform.platform()}")
    
    print("\n📦 Library Versions:")
    print(f"NumPy Version: {numpy.__version__}")
    print(f"TensorFlow Version: {tf.__version__}")

if __name__ == "__main__":
    print_versions()