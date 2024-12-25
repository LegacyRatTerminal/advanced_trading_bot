import sys
import subprocess
import platform

def check_system_details():
    """Display comprehensive system information"""
    print("🖥️ System Details:")
    print(f"Operating System: {platform.system()}")
    print(f"Python Version: {platform.python_version()}")
    print(f"Python Executable: {sys.executable}")

def resolve_numpy_tensorflow_conflict():
    """
    Resolve NumPy and TensorFlow version conflicts
    """
    dependencies = [
        # Specific version of NumPy compatible with TensorFlow
        'numpy==1.26.0',
        
        # TensorFlow with compatible NumPy version
        'tensorflow-intel==2.15.0',
        
        # Additional scientific computing libraries
        'pandas',
        'scikit-learn',
        'scipy'
    ]
    
    for package in dependencies:
        try:
            subprocess.check_call([
                sys.executable, 
                "-m", "pip", 
                "install", 
                "--upgrade",
                package
            ])
            print(f"✅ Installed/Upgraded: {package}")
        
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install: {package}")

def verify_installations():
    """Verify critical module imports"""
    test_modules = [
        'numpy',
        'tensorflow',
        'pandas',
        'sklearn'
    ]
    
    for module in test_modules:
        try:
            __import__(module)
            print(f"✅ Successfully imported: {module}")
        except ImportError:
            print(f"❌ Failed to import: {module}")

def main():
    print("🔍 Dependency Conflict Resolution")
    check_system_details()
    resolve_numpy_tensorflow_conflict()
    verify_installations()

if __name__ == "__main__":
    main()