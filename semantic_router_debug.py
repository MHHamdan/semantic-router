# semantic_router_debug.py
import os
import inspect
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("Debugging semantic_router package...")

# Let's check the version
import semantic_router
print(f"Semantic Router version: {semantic_router.__version__}")

# Let's look at the SemanticRouter implementation
from semantic_router import SemanticRouter
print("\nSemanticRouter class definition:")
print(inspect.getsource(SemanticRouter))

# Check for examples in the repo
import glob
print("\nLooking for example files in the repo:")
example_files = glob.glob("examples/*.py") + glob.glob("*.py")
for file in example_files:
    if os.path.isfile(file) and "test_" not in file and "setup" not in file:
        print(f"Found example file: {file}")

# Try to find a minimal working example
print("\nAttempting to find minimal working example code...")
for file in example_files:
    if os.path.isfile(file):
        with open(file, 'r') as f:
            content = f.read()
            if "SemanticRouter" in content and "Route" in content:
                print(f"Found potential example in: {file}")
                print("First 20 lines:")
                print("\n".join(content.split("\n")[:20]))
                print("...")