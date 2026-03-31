import sys
import os
sys.path.append(os.path.abspath(os.path.curdir))
try:
    import modules
    print("Imported modules from", modules.__file__)
except Exception as e:
    print("Failed to import modules:", e)
try:
    import configs
    print("Imported configs from", configs.__file__)
except Exception as e:
    print("Failed to import configs:", e)
EOF && python3 test_import.py && rm test_import.py
