# explore_module.py
import semantic_router
import inspect

# Print all available attributes in the semantic_router module
print("Available in semantic_router module:")
print(dir(semantic_router))

# For each attribute, let's see what type it is
for attr in dir(semantic_router):
    if not attr.startswith('__'):  # Skip internal attributes
        try:
            obj = getattr(semantic_router, attr)
            print(f"{attr}: {type(obj)}")
            
            # If it's a module, let's look inside it
            if inspect.ismodule(obj):
                print(f"  Contents of {attr} module:")
                for sub_attr in dir(obj):
                    if not sub_attr.startswith('__'):
                        print(f"    {sub_attr}")
        except Exception as e:
            print(f"Error inspecting {attr}: {e}")