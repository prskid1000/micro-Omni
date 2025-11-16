"""
Runtime compatibility fix for PyTorch 2.8.0 + Triton 3.5.0
This patches the triton_key import issue without modifying PyTorch source files.

Import this module BEFORE importing torch to apply the fix:
    import fix_triton_compat
    import torch
"""
import sys
from types import ModuleType


class TritonCompilerShim(ModuleType):
    """Shim module that provides backward compatibility for triton_key."""
    
    def __init__(self, original_module):
        self._original_module = original_module
        # Copy all attributes from original module
        for attr in dir(original_module):
            if not attr.startswith('_'):
                try:
                    setattr(self, attr, getattr(original_module, attr))
                except AttributeError:
                    pass
    
    def __getattr__(self, name):
        # If triton_key is requested, redirect to get_cache_key
        if name == 'triton_key':
            try:
                return getattr(self._original_module, 'get_cache_key')
            except AttributeError:
                raise ImportError(
                    f"Cannot import name '{name}' from 'triton.compiler.compiler'. "
                    f"This may indicate a version mismatch between PyTorch and Triton."
                )
        return getattr(self._original_module, name)


def apply_triton_compatibility_patch():
    """Apply the compatibility patch for Triton 3.5.0 with PyTorch 2.8.0."""
    try:
        # Import the actual triton.compiler.compiler module
        import triton.compiler.compiler as original_compiler
        
        # Check if patch is needed (triton_key missing but get_cache_key exists)
        has_triton_key = hasattr(original_compiler, 'triton_key')
        has_get_cache_key = hasattr(original_compiler, 'get_cache_key')
        
        if not has_triton_key and has_get_cache_key:
            print("🔧 Applying Triton 3.5.0 compatibility patch for PyTorch 2.8.0...")
            
            # Replace the module in sys.modules with our shim
            shim = TritonCompilerShim(original_compiler)
            sys.modules['triton.compiler.compiler'] = shim
            
            print("✓ Triton compatibility patch applied successfully!")
            print("  triton_key → get_cache_key (aliased)")
            return True
        elif has_triton_key:
            print("ℹ️  Triton compatibility patch not needed (triton_key already exists)")
            return False
        else:
            print("⚠️  Warning: Neither triton_key nor get_cache_key found in triton.compiler.compiler")
            return False
            
    except ImportError as e:
        print(f"⚠️  Could not apply Triton compatibility patch: {e}")
        print("   Triton may not be installed or torch.compile() will not work.")
        return False


# Auto-apply patch on import
if __name__ != "__main__":
    apply_triton_compatibility_patch()

