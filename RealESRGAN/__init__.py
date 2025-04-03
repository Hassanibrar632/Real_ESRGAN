from .model import RealESRGAN  # Import the class

__all__ = ["RealESRGAN"]  # Set what gets imported when using `import RealESRGAN`

# Define a direct alias for instantiation
def __getattr__(name):
    if name == "RealESRGAN":
        return RealESRGAN
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
