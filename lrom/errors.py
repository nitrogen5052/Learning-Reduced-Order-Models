"""Public LROM exception hierarchy."""


class LROMError(Exception):
    """Base error for the public LROM workflow."""


class LROMConfigurationError(LROMError, ValueError):
    """Invalid immutable physical configuration."""


class LROMSamplingError(LROMError, ValueError):
    """Invalid sampling request or returned sample state."""


class LROMStateError(LROMError, RuntimeError):
    """Workflow method called in an invalid lifecycle state."""


class LROMArtifactError(LROMError, ValueError):
    """Invalid or incompatible portable emulator artifact."""
