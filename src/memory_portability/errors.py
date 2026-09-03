class MemoryPortabilityError(Exception):
    """Base error for user-memory portability operations."""


class ArchiveValidationError(MemoryPortabilityError):
    """The supplied archive is malformed, unsafe, or incompatible."""


class MemoryImportError(MemoryPortabilityError):
    """A validated archive could not be applied to live memory."""


class RecoveryError(MemoryPortabilityError):
    """An interrupted import could not be recovered automatically."""
