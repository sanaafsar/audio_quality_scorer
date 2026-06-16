"""BaseQualityModel — shared contract for all ML quality models."""

from __future__ import annotations

from abc import ABC, abstractmethod

# Type alias for the score dictionary returned by all models
ScoreDict = dict[str, float]


class BaseQualityModel(ABC):
    """Shared interface for audio, video, and text quality models.

    Lifecycle::

        model = MyModel()
        model.load()            # load weights / resources once
        scores = model.predict(...)  # call as many times as needed

    Subclasses narrow the ``predict()`` signature to match their input type.
    """

    def __init__(self) -> None:
        """Initialise the load-state flag.  Subclasses override and may extend."""
        self._loaded: bool = False

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier (name + version)."""

    @abstractmethod
    def load(self) -> None:
        """Load model weights or initialise resources.

        Called once before the first ``predict()``.  Sets ``self._loaded = True``.
        """

    @abstractmethod
    def predict(self, *args, **kwargs) -> ScoreDict:
        """Run inference and return a dict of metric name → float value.

        All returned values should be in a documented range.  An ``"overall"``
        key in ``[0.0, 1.0]`` is a convention (1.0 = best quality).
        """

    @property
    def is_loaded(self) -> bool:
        """True after ``load()`` has set ``self._loaded = True``."""
        return self._loaded
