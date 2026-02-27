from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


def log_method(func):
    """Decorator der Methodenaufrufe loggt."""
    def wrapper(self, *args, **kwargs):
        logger.info(f"{self.__class__.__name__}.{func.__name__}() – {len(args[0])} Samples")
        return func(self, *args, **kwargs)
    return wrapper


class ClassifierStrategy(ABC):
    @abstractmethod
    def fit(self, X, y):
        #Trainiert das Modell mit Features X und Labels y
        pass

    @abstractmethod
    def predict(self, X):
        # Liefert Vorhersagen für die gegebenen Features X
        pass