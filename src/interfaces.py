from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

def log_method(func):
    # Decorator-Pattern für Logging
    def wrapper(self, *args, **kwargs):
        logger.info(f"{self.__class__.__name__}.{func.__name__}() – {len(args[0])} Samples")
        return func(self, *args, **kwargs)
    return wrapper

class ClassifierStrategy(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass