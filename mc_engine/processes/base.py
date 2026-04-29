from abc import ABC, abstractmethod

class StochasticProcess(ABC):

    @property
    @abstractmethod
    def process_type(self) -> str:
        ...

    @abstractmethod
    def to_cpp_params(self) -> dict:
        """Serialisiert Parameter für den C++ Kern."""
        ...

    @property
    @abstractmethod
    def noise_dim(self) -> int:
        """Anzahl benötigter Zufallszahlen pro Zeitschritt."""
        ...