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
    
    @abstractmethod 
    def set_parameters(self,*params) -> None:
        """Setzt Parameter des Modells basierend auf den MC Parametern wenn nötig"""
        ...
    @property
    @abstractmethod
    def noise_dim(self) -> int:
        """Anzahl benötigter Zufallszahlen pro Zeitschritt."""
        ...