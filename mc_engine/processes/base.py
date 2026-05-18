from abc import ABC, abstractmethod

class StochasticProcess(ABC):
    """
    Abstract class for a stochastic process
    """

    @property
    @abstractmethod
    def process_type(self) -> str:
        ...

    @abstractmethod
    def to_cpp_params(self) -> dict:
        """Serealizes Parameters for C++ processing"""
        ...
    
    @abstractmethod 
    def set_parameters(self,params:dict) -> None:
        """Set p"""
        ...