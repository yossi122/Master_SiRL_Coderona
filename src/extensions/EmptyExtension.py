from src.simulation.simulation import Simulation
class EmptyExtension(Simulation):
    def __init__(self,parent: Simulation):
        pass
    
    def start_of_day_processing(self):
        pass
    
    def end_of_day_processing(self):
        pass