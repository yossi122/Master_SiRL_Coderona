from src.simulation.simulation import Simulation
class HelloWorldExtension(Simulation):
    def __init__(self,parent: Simulation):
        print("In printing Ctor")

    def start_of_day_processing(self):
        pass
    
    def end_of_day_processing(self):
        print("Hello World")