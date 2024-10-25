import os
import json
from src.extensions.ImmuneByAgeExtension import ImmuneByAgeExtension

from src.run_utils import INITIAL_DATE
from src.seir import DiseaseState
from src.simulation.simulation import Simulation
from src.simulation.params import Params
from src.world import Person,World

#Test the amount of the  created Immune
def test_createInfectedPersons():
    """
    #create population with 5 people ages [9,19,29,39,49]
    #test that each day one of them is getting immuned
    """
    #Editig confige file saving is nessery 
    config_path = os.path.join(os.path.dirname(__file__),"..","src","config.json")
    ConfigData = None
    
    #reading the confige file 
    with open(config_path) as json_data_file:
        ConfigData = json.load(json_data_file)
        paramsDataPath = ConfigData['ParamsFilePath']
    Params.load_from(os.path.join(os.path.dirname(__file__),"..","src", paramsDataPath), override=True)

    persons_arr = list(map(Person, [9,19,29,39,49,59]))
    assert len(persons_arr) == 6
    env_arr = []
    my_world = World(
        all_people = persons_arr,
        all_environments=env_arr,
        generating_city_name = "test",
        generating_scale = 1)

    my_simulation = Simulation(world = my_world, initial_date= INITIAL_DATE)
    my_simulation.run_simulation(7,"test_simulation",datas_to_plot = None,extensionsList = ["ImmuneByAgeExtension","EmptyExtension"] )
    cnt = sum([1 for p in persons_arr if p.get_disease_state() == DiseaseState.IMMUNE])
    assert cnt == 3
    