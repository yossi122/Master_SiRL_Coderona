from io import StringIO
import json
import os
from numpy.core.numeric import Infinity
import pandas  as pd 
import random

from src.logs.summary import TableFormat
from collections import Counter
from src.seir.seir_times import daysdelta
from src.run_utils import INITIAL_DATE 
from src.seir import DiseaseState
from src.simulation.simulation import Simulation
from src.simulation.params import Params
from src.world import Person
from src.world.world import World
from src.world.population_generation import population_loader,generate_city

#Test the amount of the  created Immune
def test_CreateDeltaFile(helpers):
    helpers.clean_outputs()
    config_path = os.path.join(os.path.dirname(__file__),"..","src","config.json")
    with open(config_path) as json_data_file:
        ConfigData = json.load(json_data_file)
        paramsDataPath = ConfigData['ParamsFilePath']
    Params.load_from(os.path.join(os.path.dirname(__file__),"..","src", paramsDataPath), override=True)

    ageList = [random.randint(0,40) for i in range(10)]
    PersonList = list(map(Person, ageList))
    events_acc = []
    for person in PersonList:
        states_table = ((DiseaseState.LATENT,daysdelta(3)),
                        (DiseaseState.ASYMPTOMATICINFECTIOUS,daysdelta(3)),
                        (DiseaseState.IMMUNE, daysdelta(3)),
                        (DiseaseState.IMMUNE, None))
        events = person.gen_and_register_events_from_seir_times(date = INITIAL_DATE,states_and_times= states_table)
        events_acc += events
        # person.set_disease_state(DiseaseState.LATENT)
    env_arr = []
    my_world = World(
        all_people = PersonList,
        all_environments=env_arr,
        generating_city_name = "test",
        generating_scale = 1)

    my_simulation = Simulation(world = my_world, initial_date= INITIAL_DATE)
    my_simulation.register_events(events_acc)
    my_simulation.run_simulation(num_days=10,name="test")
    #assert events dictionary is not empty
    txt = my_simulation.stats.get_state_stratified_summary_table(table_format=TableFormat.CSV)
    test_data = StringIO(txt)
    tbl = pd.read_csv(test_data)
    assert len(tbl)==7
    
    print(tbl)
    
    assert tbl.iloc[0,DiseaseState.SUSCEPTIBLE.value] == 10
    assert tbl.iloc[3,DiseaseState.ASYMPTOMATICINFECTIOUS.value] == 10
    assert tbl.iloc[6,DiseaseState.IMMUNE.value] == 10

#Test the amount of the  created Immune
def test_CreateDeltaFileAtlit(helpers):
    helpers.clean_outputs()
    config_path = os.path.join(os.path.dirname(__file__),"..","src","config.json")
    with open(config_path) as json_data_file:
        ConfigData = json.load(json_data_file)
        citiesDataPath = ConfigData['CitiesFilePath']
        paramsDataPath = ConfigData['ParamsFilePath']
    Params.load_from(os.path.join(os.path.dirname(__file__),"..","src", paramsDataPath), override=True)
    Params.loader()["person"]["state_machine_type"] = "SIR"

    DiseaseState.init_infectiousness_list()
    pop = population_loader.PopulationLoader(citiesDataPath)
    my_world = pop.get_world(city_name = 'Atlit',scale = 1,is_smart= False)

    sim = Simulation(world = my_world, initial_date= INITIAL_DATE)
    sim.infect_random_set(num_infected =500, infection_doc = "")
    sim.run_simulation(num_days=180,name="test")
    #assert events dictionary is not empty
    txt = sim.stats.get_state_stratified_summary_table(table_format=TableFormat.CSV)
    test_data = StringIO(txt)
    
    tbl = pd.read_csv(test_data)
    cnt_start  = tbl.iloc[0,DiseaseState.SUSCEPTIBLE.value] + tbl.iloc[0,DiseaseState.LATENT.value] 
    cnt_end  = 0
    for i in range(len(tbl)):
        cnt_end = cnt_end + tbl.iloc[i,DiseaseState.IMMUNE.value] + tbl.iloc[i,DiseaseState.DECEASED.value] 
    plug_number = len([p for p in sim._world.all_people() if p.get_disease_state() == DiseaseState.SUSCEPTIBLE])
    assert cnt_start >= cnt_end + plug_number
