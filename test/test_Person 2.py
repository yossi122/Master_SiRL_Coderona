from datetime import timedelta
import os
import json

from src.run_utils import INITIAL_DATE
from src.seir import DiseaseState
from src.simulation.event import DayEvent
from src.simulation.params import Params
from src.simulation.simulation import Simulation
from src.world import Person,world


def test_immune_and_get_events1():
    config_path = os.path.join(os.path.dirname(__file__),"..","src","config.json")
    Expected  = -1
    with open(config_path) as json_data_file:
        ConfigData = json.load(json_data_file)
        paramsDataPath = ConfigData['ParamsFilePath']
    Params.load_from(os.path.join(os.path.dirname(__file__),"..","src", paramsDataPath), override=True)

    p = Person(30)
    events =  p.immune_and_get_events(INITIAL_DATE , timedelta(days =  15), \
        ((DiseaseState.SUSCEPTIBLE,timedelta(10)),(DiseaseState.IMMUNE,timedelta(21))))
    print(events)
    assert len(events) == 2
    
    persons_arr = [p]
    env_arr = []
    small_world = world.World(
        all_people = persons_arr,
        all_environments=env_arr,
        generating_city_name = "test",
        generating_scale = 1)
    
    my_simulation = Simulation(world = small_world, initial_date= INITIAL_DATE,interventions=[])
    assert p.get_disease_state() == DiseaseState.SUSCEPTIBLE
    for i in range(10):
        my_simulation.simulate_day()
    events[0].apply(simulation = my_simulation)
    assert p.get_disease_state() == DiseaseState.IMMUNE
    assert len(p._seir_times)==3
        
def test_immune_and_get_events2():
    config_path = os.path.join(os.path.dirname(__file__),"..","src","config.json")
    Expected  = -1
    with open(config_path) as json_data_file:
        ConfigData = json.load(json_data_file)
        paramsDataPath = ConfigData['ParamsFilePath']
    Params.load_from(os.path.join(os.path.dirname(__file__),"..","src", paramsDataPath), override=True)

    p = Person(30)
    events =  p.immune_and_get_events(INITIAL_DATE , timedelta(days =  20), \
        ((DiseaseState.SUSCEPTIBLE,timedelta(10)),(DiseaseState.LATENT,timedelta(5)), \
        (DiseaseState.SUSCEPTIBLE,None)))
    assert len(events) == 3

    persons_arr = [p]
    env_arr = []
    small_world = world.World(
        all_people = persons_arr,
        all_environments=env_arr,
        generating_city_name = "test",
        generating_scale = 1)
    
    my_simulation = Simulation(world = small_world, initial_date= INITIAL_DATE,interventions=[])
    
    assert p.get_disease_state() == DiseaseState.SUSCEPTIBLE
    for i in range(10):
        my_simulation.simulate_day()
    events[0].apply(simulation = my_simulation)
    assert p.get_disease_state() == DiseaseState.LATENT
    for i in range(5):
        my_simulation.simulate_day()
    events[1].apply(simulation = my_simulation)
    assert p.get_disease_state() == DiseaseState.IMMUNE
    

def test_immune_and_get_events3():
    config_path = os.path.join(os.path.dirname(__file__),"..","src","config.json")
    Expected  = -1
    with open(config_path) as json_data_file:
        ConfigData = json.load(json_data_file)
        paramsDataPath = ConfigData['ParamsFilePath']
    Params.load_from(os.path.join(os.path.dirname(__file__),"..","src", paramsDataPath), override=True)

    p = Person(30)
    events =  p.immune_and_get_events(INITIAL_DATE , timedelta(days =  20), \
        ((DiseaseState.SUSCEPTIBLE,timedelta(10)),(DiseaseState.LATENT,timedelta(5)),\
        (DiseaseState.ASYMPTOMATICINFECTIOUS,timedelta(5)),(DiseaseState.SUSCEPTIBLE,None)))
    assert len(events) == 4

    persons_arr = [p]
    env_arr = []
    small_world = world.World(
        all_people = persons_arr,
        all_environments=env_arr,
        generating_city_name = "test",
        generating_scale = 1)
    
    my_simulation = Simulation(world = small_world, initial_date= INITIAL_DATE,interventions=[])
    assert p.get_disease_state() == DiseaseState.SUSCEPTIBLE
    for i in range(10):
        my_simulation.simulate_day()
    events[0].apply(simulation = my_simulation)
    assert p.get_disease_state() == DiseaseState.LATENT
    for i in range(5):
        my_simulation.simulate_day()
    events[1].apply(simulation = my_simulation)
    assert p.get_disease_state() == DiseaseState.ASYMPTOMATICINFECTIOUS
    for i in range(5):
        my_simulation.simulate_day()
    events[2].apply(simulation = my_simulation)
    assert p.get_disease_state() == DiseaseState.IMMUNE
    

def test_immune_and_get_events4():
    config_path = os.path.join(os.path.dirname(__file__),"..","src","config.json")
    Expected  = -1
    with open(config_path) as json_data_file:
        ConfigData = json.load(json_data_file)
        paramsDataPath = ConfigData['ParamsFilePath']
    Params.load_from(os.path.join(os.path.dirname(__file__),"..","src", paramsDataPath), override=True)

    p = Person(30)
    events =  p.immune_and_get_events(INITIAL_DATE , timedelta(days =  3), \
        ((DiseaseState.LATENT , timedelta(days=10)),(DiseaseState.INCUBATINGPOSTLATENT, timedelta(days=3))))
    assert len(events) == 1
    
    persons_arr = [p]
    env_arr = []
    small_world = world.World(
        all_people = persons_arr,
        all_environments=env_arr,
        generating_city_name = "test",
        generating_scale = 1)
    
    my_simulation = Simulation(world = small_world, initial_date= INITIAL_DATE,interventions=[])
    my_simulation.simulate_day()
    my_simulation.simulate_day()
    my_simulation.simulate_day()

    for event in events:
        event.apply(simulation = my_simulation)
    assert p.get_disease_state() == DiseaseState.IMMUNE