from datetime import date
import json
from numpy.lib.function_base import median
import os
import random

from src.simulation.params import Params
from src.seir.disease_state import DiseaseState
from src.world.person import Person
from src.world.population_generation import population_loader
from src.world.environments.homogeneous_environment import HomogeneousEnvironment
from src.world.environments import NeighborhoodCommunity

def test_propagate_infection(params_path):
    """
    tests the consistency of the propagate_infection function in HomogenousEnvironment
    """
    Params.load_from(params_path)
    DiseaseState.init_infectiousness_list()
    percent_infected = 0.1
    contact_prob = 0.25
    num_of_people = 400
    weight_list = [max(0.3, random.random()-0.1) for _ in range(num_of_people)]
    env = HomogeneousEnvironment(contact_prob, "test")
    total = 0
    infections = []
    loops = 100
    for _ in range(loops):
        env._person_dict = {Person(random.randint(20, 60)):weight_list[i] for i in range(num_of_people)}
        for p, w in env._person_dict.items():
            if percent_infected < random.random():
                p._disease_state = DiseaseState.ASYMPTOMATICINFECTIOUS
            else:
                p._disease_state = DiseaseState.SUSCEPTIBLE
            p._change()
            env.sign_up_for_today(p, w)
        num_of_infections = len(env.propagate_infection(date(year = 2020,month = 12,day = 1)))
        infections.append(num_of_infections)
        total += num_of_infections
    avg = total/loops
    assert abs(avg - median(infections)) < 10

def test_single_person_infected_contact_half(params_path):
    """
    Tests that if we have only one person that is sick but he is in confinement,
    he won't infect a single person with contact_prob = 0.5
    """
    Params.load_from(params_path)
    DiseaseState.init_infectiousness_list()
    contact_prob = 0.5
    num_of_people = 400
    weight_list = [max(0.3, random.random()-0.1) for _ in range(num_of_people)]
    env = HomogeneousEnvironment(contact_prob, "test")
    infections = []
    
    
    env._person_dict = {Person(random.randint(20, 60)):weight_list[i] for i in range(num_of_people)}
    last_person = None
    for p, w in env._person_dict.items():
        p._disease_state = DiseaseState.SUSCEPTIBLE
        p._change()
        env.sign_up_for_today(p, w)
        last_person = p
    last_person._disease_state = DiseaseState.ASYMPTOMATICINFECTIOUS
    last_person._infectiousness_prob = 0 
    last_person._change()
    env.sign_up_for_today(p, w)


    num_of_infections = len(env.propagate_infection(date(year = 2020,month = 12,day = 1)))
    infections.append(num_of_infections)
    
    assert num_of_infections == 0 

def test_single_person_infected_contact1(params_path):
    """
    Tests that if we have only one person that is sick but he is in confinement,
    he won't infect a single person with contact_prob = 1
    """
    Params.load_from(params_path)
    DiseaseState.init_infectiousness_list()
    contact_prob = 1
    num_of_people = 400
    weight_list = [max(0.3, random.random()-0.1) for _ in range(num_of_people)]
    env = HomogeneousEnvironment(contact_prob, "test")
    infections = []
    
    
    env._person_dict = {Person(random.randint(20, 60)):weight_list[i] for i in range(num_of_people)}
    last_person = None
    for p, w in env._person_dict.items():
        p._disease_state = DiseaseState.SUSCEPTIBLE
        p._change()
        env.sign_up_for_today(p, w)
        last_person = p
    last_person._disease_state = DiseaseState.ASYMPTOMATICINFECTIOUS
    last_person._infectiousness_prob = 0 
    last_person._change()
    env.sign_up_for_today(p, w)


    num_of_infections = len(env.propagate_infection(date(year = 2020,month = 12,day = 1)))
    infections.append(num_of_infections)
    
    assert num_of_infections == 0 

def test_neiborhoods_diff_IDs():
    '''
    Test different neiborhood get different ids
    '''
    file_path = os.path.dirname(__file__) + "/../src/config.json"
    with open(file_path) as json_data_file:
        ConfigData = json.load(json_data_file)
        citiesDataPath = ConfigData['CitiesFilePath']
        paramsDataPath = ConfigData['ParamsFilePath']
    Params.load_from(os.path.join(os.path.dirname(__file__), paramsDataPath), override=True)
    pop = population_loader.PopulationLoader(citiesDataPath)
    my_world = pop.get_world(city_name = 'Atlit',scale = 1,is_smart= True)

    sample = []
    for env in my_world.all_environments:
        if env.name == "neighborhood_community":
            sample.append(env)
    assert not(sample[0].get_neighborhood_id() == sample[1].get_neighborhood_id())