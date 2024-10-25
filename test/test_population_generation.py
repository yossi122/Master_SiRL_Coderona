import os
import json

from src.run_utils import INITIAL_DATE
from src.simulation.simulation import Simulation
from src.simulation.initial_infection_params import InitialImmuneType, NaiveInitialInfectionParams
from src.simulation.params import Params
from src.world.population_generation import population_loader
from src.world.population_generation import generate_city


def test_CityGeneration():
    file_path = os.path.dirname(__file__) + "/../src/config.json"
    with open(file_path) as json_data_file:
        ConfigData = json.load(json_data_file)
        citiesDataPath = ConfigData['CitiesFilePath']
        paramsDataPath = ConfigData['ParamsFilePath']
    Params.load_from(os.path.join(os.path.dirname(__file__), paramsDataPath), override=True)
    pop = population_loader.PopulationLoader(citiesDataPath)
    tmp_city = pop.get_city_by_name('Haifa')
    city = generate_city(tmp_city,
                         True,
                         internal_workplaces=True,
                         scaling=1.0,
                         verbosity=False,
                         to_world=True)

    humans = city.all_people()
    assert abs(283637 / len(humans) - 1) < 0.05
    assert len(city.get_all_city_communities()) == 1
    assert city.get_person_from_id(humans[0]._id) is not None

def test_CityEnvGeneration():
    file_path = os.path.dirname(__file__) + "/../src/config.json"
    with open(file_path) as json_data_file:
        ConfigData = json.load(json_data_file)
        citiesDataPath = ConfigData['CitiesFilePath']
        paramsDataPath = ConfigData['ParamsFilePath']
    Params.load_from(os.path.join(os.path.dirname(__file__), paramsDataPath), override=True)
    pop = population_loader.PopulationLoader(citiesDataPath)
    tmp_city = pop.get_city_by_name('Haifa')
    city = generate_city(tmp_city,
                         True,
                         internal_workplaces=True,
                         scaling=1.0,
                         verbosity=False,
                         to_world=True)
    assert len(city.all_environments) > 0

def test_gethouses():
    """
    Checking the method get households
    """
    file_path = os.path.dirname(__file__) + "/../src/config.json"
    with open(file_path) as json_data_file:
        ConfigData = json.load(json_data_file)
        citiesDataPath = ConfigData['CitiesFilePath']
        paramsDataPath = ConfigData['ParamsFilePath']
    Params.load_from(os.path.join(os.path.dirname(__file__), paramsDataPath), override=True)
    pop = population_loader.PopulationLoader(citiesDataPath)
    tmp_city = pop.get_city_by_name('Haifa')
    city = generate_city(tmp_city,
                         True,
                         internal_workplaces=True,
                         scaling=1.0,
                         verbosity=False,
                         to_world=True)
    cnt  = len([p for p in city.all_environments if p.name == 'household'])
    assert len(city.get_all_city_households()) - cnt == 0 

def test_NaiveInitialInfectionParams():
    c = NaiveInitialInfectionParams(10,0.5,'atlit',immune_source=InitialImmuneType.GENERAL_POPULATION)
    assert len(c.__str__()) > 0

def test_chickagoCityEnvGeneration():
    file_path = os.path.dirname(__file__) + "/../src/config.json"
    with open(file_path) as json_data_file:
        ConfigData = json.load(json_data_file)
        citiesDataPath = ConfigData['CitiesFilePath']
        paramsDataPath = ConfigData['ParamsFilePath']
    Params.load_from(os.path.join(os.path.dirname(__file__), paramsDataPath), override=True)
    pop = population_loader.PopulationLoader(citiesDataPath)
    tmp_city = pop.get_city_by_name('Chicago')
    city = generate_city(tmp_city,
                         True,
                         internal_workplaces=True,
                         scaling=1.0,
                         verbosity=False,
                         to_world=True)
    assert len(city.all_environments) > 0
    
def test_Atlit_population_generation():
    """
    Checking that we are able to genrate houses be according to the "CitiseData" excel file/
    """
    file_path = os.path.dirname(__file__) + "/../src/config.json"
    with open(file_path) as json_data_file:
        ConfigData = json.load(json_data_file)
        citiesDataPath = ConfigData['CitiesFilePath']
        paramsDataPath = ConfigData['ParamsFilePath']
    Params.load_from(os.path.join(os.path.dirname(__file__), paramsDataPath), override=True)
    pop = population_loader.PopulationLoader(citiesDataPath)
    my_world = pop.get_world(city_name = 'Atlit',scale = 1,is_smart= False)

    sim = Simulation(world = my_world, initial_date= INITIAL_DATE)
    agesArr = [i for i in range(5,80,5)]
    amountArr = [834,926,670,462,393,411,533,668,664,539,365,327,374,335,252]
    assert len(agesArr) == len(amountArr)
    for i in range(len(agesArr)):
        assert (sum([1 for p in my_world.all_people() if (agesArr[i]-5 < p.get_age()<agesArr[i])]) - amountArr[i]) < 10 
    for i in range(len(agesArr)):
        assert (sum([1 for p in my_world.all_people() if p.get_age() > 75]) - 294) < 10
    
    house_size_arr = [0,0,0,0,0,0,0]
    for h in my_world.get_all_city_households():
        #Two protections one against large numbers and -1 because we start from zero
        house_size = min(7,len(h.get_people()))-1
        house_size_arr[house_size] = house_size_arr[house_size]+1
    total = sum(house_size_arr)
    for i in range(len(house_size_arr)):
        house_size_arr[i] = house_size_arr[i]/total * 100
    expected = [7.4,22.5,20.1,28.7,13.7,6.2,1.4]
    for i in range(len(house_size_arr)):
        assert (abs(house_size_arr[i]-expected[i])<1),str(i)

    #Count houses with personos above 65 and / or less then 18
    a65 = 0
    l17 = 0
    for h in my_world.get_all_city_households():
        for p in h.get_people():
            if p.get_age() < 18:
                l17 = l17 + 1
                break

        for p in h.get_people():
            if p.get_age() > 64:
                a65 = a65 + 1
                break
    #Atlit (head count now)/(head count in 2008)
    growth_factor = 8047/5000
    households_num = len(my_world.get_all_city_households())
    print("a65:" + str(a65))
    print("l17:" + str(l17))
    print("total:" + str(households_num))
    assert abs((households_num / growth_factor - 1350)) < 1350*0.01
    #Tests that fails 
    assert abs((a65 / (households_num*growth_factor))*100 - 21.3) < 1
    assert abs((l17 / (households_num*growth_factor))*100 - 50.7) < 1

    kids_expected = [44.7,32.3,20.3,2.5,0.2]
    kids_reality = [0,0,0,0,0,0]
    
    for house in my_world.get_all_city_households():
        kidCnt =0 
        for person in house.get_people():
            if person.get_age() < 18:
                kidCnt+=1
        kidCnt = min(5,kidCnt)
        kids_reality[kidCnt] += 1
    for i in range(len(kids_expected)):
        kids_reality[i] = kids_reality[i] / len(my_world.get_all_city_households()) * 100
    print(kids_reality)
    for i in range(len(kids_reality)):
        assert abs(kids_reality[i+1] - kids_expected[i]) < 0.1
        