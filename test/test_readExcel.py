import pytest
import json
import os

from src.simulation.params import Params
from src.world.population_generation import population_loader
from src.world.city_data import get_city_list_from_dem_xls


def test_Init_haifa():
    citiesDataPath = ""
    config_path = os.path.join(os.path.dirname(__file__),"..","src","config.json")
    with open(config_path) as json_data_file:
        ConfigData = json.load(json_data_file)
        citiesDataPath = ConfigData['CitiesFilePath']
        paramsDataPath = ConfigData['ParamsFilePath']

    Params.load_from(os.path.join(os.path.dirname(__file__), paramsDataPath), override=True)
       
    pop = population_loader.PopulationLoader(
            citiesDataPath,
            added_description="",
            with_caching=False,
            verbosity=False
        )
    world = pop.get_world(city_name='Haifa', scale=1,is_smart= False)
    assert world is not None


def test_Init_haifaParms():
    citiesDataPath = ""
    config_path = os.path.join(os.path.dirname(__file__),"..","src","config.json")
    with open(config_path) as json_data_file:
        ConfigData = json.load(json_data_file)
        citiesDataPath = ConfigData['CitiesFilePath']
        paramsDataPath = ConfigData['ParamsFilePath']

    Params.load_from(os.path.join(os.path.dirname(__file__), paramsDataPath), override=True)
       
    pop = population_loader.PopulationLoader(
            citiesDataPath,
            added_description="",
            with_caching=False,
            verbosity=False
        )
    City1 = pop.get_city_by_name('Haifa')
    assert City1.region == 3
    assert City1.nafa == 31


def test_Init_SmallTown():
    file_path = os.path.dirname(__file__)+"/../src/config.json"
    with open(file_path) as json_data_file:
        ConfigData = json.load(json_data_file)
        citiesDataPath = ConfigData['CitiesFilePath']
    with pytest.raises(Exception):
        pop = population_loader.PopulationLoader(citiesDataPath)
        City1 = pop.get_city_by_name('Roah Midbar')


def test_Init_TownNotExist():
    file_path = os.path.dirname(__file__)+"/../src/config.json"
    with open(file_path) as json_data_file:
        ConfigData = json.load(json_data_file)
        citiesDataPath = ConfigData['CitiesFilePath']
    with pytest.raises(Exception):
        pop = population_loader.PopulationLoader(citiesDataPath)
        City1 = pop.get_city_by_name('lala')


def test_GetCities():
    #There are only 198 cities that we know all the needed data
    file_path = os.path.dirname(__file__) + "/../src/config.json"
    with open(file_path) as json_data_file:
        ConfigData = json.load(json_data_file)
        citiesDataPath = ConfigData['CitiesFilePath']
    lst  = get_city_list_from_dem_xls(citiesDataPath)
    assert len(lst) == 199

