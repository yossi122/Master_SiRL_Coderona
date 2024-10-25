import os
import json
import pytest
from shutil import rmtree

from src.world.city_data import get_city_list_from_dem_xls
from src.simulation.params import Params

class Helpers:
    @staticmethod
    def clean_outputs():
        path1 = os.path.join(os.path.dirname(__file__),'..','outputs')
        path2 = os.path.join(os.path.dirname(__file__),'..','src','outputs')
        if os.path.isdir(path1):
            rmtree(path1)
        if os.path.isdir(path2):
            rmtree(path2)
        return True

@pytest.fixture
def helpers():
    return Helpers
    
@pytest.fixture
def cities():
    file_path = os.path.dirname(__file__) + "/../src/config.json"
    with open(file_path) as json_data_file:
        ConfigData = json.load(json_data_file)
        citiesDataPath = ConfigData['CitiesFilePath']

    cities = get_city_list_from_dem_xls(citiesDataPath)
    return cities


@pytest.fixture
def params_path():
    file_path = os.path.dirname(__file__) + "/../src/config.json"
    with open(file_path) as json_data_file:
        ConfigData = json.load(json_data_file)
        ParamsDataPath = ConfigData['ParamsFilePath']
    return os.path.dirname(os.path.dirname(__file__))+"/Assets/"+ParamsDataPath


@pytest.fixture
def params(params_path):
    Params.load_from(params_path)
    return Params.loader()


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
