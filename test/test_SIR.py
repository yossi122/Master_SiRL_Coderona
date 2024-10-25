import os
import json
from collections import Counter

from src.run_utils import INITIAL_DATE
from src.seir import DiseaseState, disease_state,sample_seir_times,RealDataSeirTimesGeneration,SIRS
from src.simulation.params import Params
from src.world import Person
from src.world.environments import InitialGroup
from src.util.Enumerations import machine_type


#Test the amount of the  created Immune
def test_create_SIR_machine():
    #Pretest
    Params.clean()
    RealDataSeirTimesGeneration.clean()

    config_path = os.path.join(os.path.dirname(__file__),"..","src","config.json")
    Expected  = -1
    with open(config_path) as json_data_file:
        ConfigData = json.load(json_data_file)
        paramsDataPath = ConfigData['ParamsFilePath']
    Params.load_from(os.path.join(os.path.dirname(__file__),"..","src", paramsDataPath), override=True)

    #Assert able to create statte machines properly
    a= RealDataSeirTimesGeneration.make()
    assert isinstance(a,RealDataSeirTimesGeneration)

def test_create_SIRS_machine():
    #Pretest
    Params.clean()
    SIRS.clean()
    config_path = os.path.join(os.path.dirname(__file__),"..","src","config.json")
    with open(config_path) as json_data_file:
        ConfigData = json.load(json_data_file)
        paramsDataPath = ConfigData['ParamsFilePath']
    Params.load_from(os.path.join(os.path.dirname(__file__),"..","src", paramsDataPath), override=True)

    
    b= SIRS.make()
    assert isinstance(b,SIRS)

def test_sample_SIRS_seirs_times():
    #Pretest
    Params.clean()
    SIRS.clean()

    config_path = os.path.join(os.path.dirname(__file__),"..","src","config.json")
    with open(config_path) as json_data_file:
        ConfigData = json.load(json_data_file)
        paramsDataPath = ConfigData['ParamsFilePath']
    Params.load_from(os.path.join(os.path.dirname(__file__),"..","src", paramsDataPath), override=True)

    curr_machine_type = machine_type["SIRS"]
    p=Person(94)
    final_states = []
    for i in range(1000):
        table = sample_seir_times(curr_machine_type,p)
        assert table[0][0] == DiseaseState.LATENT
        final_states.append(table[-1][0])
    cnt_table  = Counter(final_states)
    assert DiseaseState.DECEASED in cnt_table.keys()
    assert DiseaseState.SUSCEPTIBLE in cnt_table.keys()
    assert len(cnt_table.keys())==2

#Tests for every junction in the state machine
#--------------------------------------------
def test_latent_to_incubating():
    #Pretest
    Params.clean()
    SIRS.clean()

    config_path = os.path.join(os.path.dirname(__file__),"tests_config_files","test_latent_to_incubating_config.json")
    with open(config_path) as json_data_file:
        ConfigData = json.load(json_data_file)
        paramsDataPath = ConfigData['ParamsFilePath']
        path = os.path.join(os.path.dirname(__file__),"tests_params_files", paramsDataPath)
    Params.load_from(path, override=True)

    p=Person(30)
    for _ in range(100):
        for cur_type in machine_type:
            tbl = sample_seir_times(cur_type,p)
            assert tbl[0][0] == DiseaseState.LATENT,"{}".format(cur_type.name)
            assert tbl[1][0] == DiseaseState.INCUBATINGPOSTLATENT,"{}".format(cur_type.name)
            assert tbl[2][0] == DiseaseState.SYMPTOMATICINFECTIOUS,"{}".format(cur_type.name)

def test_latent_no_incubating():
    #Pretest
    Params.clean()
    SIRS.clean()
    config_path = os.path.join(os.path.dirname(__file__),"tests_config_files","test_latent_no_incubating_config.json")
    with open(config_path) as json_data_file:
        ConfigData = json.load(json_data_file)
        paramsDataPath = ConfigData['ParamsFilePath']
    Params.load_from(os.path.join(os.path.dirname(__file__),"tests_params_files", paramsDataPath), override=True)

    p=Person(30)
    for _ in range(100):
        for cur_type in machine_type:
            tbl = sample_seir_times(cur_type,p)
            assert tbl[0][0] == DiseaseState.LATENT,"{}".format(cur_type.name)
            assert tbl[1][0] == DiseaseState.ASYMPTOMATICINFECTIOUS,"{}".format(cur_type.name)
            assert tbl[2][0] == DiseaseState.IMMUNE,"{}".format(cur_type.name)

def test_latent_incubating_immune():
    #Pretest
    Params.clean()
    SIRS.clean()

    config_path = os.path.join(os.path.dirname(__file__),"tests_config_files","test_latent_incubating_immune_config.json")
    with open(config_path) as json_data_file:
        ConfigData = json.load(json_data_file)
        paramsDataPath = ConfigData['ParamsFilePath']
    Params.load_from(os.path.join(os.path.dirname(__file__),"tests_params_files", paramsDataPath), override=True)

    p=Person(30)
    for _ in range(100):
        for cur_type in machine_type:
            tbl = sample_seir_times(cur_type,p)
            assert tbl[0][0] == DiseaseState.LATENT,"{}".format(cur_type.name)
            assert tbl[1][0] == DiseaseState.INCUBATINGPOSTLATENT,"{}".format(cur_type.name)
            assert tbl[2][0] == DiseaseState.SYMPTOMATICINFECTIOUS,"{}".format(cur_type.name)
            assert tbl[3][0] == DiseaseState.IMMUNE,"{}".format(cur_type.name)
    
def test_latent_incubating_critical():
    #Pretest
    Params.clean()
    SIRS.clean()

    config_path = os.path.join(os.path.dirname(__file__),"tests_config_files","test_latent_incubating_critical_config.json")
    with open(config_path) as json_data_file:
        ConfigData = json.load(json_data_file)
        paramsDataPath = ConfigData['ParamsFilePath']
    Params.load_from(os.path.join(os.path.dirname(__file__),"tests_params_files", paramsDataPath), override=True)

    p=Person(30)
    for _ in range(100):
        for cur_type in machine_type:
            tbl = sample_seir_times(cur_type,p)
            assert tbl[0][0] == DiseaseState.LATENT,"{}".format(cur_type.name)
            assert tbl[1][0] == DiseaseState.INCUBATINGPOSTLATENT,"{}".format(cur_type.name)
            assert tbl[2][0] == DiseaseState.SYMPTOMATICINFECTIOUS,"{}".format(cur_type.name)
            assert tbl[3][0] == DiseaseState.CRITICAL,"{}".format(cur_type.name)

def test_latent_incubating_critical_immune():
    #Pretest
    Params.clean()
    SIRS.clean()

    config_path = os.path.join(os.path.dirname(__file__),"tests_config_files","test_latent_incubating_critical_immune_config.json")
    with open(config_path) as json_data_file:
        ConfigData = json.load(json_data_file)
        paramsDataPath = ConfigData['ParamsFilePath']
    Params.load_from(os.path.join(os.path.dirname(__file__),"tests_params_files", paramsDataPath), override=True)

    p=Person(30)
    for _ in range(100):
        for cur_type in machine_type:
            tbl = sample_seir_times(cur_type,p)
            assert tbl[0][0] == DiseaseState.LATENT,"{}".format(cur_type.name)
            assert tbl[1][0] == DiseaseState.INCUBATINGPOSTLATENT,"{}".format(cur_type.name)
            assert tbl[2][0] == DiseaseState.SYMPTOMATICINFECTIOUS,"{}".format(cur_type.name)
            assert tbl[3][0] == DiseaseState.CRITICAL,"{}".format(cur_type.name)
            assert tbl[4][0] == DiseaseState.IMMUNE,"{}".format(cur_type.name)

def test_latent_incubating_critical_deceased():
    #Pretest
    Params.clean()
    SIRS.clean()

    config_path = os.path.join(os.path.dirname(__file__),"tests_config_files","test_latent_incubating_critical_deceased_config.json")
    with open(config_path) as json_data_file:
        ConfigData = json.load(json_data_file)
        paramsDataPath = ConfigData['ParamsFilePath']
    Params.load_from(os.path.join(os.path.dirname(__file__),"tests_params_files", paramsDataPath), override=True)

    p=Person(30)
    for _ in range(100):
        for cur_type in machine_type:
            tbl = sample_seir_times(cur_type,p)
            assert tbl[0][0] == DiseaseState.LATENT,"{}".format(cur_type.name)
            assert tbl[1][0] == DiseaseState.INCUBATINGPOSTLATENT,"{}".format(cur_type.name)
            assert tbl[2][0] == DiseaseState.SYMPTOMATICINFECTIOUS,"{}".format(cur_type.name)
            assert tbl[3][0] == DiseaseState.CRITICAL,"{}".format(cur_type.name)
            assert tbl[4][0] == DiseaseState.DECEASED,"{}".format(cur_type.name)

def test_SIRS_second_infection():
    """
    Test that in SIRS model in case that a person get sick twice,
     (and get recovered between them). 
     He will experience two different time schedule of the illness
    """
    #Pretest
    Params.clean()
    SIRS.clean()

    config_path = os.path.join(os.path.dirname(__file__),"tests_config_files","test_latent_incubating_critical_immune_config.json")
    with open(config_path) as json_data_file:
        ConfigData = json.load(json_data_file)
        paramsDataPath = ConfigData['ParamsFilePath']
    Params.load_from(os.path.join(os.path.dirname(__file__),"tests_params_files", paramsDataPath), override=True)
    Params.loader()["person"]["state_machine_type"] = "SIRS"
    p=Person(30)

    event_lst = p.infect_and_get_events(INITIAL_DATE,InitialGroup.initial_group())
    p.set_disease_state(DiseaseState.SUSCEPTIBLE)
    event_lst2 = p.infect_and_get_events(INITIAL_DATE,InitialGroup.initial_group())
    assert not (event_lst == event_lst2)
    