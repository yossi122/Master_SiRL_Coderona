from datetime import timedelta
import json
import os
import pytest
import random
from test.conftest import helpers

from src.run_utils import SimpleJob, run, INITIAL_DATE
from src.seir import daysdelta
from src.seir.disease_state import DiseaseState
from src.simulation.event import DayEvent
from src.simulation.interventions import *
from src.simulation.initial_infection_params import SmartInitialInfectionParams
from src.simulation.params import Params
from src.simulation.simulation import Simulation
from src.logs import Statistics
from src.world import Person,World
from src.world.environments.household import Household



@pytest.fixture(params=[False, True])
def hi_exit(request, params):
    """
    This fixture creates params for HouseholdIsolationIntervention.
    Returns params that makes sure the household stays at home until the symptomatic people are immune.
    :param request: states the is_after value
    :param params: simulation params
    :return: (delay_on_exit, is_exit_after_recovery) params for intervention
    """
    is_after = request.param
    if is_after:
        return (1, True)
    else:
        disease_params = params['disease_parameters']
        delay_on_exit = 1
        for param in disease_params.values():
            if 'max_val' in param:
                delay_on_exit += param['max_val']
        return (delay_on_exit, False)


def test_household_isolation_intervention_simulation(hi_exit):
    """
    Tests that when asymptomatic and presymptomatic people are not infectious and household isolation occur with no delay,
    all the poeple that are infected are infected at home (of at the beginning), due to the household isolation
    :param hi_exit: params that make sure there are no delay or too short of isolation
    :return: None (since this is a test), but it is asserted that all infections occur within the household
    """
    scenario_name = "test_HI"
    params_to_change = {
        ("disease_parameters", "infectiousness_per_stage", "incubating_post_latent"): 0.0,
        ("disease_parameters", "infectiousness_per_stage", "asymptomatic"): 0.0
    }
    delay_on_exit, is_exit_after_recovery = hi_exit
    job = SimpleJob(scenario_name, city_name='kefar yona', interventions=[
        HouseholdIsolationIntervention(
            compliance=1,
            start_date=INITIAL_DATE, duration=timedelta(250),
            delay_on_enter=0, delay_on_exit=delay_on_exit,
            is_exit_after_recovery=is_exit_after_recovery
        )
    ], scale=1.0, params_to_change=params_to_change, infection_params=SmartInitialInfectionParams(100, 50))
    outdir = run([job], multi_processed=False, with_population_caching=False)
    results = Statistics.load(os.path.join(outdir, scenario_name, 'statistics.pkl'))
    summary = results.get_summary_data_for_age_group((0, 99))
    assert summary["Total infected in household"] + summary["Total infected in initial_group"] == summary[
        "Total infected"]


def test_school_closure_intervention_simulation():
    """
    Test that when school isolation is on, no one is infected at school.
    :return: True is there are 0 infections at school
    """
    scenario_name = "test_PC"
    job = SimpleJob(scenario_name, city_name='kefar yona', interventions=[
        SchoolClosureIntervention(
            start_date=INITIAL_DATE,
            duration=timedelta(250),
            compliance=1.0,
            proportion_of_envs=1.0,
            city_name='kefar yona',
            age_segment=(0, 9)
        )
    ], scale=1.0, infection_params=SmartInitialInfectionParams(100, 50))
    outdir = run([job], multi_processed=False, with_population_caching=False)
    results = Statistics.load(os.path.join(outdir, scenario_name, 'statistics.pkl'))
    summary = results.get_summary_data_for_age_group((0, 9))
    assert summary["Total infected in school"] == 0


def test_school_isolation_intervention_simulation():
    """
    Test that when school isolation is on, all the relevant people are infected only at shool
    """
    scenario_name = "test_PI"
    job = SimpleJob(scenario_name, city_name='kefar yona', interventions=[
        SchoolIsolationIntervention(
            start_date=INITIAL_DATE,
            duration=timedelta(250),
            compliance=1.0,
            proportion_of_envs=1.0,
            city_name='kefar yona',
            age_segment=(4, 12)
        )
    ], scale=1.0, infection_params=SmartInitialInfectionParams(100, 50))
    outdir = run([job], multi_processed=False, with_population_caching=False)
    results = Statistics.load(os.path.join(outdir, scenario_name, 'statistics.pkl'))
    summary = results.get_summary_data_for_age_group((4, 12))
    assert summary["Total infected in school"] + summary["Total infected in initial_group"] == summary["Total infected"]

def test_SymptomaticIsolationIntervention_Genarete_events(helpers):
    #pretesting
    helpers.clean_outputs()

    config_path = os.path.join(os.path.dirname(__file__),"..","src","config.json")
    with open(config_path) as json_data_file:
        ConfigData = json.load(json_data_file)
        paramsDataPath = ConfigData['ParamsFilePath']
    Params.load_from(os.path.join(os.path.dirname(__file__),"..","src", paramsDataPath), override=True)

    my_intervention = SymptomaticIsolationIntervention(compliance = 1, start_date = INITIAL_DATE,duration  = daysdelta(40))
    assert my_intervention is not None

    persons_arr = list(map(Person, [10,20,30]))
    assert len(persons_arr) == 3
    env_arr = []
    small_world = World(
        all_people = persons_arr,
        all_environments=env_arr,
        generating_city_name = "test",
        generating_scale = 1)
    
    my_simulation = Simulation(world = small_world, initial_date= INITIAL_DATE,interventions=[my_intervention])

    #test
    lst =  my_intervention.generate_events(small_world)
    #Assert results 
    assert lst is not None
    assert len(lst) == 3
    for i in range(1):
        assert isinstance(lst[i],DayEvent)
    for person in persons_arr:
        assert len(list(person.state_to_events.keys())) == (1+4)
    
    my_simulation.run_simulation(name="test",num_days = 60)
    
    
def test_ImmuneGeneralPopulationIntervention():
    #pretesting
    config_path = os.path.join(os.path.dirname(__file__),"..","src","config.json")
    with open(config_path) as json_data_file:
        ConfigData = json.load(json_data_file)
        paramsDataPath = ConfigData['ParamsFilePath']
    Params.load_from(os.path.join(os.path.dirname(__file__),"..","src", paramsDataPath), override=True)

    my_intervention = ImmuneGeneralPopulationIntervention(
        compliance = 1, 
        start_date = INITIAL_DATE,
        duration  = daysdelta(40),
        people_per_day = 1,
        min_age = 15) 
    assert my_intervention is not None

    persons_arr = list(map(Person, [10,20,30]))
    assert len(persons_arr) == 3
    env_arr = []
    small_world = World(
        all_people = persons_arr,
        all_environments=env_arr,
        generating_city_name = "test",
        generating_scale = 1)
    
    my_simulation = Simulation(world = small_world, initial_date= INITIAL_DATE,interventions=[my_intervention])
    
    #test
    lst =  my_intervention.generate_events(small_world)
    #Assert results 
    assert lst is not None
    assert len(lst) == 2
    for i in range(1):
        assert isinstance(lst[i],DayEvent)
    
    my_simulation.simulate_day()
    cnt_immune = sum([1 for p in persons_arr if p.get_disease_state()==DiseaseState.IMMUNE])
    assert cnt_immune == 1
    my_simulation.simulate_day()
    cnt_immune = sum([1 for p in persons_arr if p.get_disease_state()==DiseaseState.IMMUNE])
    assert cnt_immune == 2
    my_simulation.simulate_day()
    cnt_immune = sum([1 for p in persons_arr if p.get_disease_state()==DiseaseState.IMMUNE])
    assert cnt_immune == 2

def test_ImmuneByHouseholdIntervention():
    #pretesting
    config_path = os.path.join(os.path.dirname(__file__),"..","src","config.json")
    with open(config_path) as json_data_file:
        ConfigData = json.load(json_data_file)
        paramsDataPath = ConfigData['ParamsFilePath']
    Params.load_from(os.path.join(os.path.dirname(__file__),"..","src", paramsDataPath), override=True)

    my_intervention = ImmuneByHouseholdIntervention(
        compliance = 1, 
        start_date = INITIAL_DATE,
        duration  = daysdelta(40),
        houses_per_day = 1,
        min_age = 18) 
    assert my_intervention is not None

    config_path = os.path.join(os.path.dirname(__file__),"..","src","config.json")
    with open(config_path) as json_data_file:
        ConfigData = json.load(json_data_file)
        paramsDataPath = ConfigData['ParamsFilePath']
    Params.load_from(os.path.join(os.path.dirname(__file__),"..","src", paramsDataPath), override=True)

    #create diff enviroments
    KidsHouse = Household(city = None,contact_prob_between_each_two_people=1)
    AdultsHouse = Household(city = None,contact_prob_between_each_two_people=1)
    MixedHouse = Household(city = None,contact_prob_between_each_two_people=1)

    kidsAges = [random.randint(0,17) for i in range(4)]    
    adultsAges  = [random.randint(19,40) for i in range(3)]    
    KidsLst  = list(map(Person, kidsAges))
    adultsLst = list(map(Person, adultsAges))
    persons_arr = KidsLst + adultsLst

    #register people to diff env
    KidsHouse.sign_up_for_today(KidsLst[0],1)
    KidsHouse.sign_up_for_today(KidsLst[1],1)

    AdultsHouse.sign_up_for_today(adultsLst[0],1)
    AdultsHouse.sign_up_for_today(adultsLst[1],1)

    MixedHouse.sign_up_for_today(adultsLst[2],1)
    MixedHouse.sign_up_for_today(KidsLst[2],1)
    MixedHouse.sign_up_for_today(KidsLst[3],1)
    
    assert len(KidsHouse.get_people()) == 2
    assert len(AdultsHouse.get_people()) == 2
    assert len(MixedHouse.get_people()) == 3

    env_arr = [KidsHouse,AdultsHouse,MixedHouse]
    my_world = World(
        all_people = persons_arr,
        all_environments=env_arr,
        generating_city_name = "test",
        generating_scale = 1,)
    
    my_simulation = Simulation(world = my_world, initial_date= INITIAL_DATE,interventions=[my_intervention])
    
    #test
    lst =  my_intervention.generate_events(my_world)
    #Assert results 
    assert lst is not None
    assert len(lst) == 5
    for i in range(1):
        assert isinstance(lst[i],DayEvent)
    
    my_simulation.simulate_day()
    cnt_immune = sum([1 for p in persons_arr if p.get_disease_state()==DiseaseState.IMMUNE])
    assert cnt_immune <=3
    my_simulation.simulate_day()
    cnt_immune = sum([1 for p in persons_arr if p.get_disease_state()==DiseaseState.IMMUNE])
    assert cnt_immune == 5 
    my_simulation.simulate_day()
    