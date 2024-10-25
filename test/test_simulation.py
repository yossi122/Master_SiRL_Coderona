import pytest
import os

from src.seir import DiseaseState
from src.simulation.initial_infection_params import NaiveInitialInfectionParams
from src.run_utils import SimpleJob, run
from src.logs import Statistics


def test_simple_simulation_single():
    """
    Runs the simple simulation of the disease on kefar yone (a city with about 20,000 people)
    """
    jobs = [SimpleJob("test_default", 'kefar yona', 1.0)]
    run(jobs, multi_processed=False)


def test_simple_simulation_multi():
    """
    Runs 3 simulations of the disease multi processed
    """
    jobs = [SimpleJob("test_default", 'kefar yona', 1.0), SimpleJob("test_default2", 'kefar yona', 1.0),
            SimpleJob("test_default3", 'kefar yona', 1.0)]
    run(jobs, multi_processed=True)


def test_simulation_haifa():
    """
    Runs Haifa simulation of the disease
    """
    job = SimpleJob("test_default_haifa", 'haifa', 1.0)
    run([job])

@pytest.mark.slow
def test_all_country_simulation_slow():
    """
    Runs a simulation on 10% of the entire country.
    This takes a while, so it's marked as "slow" (and should be run nightly rather than on every commit)
    """
    job = SimpleJob("test_default_all_0.1", 'all', 0.1)
    run([job])


def test_param_change_base_infectiousness(params):
    """
    Tests that when the base infectiousness is 0, the disease does not spread
    :param params: application params
    :return: True if all the sick people are from the initial set
    """
    params_to_change = {('person', 'base_infectiousness'): 0.0}
    job = SimpleJob("test_base_infectiousness_0", 'kefar yona', 1.0,
                    infection_params=NaiveInitialInfectionParams(10),
                    params_to_change=params_to_change)
    outdir = run([job], verbosity=True, multi_processed=False, with_population_caching=False)
    results = Statistics.load(os.path.join(outdir, 'test_base_infectiousness_0', 'statistics.pkl'))
    total_infected = results.sum_days_data(
        lambda person: person.disease_state != DiseaseState.SUSCEPTIBLE,
        True
    )[-1]
    assert total_infected == 10

def test_param_change_immune_parm(params):
    """
    Tests that when when we immune most od the population at the start of the disease
    they stay Immune and not deased while the simulation
    :param params: application params
    :return: True if the amount of people that are immune to the diease increased while the simulation
    """
    params_to_change = {('person', 'base_infectiousness'): 0.9}
    job = SimpleJob("test_base_infectiousness_0", 'kefar yona', 1.0,
                    infection_params=NaiveInitialInfectionParams(10,per_to_Immune= 0.9),
                    params_to_change=params_to_change)
    outdir = run([job], verbosity=True, multi_processed=False, with_population_caching=False)
    results = Statistics.load(os.path.join(outdir, 'test_base_infectiousness_0', 'statistics.pkl'))
    total_immuned = results.sum_days_data(
        lambda person: person.disease_state == DiseaseState.IMMUNE,
        True
    )[-1]
    assert 0.89 * 23061 <= total_immuned 

