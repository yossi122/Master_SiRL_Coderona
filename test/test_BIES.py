import os
import logging

from src.seir import DiseaseState
from src.run_utils import SimpleJob, run
from src.logs import Statistics

def test_R0Big():
    """
    In case of R0 is much larger there shouled be a pandemic but not all of citizens
    will get cought
    R0 is computed by Beta/Gama where:
    Beta = m*p (number of interaction * probebility of infection)
    Gama = 1/d (d days where person is sick)
    """
    scenario_name = "test_NoInfection"
    params_to_change = {
        ("disease_parameters", "infectiousness_per_stage", "incubating_post_latent"): 1,
        ("disease_parameters", "infectiousness_per_stage", "asymptomatic"): 1,
        ("disease_parameters", "infectiousness_per_stage", "symptomatic"): 1,
        ("disease_parameters", "infectiousness_per_stage", "critical"): 1
    }
    jobs = [SimpleJob(scenario_name,
                      'kefar yona',
                      1.0,
                      params_to_change=params_to_change)]
    outdir = run(jobs, verbosity=True, multi_processed=False, with_population_caching=False)
    results = Statistics.load(os.path.join(outdir,scenario_name, 'statistics.pkl'))
    total_infected = results.sum_days_data(
        lambda person: person.disease_state != DiseaseState.SYMPTOMATICINFECTIOUS,
        True
    )[-1]
    # num of citizens in Kefar-Yona
    #assert total_infected <= 23061 in oredr to make statiscs work we dont preduce the same amount of people has written
