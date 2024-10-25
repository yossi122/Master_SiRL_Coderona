import json
import logging
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # Adding the src folder to PYTHONPATH

from src.scenarios import *
from src.seir import DiseaseState
from src.simulation.initial_infection_params import NaiveInitialInfectionParams, InitialImmuneType
from src.logs import make_age_and_state_datas_to_plot
from src.simulation.params import Params
from src.simulation.simulation import ORDER
from src.run_utils import RepeatJob, SimpleJob, run, make_base_infectiousness_to_r_job
import src.util.seed as seed

seed.set_random_seed()
log = logging.getLogger(__name__)


def generate_scenario_name(
        city_name,
        scenario,
        initial_num_infected,
        percentage_immuned,
        immune_per_day,
        immune_order,
        immune_complience_at_start,
        immune_source,
        min_age,
        compliance,
        ci_delay,
        hi_delay,
        symptomatic_probs_scale,
        minimum_infectiousness_age):
    return f"{city_name},{scenario},inf={initial_num_infected},immune={percentage_immuned}\n" + \
           f",imm_per_day={immune_per_day},imm_order={immune_order}\n" + \
           f",imm_comp={immune_complience_at_start}\n" + \
           f",comp={compliance}\n" + \
           f",imm_src={immune_source},min_age={min_age}\n" + \
           f",min_inf_age={minimum_infectiousness_age}"


def get_rescaled_symptomatic_probs(symptomatic_probs_scale):
    current_probs = Params.loader()['disease_parameters']['symptomatic_given_infected_per_age']
    return [min(1, symptomatic_probs_scale * t) for t in current_probs]


def get_datas_to_plot():
    graphs = {
        "infected": [
            DiseaseState.SYMPTOMATICINFECTIOUS,
            DiseaseState.INCUBATINGPOSTLATENT,
            DiseaseState.LATENT,
            DiseaseState.CRITICAL,
            DiseaseState.ASYMPTOMATICINFECTIOUS
        ],
        "critical": [
            DiseaseState.CRITICAL
        ],
        "susceptible": [
            DiseaseState.SUSCEPTIBLE
        ],
        "Immuned": [
            DiseaseState.IMMUNE
        ],
        "deceased": [
            DiseaseState.DECEASED
        ]
    }
    return {
        'amit_graph': make_age_and_state_datas_to_plot(age_groups=((0, 19), (20, 59), (60, 99)),
                                                       disease_state_groups=list(graphs.items())),
        'amit_graph_daily': make_age_and_state_datas_to_plot(age_groups=[(0, 99)],
                                                             disease_state_groups=list(graphs.items()),
                                                             is_integral=False),
        'amit_graph_integral': make_age_and_state_datas_to_plot(age_groups=[(0, 99)],
                                                                disease_state_groups=list(graphs.items()),
                                                                is_integral=True)
    }


def main():
    """
    This is the main function that runs the simulation.
    here we are able to add different intervention, params, and more configurations to the run.
    This is an example for using the code in the project, and running different simulations of the disease.
    """
    # sets the logging output to be at debug level, meaning more output than a regular run
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    scenarios = {
        # "scenario_1": scenario_1_interventions,
        # "scenario_21": scenario_21_interventions,
        # "scenario_22": scenario_22_interventions,
        # "scenario_23": scenario_23_interventions,
        # "scenario_24": scenario_24_interventions,
        # "scenario_25": scenario_25_interventions,
        # "scenario_26": scenario_26_interventions,
        # "scenario_232": scenario_232_interventions,
        # "scenario_262": scenario_262_interventions,
        # "scenario_272": scenario_272_interventions,
        # "scenario_282": scenario_282_interventions,
        # "scenario_36": scenario_36_interventions,
        # "scenario_39": scenario_39_interventions,
        # "scenario_365": scenario_365_interventions,
        # "scenario_395": scenario_395_interventions,
         "reality1" : scenario_reality1,
        #"hh_isolation": householdisolation_sd_interventions,
        # "check" : scenario_check,
        # "reality2" : scenario_reality2,
        # "reality3": scenario_reality3,
        # "reality4": scenario_reality4,
        # "no_interventions": no_interventions,
        # "not_relaxing_interventions": not_relaxing_interventions,
        # "grant_time1" : grant_time1,
        # "grant_time2" : grant_time2,
        # "paper_1" : paper_1,
        # "paper_2" : paper_2,
        # "noam_lockdown_scenario": noam_lockdown_scenario,
        # "no_interventions": no_interventions,
        # "paper_3" : paper_3,
        # "paper_4" : paper_4,
        # "paper_5": paper_5,
        # "paper_6": paper_6,
        # "paper_7": paper_7,
        # "paper_8": paper_8,
        # "paper_2_comp_9": paper_2_comp_9,
        #"vaccinations_scenario_general": vaccinations_scenario_general, # =>?
        #"vaccinations_scenario_households": vaccinations_scenario_households,
        # Empty_scenario": Empty_scenario,
        # "school_closure": children_school_closure_intervention,
        # "asymptomatic_detection": children_asymptomatic_detection_intervention,
        # "only_children_asymptomatic_detection":only_children_asymptomatic_detection
        # "noHH_children_specific_interventions": children_specific_noHH_interventions,
        # "HH_adult_specific_interventions": adult_specific_HH_interventions,
        # "noHH_adult_specific_interventions": adult_specific_noHH_interventions,
        # "HH_old_specific_interventions": old_specific_HH_interventions,
        # "noHH_old_specific_interventions": old_specific_noHH_interventions,
    }

    datas_to_plot = get_datas_to_plot()

    # choosing the city and the scale of the run:
    # the city name and scale determine the city and the size proportion to take (use 'all' for entire country)
    # city_name, scale = 'holon', 1
    # city_name, scale = 'all', 0.01 # This means loading 1% of the entire country
    # city_name, scale = 'all', 0.1 # This means loading the entire country
    print("Running all simulations...")
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path) as json_data_file:
        ConfigData = json.load(json_data_file)
        paramsDataPath = ConfigData['ParamsFilePath']

    Params.load_from(os.path.join(os.path.dirname(__file__), paramsDataPath), override=True)

    # we build a list of the jobs to run:
    # each job can be run as a different process, and the population generation will be done once
    # if caching option is on

    jobs = []
    #
    for target_immune_percentage, immune_compliance in [(0.8, 1)]:  # [(0.0,1),(0.5,1)]:
        for people_per_day in [60]:
            for immune_source, min_age in [(InitialImmuneType.BY_NEIGHBORHOOD, 18)]:  # the options are:GENERAL_POPULATION,HOUSEHOLDS
                for initial_num_infected in [700]:  # [25, 100, 250, 500]:
                    for city_name, scale in [("Holon", 1)]:  # [("Bene Beraq", 1), ("Holon", 1)]
                        for compliance in [0.7]:
                            for order in [ORDER.ASCENDING]:
                                for ci_delay in [4]:
                                    for hi_delay in [4]:
                                        # people aging less than minimum_infectioness_age will not infect others
                                        for minimum_infectiousness_age in [0]:  # [0, 18]
                                            for symptomatic_probs_scale in [1]:
                                                for scenario_name, intervention_scheme in scenarios.items():
                                                    params_to_change = {
                                                        ('disease_parameters',
                                                         'symptomatic_given_infected_per_age'): get_rescaled_symptomatic_probs(
                                                            symptomatic_probs_scale),
                                                        ('person',
                                                         'minimum_infectiousness_age'): minimum_infectiousness_age
                                                    }
                                                    full_scenario_name = generate_scenario_name(city_name,
                                                                                                scenario_name,
                                                                                                initial_num_infected,
                                                                                                target_immune_percentage,
                                                                                                people_per_day,
                                                                                                order,
                                                                                                immune_compliance,
                                                                                                immune_source,
                                                                                                min_age,
                                                                                                compliance,
                                                                                                ci_delay,
                                                                                                hi_delay,
                                                                                                symptomatic_probs_scale,
                                                                                                minimum_infectiousness_age)
                                                    #                                    full_scenario_name = "res"
                                                    jobs.append(RepeatJob(SimpleJob(full_scenario_name,
                                                                                    days=150,
                                                                                    city_name=city_name,
                                                                                    scale=scale,
                                                                                    infection_params=NaiveInitialInfectionParams(
                                                                                        num_to_infect=initial_num_infected,
                                                                                        per_to_Immune=target_immune_percentage, \
                                                                                        Immune_compliance=immune_compliance,
                                                                                        city_name_to_infect=city_name,
                                                                                        order=order,
                                                                                        immune_source=immune_source, \
                                                                                        min_age=min_age,
                                                                                        people_per_day=people_per_day),
                                                                                    # infection_params=SmartInitialInfectionParams(initial_num_infected, round(initial_num_infected/10)),
                                                                                    params_to_change=params_to_change,
                                                                                    interventions=intervention_scheme(
                                                                                        compliance, ci_delay, hi_delay),
                                                                                    datas_to_plot=datas_to_plot),
                                                                          num_repetitions=2)) # 800

                                        # add job to make r to base infectiousness graph:
                                        # jobs += [make_base_infectiousness_to_r_job(
                                        #             'r_graph_' + full_scenario_name, city_name, scale,
                                        #             np.arange(0.05, 0.15, 0.25),  # [0.03, 0.06, 0.1, 0.13, 0.16, 0.2],
                                        #             interventions=intervention_scheme(compliance, ci_delay, hi_delay),
                                        #             num_repetitions=3, initial_num_infected=initial_num_infected)]
    # this start the run of the jobs
    run(jobs, multi_processed=False, with_population_caching=False, verbosity=False)
    # multiprocessed = Flase

if __name__ == "__main__":
    main()
