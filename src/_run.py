import json
import logging
import os
import sys
import numpy as np
# Add the 'src' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  
# Import project modules
from src.scenarios import *
from src.seir import DiseaseState
from src.simulation.initial_infection_params import NaiveInitialInfectionParams, InitialImmuneType
from src.logs import make_age_and_state_datas_to_plot
from src.simulation.params import Params
from src.run_utils import RepeatJob, SimpleJob, run, make_base_infectiousness_to_r_job
import src.util.seed as seed
from src.simulation.simulation import ORDER
from src.simulation.initial_infection_params import NaiveInitialInfectionParams, InitialImmuneType
# Set global random seed
seed.set_random_seed()


# wandb logging and DQN reinforcement learning
import wandb
#from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
from src.simulation._DQN import wandb_routine, get_log_history

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
    #return f"""{city_name},{scenario},inf={initial_num_infected},immune={percentage_immuned}\n, imm_per_day={immune_per_day},imm_order={immune_order}\n,imm_comp={immune_complience_at_start}\n, comp={compliance},ci_delay={ci_delay},hi_delay={hi_delay}\n, imm_src={immune_source}, min_age={min_age}\n, min_inf_age={minimum_infectiousness_age}"""
    # Generate a simplified name for each scenario            
    return f"{city_name},{scenario},imm_per_day={immune_per_day},comp={compliance},ci_delay={ci_delay},hi_delay={hi_delay}"

def get_rescaled_symptomatic_probs(symptomatic_probs_scale):
    # Rescale the probabilities of showing symptoms given infection    
    current_probs = Params.loader()['disease_parameters']['symptomatic_given_infected_per_age']
    return [min(1, symptomatic_probs_scale * t) for t in current_probs]


def get_datas_to_plot():
    # Specify which disease states to track for plotting    
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
        ],
        "vaccinated": [DiseaseState.VACCINATED]

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


def setup_logger(log_dir=os.getcwd(), log_file='app.log'):
    # Create a directory if it doesn't exist
    # os.makedirs(log_dir, exist_ok=True)

    # Configure logging
    if os.path.exists(os.path.join(log_dir, log_file)):
        os.remove(os.path.join(log_dir, log_file))
    log_path = os.path.join(log_dir, log_file)

    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Optionally, add a console handler to see logs on the console as well
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Get the root logger

    logger = logging.getLogger()
    logger.addHandler(console_handler)


def custom_exception_handler(exc_type, exc_value, exc_traceback):
    # Custom global exception handler for uncaught exceptions
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


# Set the custom exception handler
sys.excepthook = custom_exception_handler


def config_dict_to_tuple(dicts_list):
    # Convert list of dicts to list of 2-tuples based on first two keys    
    keys = list(dicts_list[0].keys())
    return [(e[keys[0]], e[keys[1]]) for e in dicts_list]

def load_config_params(ConfigData):
    # Extract experiment parameters and scenarios from configuration dictionary    
    experiment_params = {
        "experiment_type": ConfigData["experiment_type"],
        "experiment_mode": ConfigData["experiment_mode"],
        "linked_immune_compliance": ConfigData["linked_immune_compliance"],
        "people_per_day": ConfigData["people_per_day"],
        "linked_immune_age": ConfigData["linked_immune_age"],
        "initial_num_infected": ConfigData["initial_num_infected"],
        "linked_city_scale": ConfigData["linked_city_scale"],
        "compliance": ConfigData["compliance"],
        "order": ConfigData["order"],
        "ci_delay": ConfigData["ci_delay"],
        "hi_delay": ConfigData["hi_delay"],
        "minimum_infectiousness_age": ConfigData["minimum_infectiousness_age"],
        "symptomatic_probs_scale": ConfigData["symptomatic_probs_scale"],
        "wandb": ConfigData["wandb"]
    }
    # Map scenario names to functions    
    scenarios = {
        name: globals()[func_name] for name, func_name in ConfigData['scenarios'].items()
    }
    # Set prefix and scenario handling based on experiment type
    if experiment_params['experiment_type'].lower() == 'dqn':
        t = scenarios['Empty_scenario']
        scenarios.clear()
        scenarios['Empty_scenario'] = t
        prefix = 'reinforcement'
    else:
        scenarios.pop('Empty_scenario', None)
        prefix = 'basic'
    # Limit test mode to first city and age setting
    if experiment_params["experiment_mode"].lower() == 'test': # train - full list, test - first city
        experiment_params["linked_immune_age"] = [experiment_params["linked_immune_age"][0]]
        experiment_params["linked_city_scale"] = [experiment_params["linked_city_scale"][0]]
    # Convert order strings to enum values
    if experiment_params["order"] == ['ASCENDING']:
        experiment_params["order"] = [ORDER.ASCENDING]
    elif experiment_params["order"] == ['DESCENDING']:
        experiment_params["order"] = [ORDER.DESCENDING]
    elif experiment_params["order"] == ['NONE']:
        experiment_params["order"] = [ORDER.NONE]
    # Replace immune source strings with enum values
    for item in experiment_params["linked_immune_age"]:
        source = item["immune_source"]
        if source in InitialImmuneType.__dict__:
            item["immune_source"] = InitialImmuneType.__dict__[source]

    return experiment_params, scenarios, prefix

def main():
    # Initialize logger and plotting data structures
    setup_logger()
    datas_to_plot = get_datas_to_plot()

    print("Running all simulations...")
    # Load configuration file    
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path) as json_data_file:
        ConfigData = json.load(json_data_file)
        paramsDataPath = ConfigData['ParamsFilePath']

    # Extract simulation parameters and scenarios
    experiment_params, scenarios, prefix = load_config_params(ConfigData)
    # Initialize Weights & Biases if enabled
    if experiment_params["wandb"]["usage_flag"]:
        flag = experiment_params["wandb"]["usage_flag"]
        proj = experiment_params["wandb"]["experiment_name"]
        team = experiment_params["wandb"]["team_name"]
        wandb_routine(flag=flag,
                      project_name=proj,
                      team_name=team)
        # wandb.agent(sweep_id, function=wandb_routine, count=10)
    # Load disease parameters
    Params.load_from(os.path.join(os.path.dirname(__file__), paramsDataPath), override=True)
    # Prepare simulation jobs    
    jobs = []
    zoo = experiment_params["linked_city_scale"]
    foo = config_dict_to_tuple(experiment_params["linked_city_scale"])
    for target_immune_percentage, immune_compliance in config_dict_to_tuple(
            experiment_params["linked_immune_compliance"]):
        for people_per_day in experiment_params["people_per_day"]:
            for immune_source, min_age in config_dict_to_tuple(experiment_params["linked_immune_age"]):
                for initial_num_infected in experiment_params["initial_num_infected"]:  # [25, 100, 250, 500]:
                    for city_name, scale in config_dict_to_tuple(experiment_params["linked_city_scale"]):
                        for compliance in experiment_params["compliance"]:
                            for order in experiment_params["order"]:
                                for ci_delay in experiment_params["ci_delay"]:
                                    for hi_delay in experiment_params["hi_delay"]:
                                        # people aging less than minimum_infectioness_age will not infect others
                                        for minimum_infectiousness_age in experiment_params[
                                            "minimum_infectiousness_age"]:  # [0, 18]
                                            for symptomatic_probs_scale in experiment_params["symptomatic_probs_scale"]:
                                                for scenario_name, intervention_scheme in scenarios.items():
                                                    # Set dynamic parameters    
                                                    params_to_change = {
                                                        ('disease_parameters',
                                                         'symptomatic_given_infected_per_age'): get_rescaled_symptomatic_probs(
                                                            symptomatic_probs_scale),
                                                        ('person',
                                                         'minimum_infectiousness_age'): minimum_infectiousness_age
                                                    }
                                                    # Create a scenario name
                                                    full_scenario_name = generate_scenario_name(f'{prefix}_{city_name}',
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
                                                    # Define job
                                                    simple = SimpleJob(full_scenario_name,
                                                                       days=150,
                                                                       city_name=city_name,
                                                                       scale=scale,
                                                                       infection_params=NaiveInitialInfectionParams(
                                                                           num_to_infect=initial_num_infected,
                                                                           per_to_Immune=target_immune_percentage,
                                                                           Immune_compliance=immune_compliance,
                                                                           city_name_to_infect=city_name,
                                                                           order=order,
                                                                           immune_source=immune_source,
                                                                           min_age=min_age,
                                                                           people_per_day=people_per_day),
                                                                       # infection_params=SmartInitialInfectionParams(initial_num_infected, round(initial_num_infected/10)),
                                                                       params_to_change=params_to_change,
                                                                       interventions=intervention_scheme(
                                                                           compliance, ci_delay, hi_delay),
                                                                       datas_to_plot=datas_to_plot)
    # Remove old output if exists                                                jobs.append(simple)
    if os.path.exists('Holon.csv'):
        os.remove('Holon.csv')
    # Log experiment history and run jobs
    get_log_history()
    run(jobs, multi_processed=False, with_population_caching=False, verbosity=False)


if __name__ == "__main__":
    main()
