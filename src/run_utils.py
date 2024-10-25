import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from datetime import date
import copy
import logging
import datetime, re

import shutil
import numpy as np
import wandb
from tqdm import tqdm
import gc

from src.util import seed
from src.logs import *
from src.simulation.params import Params
from src.world.population_generation import PopulationLoader
from src.simulation.simulation import Simulation
from src.simulation.initial_infection_params import SmartInitialInfectionParams, NaiveInitialInfectionParams, InitialImmuneType
from src.seir import DiseaseState
from src.simulation.simulation import ORDER

# ===== NEW BLOCK ===
from src.simulation._DQN import AgentLSTM, EpidemicStats, Runner, wandb_routine, get_config_param
from collections import Counter
#from tensorflow.keras.models import load_model

INITIAL_DATE = date(year=2020, month=2, day=27)
logger = logging.getLogger(__name__)
log = logging.getLogger(__name__)


# ===== NEW BLOCK ===

class Task:
    """""
    An object which represents a task of running a single simulation once
    Each task, has to be single processed
    """
    __slots__ = ('func', 'params', 'is_done')

    def __init__(self, func, params):
        """
        Init a task object
        :param func: the main function of the task (currently only create_and_run_simulation)
        :param params: the parameters to pass to the func
        """
        self.func = func
        self.params = params
        self.is_done = False


class RunningJob(object):
    """
    A job creates tasks in order run the simulation(s) and then process the outputs.
    One job handles the run of all it's tasks, and can be run as a single process or multi processed
    """

    def __init__(
            self, scenario_name, city_name, scale,
            days=30, initial_date=INITIAL_DATE,
            params_to_change=None, interventions=None
    ):
        """
        initialize the job. The params are given for all the task that would be generated in this job.
        :param scenario_name: str name to use for the directories and filenames of the outputs
        :param city_name: str city name, can be 'all' for entire country simulation
        :param scale: float between 0-1, represents the size scale of the city
        :param days: length of the simulation in days
        :param initial_date: datetime.date start date of the run
        :param params_to_change: dict of the temporary changes to make to Param object
        :param interventions: list of Intervention objects to apply during the simulation
        """
        self.scenario_name = scenario_name
        self.city_name = city_name.lower()
        self.scale = scale

        self.days = days
        self.initial_date = initial_date

        self.params_to_change = params_to_change
        if self.params_to_change is None:
            self.params_to_change = {}

        self.interventions = interventions
        if self.interventions is None:
            self.interventions = []

    def update_params(self, params_change):
        """
        Add more params to change to the param changes dict.
        The updated will affect the run, but will not be saved on disk
        :param params_change: dict of param changes
        """
        for param, val in params_change.items():
            self.params_to_change[param] = val

    def generate_tasks(self, outdir, stop_early=None):
        """
        Has to be implemented by the subclass
        :param outdir: path to the output directory, for all the output files of the tasks
        :param stop_early: relevant for R computation, see Simulation doc
        :return: list of Task object to run
        """
        raise NotImplementedError()

    def finalize(self, outdir):
        """
        This func handles the outputs that need to be written after all the tasks are finished.
        For example, compute mean, std and confidence of all the runs.
        :param outdir: path to output directory
        """
        raise NotImplementedError()

    def get_all_params_changes(self):
        """
        gets all the param changes of this run
        :return: list of param changes
        """
        return [self.params_to_change]


class SimpleJob(RunningJob):
    """
    A subclass that implements a job of a single task, that runs the simulation once.
    """

    def __init__(self, scenario_name, city_name, scale, infection_params=SmartInitialInfectionParams(100, 50),
                 days=250, city_name_to_infect=None, initial_date=INITIAL_DATE,
                 params_to_change=None, datas_to_plot=None, interventions=None):
        """
        Initialize a simple job, that runs one simulation task
        :param scenario_name: str name to use for the directories and filenames of the outputs
        :param city_name: str city name, can be 'all' for entire country simulation
        :param scale: float between 0-1, represents the size scale of the city
        :param days: length of the simulation in days
        :param interventions: list of Intervention objects to apply during the simulation
        :param infection_params: states how the initial infection will happen, see InitialInfectionParams doc
        :param city_name_to_infect: str name of the city that will be infected first, before the simulation
        :param initial_date: datetime.date start date of the run
        :param params_to_change: dict of the temporary changes to make to Param object
        :param datas_to_plot: states what data from the simulation will be counted and saved to output plots,
        see DataToPlot doc. Has a default behavior if the param is omitted.
        """

        super(SimpleJob, self).__init__(
            scenario_name, city_name, scale,
            days, initial_date, params_to_change,
            interventions
        )
        self.infection_params = infection_params
        self.city_name_to_infect = city_name_to_infect
        self.datas_to_plot = datas_to_plot
        if self.datas_to_plot is None:
            self.datas_to_plot = {
                "persons": make_age_and_state_datas_to_plot(),
                "infector_age": make_infections_age_datas_to_plot(),
                "infector_state": make_infections_infector_state_datas_to_plot(),
                "infection_environment": make_infections_environment_datas_to_plot()
            }
        self.is_exp_name_updated = False  #!@

    def generate_tasks(self, outdir, stop_early=None):
        """
        creates the output directory for the run, and returns the single task to run
        :param outdir: the path to the output directory
        :param stop_early: only relevant to R computation, see Simulation doc
        :return: list of one Task
        """
        outdir = os.path.join(outdir, self.scenario_name)
        assert not os.path.exists(outdir), "Directory '%s' already exists!" % outdir
        os.makedirs(outdir)
        return [Task(self.create_and_run_simulation, (outdir, stop_early))]

    def finalize(self, outdir):
        pass

    # ===== NEW BLOCK ===
    def create_and_run_simulation(self, outdir, stop_early, with_population_caching=True, verbosity=False):
        '''~>
        seed.set_random_seed()
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        with open(config_path) as json_data_file:
            ConfigData = json.load(json_data_file)
            #Extensionslst = ConfigData['ExtensionsNamelst']
            train_epochs = ConfigData["train_epochs"]
            stop_condition = ConfigData["stop_condition"]

        if stop_condition["rt_flag"]:
            stop_condition = stop_condition["rt"]
        else:
            stop_condition = {key: value for key, value in stop_condition.items()
                              if key not in ["rt_flag", "rt"]}
        '''

        seed.set_random_seed()
        train_epochs = get_config_param('train_epochs')
        stop_condition = get_config_param("stop_condition")
        if stop_condition["rt_flag"]:
            stop_condition = stop_condition["rt"]
        else:
            stop_condition = {key: value for key, value in stop_condition.items()
                              if key not in ["rt_flag", "rt"]}

        checkpoint_date = get_config_param("dqn_agent")["checkpoint_date"]
        checkpoint_date = checkpoint_date.replace('-', '_') if checkpoint_date else None
        log.info(checkpoint_date)

        model_path = self.check_and_find_folder(checkpoint_date, '../outputs', self.city_name)
        log.info(f'model_path {model_path}')
       
        if model_path:
            for idx, p in enumerate(model_path):
                if self.city_name.lower() in p.lower():
                    dirname, _ = os.path.split(p)
                    model_path = model_path[idx]
            json_path = f'{dirname}/params.json'

            runner = Runner(outdir, condition_strategy=stop_condition, episode=json_path)

            with open(json_path) as json_data_file:
                ConfigData = json.load(json_data_file)

            runner.history_actions_reward = {i: [] for i in range(11)}
            runner.history_actions_counter = ConfigData['action_select_counter']
            f = json_path
        else:
            runner = Runner(outdir, condition_strategy=stop_condition)
            runner.history_actions_reward = {i: [] for i in range(11)}
            runner.history_actions_counter = {i: 0 for i in range(11)}
            f = None

        runner.get_compliance(outdir)
        #runner.history_memory_access = {'EPS': False, 'MEMORY': False}

        economic = EpidemicStats(is_ratio=get_config_param("absolute_stats_flag"))
        model_usage_mode = get_config_param('dqn_agent')['mode'].lower()
        
        if runner:
            if model_usage_mode == "train":
                log.info(f'path, {model_path}')
                agent = AgentLSTM(runner, economic, self, model= model_path, epsilon=f)
                #agent.set_model(economic)
                log.info(
                    f">> Model. Input shape {agent.model.input_shape}[, d, s]\t Output shape {agent.model.output_shape} [, d, a]")

                e = 1 if runner.simulation_type == 'basic' else train_epochs  # for basic simulation episodes = 1
                e_start = runner.episode if model_path else 0

                x = get_config_param("wandb")
                self.is_exp_name_updated = wandb_routine(name=(self.scenario_name, self.is_exp_name_updated),
                                                         team_name=x["team_name"],
                                                         project_name=x["experiment_name"])  # here renaming

                #self.infection_params = NaiveInitialInfectionParams
               #~> runner.people_per_day_to_vac = self.infection_params.people_per_day

                if x["usage_flag"]:
                    log.info(f"== STARTING JOB == {wandb.run.name}")


                for episode in range(e_start, e):
                    runner.init_infect = self.infection_params.num_to_infect
                    log.info(f"== episode {episode}  ==\n")
                    log.info(f"~ COMPLIANCE {runner.compliance}")
                    log.info(f"~ Infected number {self.infection_params.num_to_infect}")
                    path = os.path.dirname(__file__)
                    world, Extensionslst = runner.population_init(self, path,
                                                                  with_population_caching=with_population_caching,
                                                                  verbosity=verbosity)
                    print(f'~~~ EPOCH INITIALIZATION {Counter([person.get_disease_state() for person in world._people_dict.values()])}')
                    extenstion_name = get_config_param("ExtensionsNamelst")
                    ext_param = {extenstion_name[0]: self.infection_params} if extenstion_name != [] else {}

                    sim = Simulation(world, self.initial_date, self.interventions,
                                     verbosity=verbosity, outdir=outdir, stop_early=stop_early,
                                     extension_params={"ImmuneByAgeExtension": self.infection_params})
                    self.interventions = []
                    runner.cur_action = -1 #!@ FIRST N days just epidemic raises
                    sim.update_events(self.interventions, ext_param)
                    self.infection_params.infect_simulation(sim, outdir)
                    # TODO: is this extra statistics calc so necessary
    # economic.get_population_statistic(sim, [], self.infection_params.num_to_infect,
                    #                                  aged_flag=get_config_param("age_stats"))

                    datas_to_plot, extensions = sim.get_extensions(self.days,
                                                                   self.datas_to_plot,
                                                                   self.scenario_name,
                                                                   Extensionslst)
                    economic.cum_vaccinated = 0  # init with 0 for each new epoch
                    info = (runner, agent, economic, sim, extensions, self, outdir, ext_param)

                    if len(Extensionslst) > 0:
                        extensions, self = sim.run_simulation(self.days, self.scenario_name,
                                                              datas_to_plot=datas_to_plot,
                                                              extensionsList=Extensionslst,
                                                              info=info)
                    else:
                        extensions, self = sim.run_simulation(self.days, self.scenario_name,
                                                              datas_to_plot=datas_to_plot,
                                                              extensionsList=None, info=info)

                    runner.episode += 1
                    comp = get_config_param("compliance")[0]
                    comp_spread, comp_step = get_config_param("compliance_spread")
                    if comp_spread == 0 and comp_step == 0:
                        runner.compliance = comp
                    else:
                        runner.compliance = comp + random.choice(np.arange(-comp_spread, comp_spread + comp_step, comp_step))

                    init_infected = get_config_param("initial_num_infected")[0]
                    inf_spread, inf_step = get_config_param("num_infected_spread")
                    if inf_spread == 0 and inf_step == 0:
                        to_infect = init_infected
                    else:
                        to_infect = (init_infected + random.choice(np.arange(-inf_spread, inf_step + inf_spread, inf_step)))
                    '''
                    self.infection_params.num_to_infect = (init_infected +
                                                           random.choice(
                                                               np.arange(-inf_spread, inf_step + inf_spread, inf_step)))
                    '''
                    print("new infect number:", to_infect)
                    self.infection_params = NaiveInitialInfectionParams(num_to_infect=to_infect,
                                                                        per_to_Immune=self.infection_params.per_to_Immune,
                                                                        Immune_compliance=runner.compliance,
                                                                        city_name_to_infect= self.infection_params.city_name_to_infect,
                                                                        order=self.infection_params.order,
                                                                        immune_source=self.infection_params.immune_source,
                                                                        min_age=self.infection_params.min_age,
                                                                        people_per_day=self.infection_params.people_per_day)

                    print('!' * 70)
                    print(self.infection_params)
                    print('!' * 70)
                sim.save_stats(self.datas_to_plot, self.scenario_name, info=(runner, agent))

            elif model_usage_mode == 'test':
                from pathlib import Path

                seed.set_random_seed()
                config_path = os.path.join(os.path.dirname(__file__), "config.json")
                with open(config_path) as json_data_file:
                    ConfigData = json.load(json_data_file)
                    # Extensionslst = ConfigData['ExtensionsNamelst']
                    train_epochs = ConfigData["train_epochs"]
                    stop_condition = ConfigData["stop_condition"]

                if stop_condition["rt_flag"] == 1:
                    stop_condition = stop_condition["rt"]
                else:
                    stop_condition = {key: value for key, value in stop_condition.items()
                                      if key not in ["rt_flag", "rt"]}

                if get_config_param('dqn_agent')['test_params']['use_latest']:
                    script_location = Path(__file__).resolve().parent
                    folder = script_location.parent / 'outputs'

                    if not folder.exists() or not folder.is_dir():
                        raise FileNotFoundError(f"The directory {folder} does not exist or is not a directory.")

                    # Get a list of all directories in folder A
                    folders = [f for f in folder.iterdir() if f.is_dir()]

                    if len(folders) < 2:
                        raise FileNotFoundError(
                            "Not enough directories found inside the outputs folder to select the second most recent")

                    # Filter directories to only those containing .h5 files
                    folders_with_h5 = [f for f in folders if list(f.glob('**/*.h5'))]

                    if not folders_with_h5:
                        raise FileNotFoundError("No directories containing .h5 files found inside folder outputs")

                    # Sort directories by creation time
                    latest_root = max(folders_with_h5, key=lambda f: f.stat().st_ctime)
                    h5_files = list(latest_root.glob('**/*.h5'))
                    outdir = str(latest_root)

                agents = []
                for idx, m in enumerate(h5_files):
                    model = load_model(m)
                    agent = AgentLSTM(runner, economic, self, model=model)
                    agents.append(agent)

                    #agents.append(AgentLSTM(runner, economic, self))
                    log.info(f"TESTING DQN MODEL :: {os.path.basename(m)}")
                    #agents[idx].load_model(str(m))

                    path = os.path.dirname(__file__)
                    world, Extensionslst = runner.population_init(self, path,
                                                                  with_population_caching=with_population_caching,
                                                                  verbosity=verbosity)
                    sim = Simulation(world, self.initial_date, self.interventions,
                                     verbosity=verbosity, outdir=outdir, stop_early=stop_early,
                                     extension_params={"ImmuneByAgeExtension": self.infection_params})
                    extenstion_name = get_config_param("ExtensionsNamelst")
                    ext_param = {extenstion_name[0]: self.infection_params} if extenstion_name != [] else {}

                    self.interventions = []
                    sim.update_events(self.interventions, ext_param)

                    self.infection_params.infect_simulation(sim, outdir)

                    datas_to_plot, extensions = sim.get_extensions(self.days,
                                                                   self.datas_to_plot,
                                                                   self.scenario_name,
                                                                   Extensionslst)

                    runner.get_trajectory(info=(sim, agents[idx], economic, runner, extensions, self))

                    # sim.save_stats(self.datas_to_plot, self.scenario_name, info=(runner, agent))
            else:
                raise "No mode for model usage specified"
        else:
            raise "NO corresponding model file is found. Please specify correct path"

    def check_and_find_folder(self, date_time_str, search_directory, city_name):
        # Define the regex pattern for the date and time format "YYYY-MM-DD_HH-MM-SS"
        # seconds on disk and wands are different - so without second
       
        if date_time_str: 
            pattern = r"^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}"#_\d{2}$"
            date_time_str = date_time_str[:-3]
            print(date_time_str)
#        else:
#            date_time_str = None


            log.info(f'~~~{date_time_str}')
            if date_time_str:
            # Check if the string matches the date-time pattern
                if re.match(pattern, date_time_str):
                # Try to parse the string into a datetime object to ensure it's a valid date-time
                    try:
                        target_datetime = datetime.datetime.strptime(date_time_str, "%Y_%m_%d_%H_%M")  # No %S (seconds)# %S
                    except ValueError:
                        return False

                    found_paths = []  # List to store paths to matching files

                # Search for folders with this date in the name
                    for root, dirs, files in os.walk(search_directory):
                        for dir_name in dirs:
                            if date_time_str in dir_name:
                                dir_path = os.path.join(root, dir_name)
                                for sub_root, sub_dirs, sub_files in os.walk(dir_path):
                                    for s in [f'{dir_path}/{x}' for x in sub_dirs]:
                                         for _,_,f in os.walk(s):
                                            for file in f:
                                                if file.endswith('.h5') or city_name.lower() in file:
                                                    found_paths.append(f'{s}/{file}')
                                    break# Add path to the list

                    if found_paths:
                        self.drop_empty_folders(search_directory, target_datetime)
                        return found_paths  # Return the list of paths if any are found
                    else:
                     #self.drop_empty_folders(search_directory, target_datetime)
                        return False
                else:
                    return False
            else:
                return False
        else:
            return None

    def drop_empty_folders(self, search_directory, target_datetime):
        empty_folders = []
        for root, dirs, _ in os.walk(search_directory):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)

                # Get directory creation/modification time
                dir_creation_time = datetime.datetime.fromtimestamp(os.path.getmtime(dir_path))

                # Check if the directory is created after the target date and time
                if dir_creation_time > target_datetime:
                    # Check if the directory is empty (i.e., no files or subdirectories)
                    if self.is_root_empty(dir_path):
                        empty_folders.append(dir_path)
                    '''
                    if self.is_root_empty(dir_path):
                        print(f"Deleting empty folder: {dir_path}")
                        shutil.rmtree(dir_path)
                    '''
                    '''
                    if not any(os.scandir(dir_path)):
                        print(f"Deleting empty folder: {dir_path}")
                        shutil.rmtree(dir_path)  # Remove the empty directory and its contents
                    '''
            if empty_folders:
                empty_folders.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                print('+++++',empty_folders)
                # Keep the latest empty folder, delete others
                for folder in empty_folders[1:]:  # Skip the most recent one
                    print(f"Deleting empty folder: {folder}")
                    shutil.rmtree(folder)
                return

    def is_root_empty(self, root_path):
        """Check if the root directory or any of its subdirectories contains no files."""
        for _, _, files in os.walk(root_path):
            if files:  # If there are any files in the structure, it's not empty
                return False
        return True

class RepeatJob(RunningJob):
    """
    A subclass that implements a job which repeat a given SimpleJob several times,
    and plots the mean, std and confidence of each of the outputs.
    If datas_to_plot is not given it plots the same graphs of its repeated job.
    """

    def __init__(self, job, num_repetitions, datas_to_plot=None):
        """
        initialize a repeated job
        :param job: SimpleJob object to repeatedly run
        :param num_repetitions: int times to run the job
        :param datas_to_plot: states what data from the simulation will be counted and saved to output plots,
        see DataToPlot doc. The default behavior if the param is omitted, is to generate the outputs of the given job
        """
        super(RepeatJob, self).__init__(
            job.scenario_name, job.city_name, job.scale,
            job.days, job.initial_date, job.params_to_change, job.interventions
        )
        assert isinstance(job, SimpleJob), "No implementation of RepeatJob(job) for job of type %s" % type(job)
        self.num_repetitions = num_repetitions
        self.datas_to_plot = datas_to_plot
        if self.datas_to_plot is None:
            self.datas_to_plot = job.datas_to_plot
        self.jobs = [copy.deepcopy(job) for ind in range(num_repetitions)]
        for ind in range(num_repetitions):
            self.jobs[ind].scenario_name = "sample_" + str(ind)

    def update_params(self, params_change):
        """
        Update the param changes from the given job to all the jobs
        :param params_change: dict of changes to Params
        """
        super(RepeatJob, self).update_params(params_change)
        for job in self.jobs:
            job.update_params(params_change)

    def generate_tasks(self, outdir, stop_early=None):
        """
        creates the output directory for the run, and returns the all tasks to run
        :param outdir: the path to the output directory
        :param stop_early: only relevant to R computation, see Simulation doc
        :return: list of one Task
        """
        outdir = os.path.join(outdir, self.scenario_name)
        assert not os.path.exists(outdir), "Directory '%s' already exists!" % outdir
        os.makedirs(outdir)
        return sum([job.generate_tasks(outdir, stop_early) for job in self.jobs], [])

    def finalize(self, outdir):
        """
        handles the outputs that need to be done after all the tasks are finished.
        Here the computation of mean, std and confidence of all the task is done, and saved to output directory
        :param outdir: output directory path
        """
        outdir = os.path.join(outdir, self.scenario_name)
        stats_files = []
        for index in range(self.num_repetitions):
            stats_files.append(os.path.join(
                outdir, self.jobs[index].scenario_name,
                "statistics.pkl"
            ))
        for name, data_to_plot in self.datas_to_plot.items():
            get_mean_and_confidence_from_statistics(stats_files, data_to_plot, self.scenario_name + "_" + name, outdir)
        get_multiple_stats_summary_file(stats_files, self.scenario_name, outdir, shortened=False)
        get_multiple_stats_summary_file(stats_files, self.scenario_name, outdir, shortened=True)
        get_r_mean_and_confidence_from_statistics(stats_files, self.scenario_name, outdir)

    def get_all_params_changes(self):
        """
        return all the params changes of all the jobs
        """
        return sum([job.get_all_params_changes() for job in self.jobs], [])


class ParamChangeRJob(RunningJob):
    """
    A subclass that implements a job which repeat a given job and change each time one given parameter value.
    each repeated job is the same, except for the parameter value. Also, each job can be repeated several times.

    The job plots R0 as a function of the parameter value.
    (It can plot additional Rs (R1, R2, ...) when given num_rs)
    """

    def __init__(self, job, param_name, param_range, num_rs=0):
        """
        initialize the job
        :param job: RepeatJob
        :param param_name: the parameter name
        :param param_range: the list of parameter values
        :param num_rs: number of Rs to plot, the default is 0 for R0
        """
        super(ParamChangeRJob, self).__init__(
            job.scenario_name, job.city_name, job.scale,
            job.days, job.initial_date, job.params_to_change, job.interventions
        )
        assert isinstance(job, RepeatJob), "No implementation of ParamChangeRJob(job) for job of type %s" % type(job)
        self.param_name = param_name
        self.param_range = param_range
        self.num_repetitions = len(param_range)
        self.jobs = [copy.deepcopy(job) for ind in range(self.num_repetitions)]
        self.num_rs = num_rs
        for ind in range(self.num_repetitions):
            self.jobs[ind].scenario_name = "_".join(param_name) + "_" + str(param_range[ind])
            self.jobs[ind].update_params({param_name: param_range[ind]})

    def update_params(self, params_change):
        """
        Pass all the params changes to all the jobs
        :param params_change: dict o f param changes
        """
        super(ParamChangeRJob, self).update_params(params_change)
        for job in self.jobs:
            job.update_params(params_change)

    def generate_tasks(self, outdir, stop_early=None):
        """
        generates all the task of each job, one job per param change
        :param outdir: the output directory path
        :param stop_early: TBD, multiple stopping when computing R
        :return: list of all tasks to run
        """
        assert stop_early is None, "multiple stopping not yet supported"
        outdir = os.path.join(outdir, self.scenario_name)
        os.makedirs(outdir)
        return sum([job.generate_tasks(outdir, ("r", self.num_rs)) for job in self.jobs], [])

    def finalize(self, outdir):
        """
        After all the jobs are done, make the output files that are related to R computations
        :param outdir: output directory path
        """
        outdir = os.path.join(outdir, self.scenario_name)
        param_and_stats_files = []
        for index in range(self.num_repetitions):
            stats_files = []
            job = self.jobs[index]
            for job2 in job.jobs:
                stats_files.append(os.path.join(
                    outdir, job.scenario_name, job2.scenario_name,
                    "statistics.pkl"
                ))
            param_and_stats_files.append((self.param_range[index], stats_files))
        compute_r_from_statistics(param_and_stats_files, self.num_rs, self.scenario_name, outdir)

    def get_all_params_changes(self):
        """
        returns all the param changes of this job
        :return: list of param changes
        """
        return sum([job.get_all_params_changes() for job in self.jobs], [])


def create_city_and_serialize(city_name, scale, params_to_change):
    """
    Generate population of a given city.
    Done once for each triplet (city, scale, params_to_change).
    :param city_name: str city name, "all" for entire country
    :param scale: float between 0-1, that states the size proportion of the city. 1 if for actual size
    :param params_to_change: dict of params to change, in Params object
    :return: World object
    """
    config_path = os.path.dirname(__file__) + "/config.json"
    with open(config_path) as json_data_file:
        ConfigData = json.load(json_data_file)
        citiesDataPath = ConfigData['CitiesFilePath']
        paramsDataPath = ConfigData['ParamsFilePath']

    Params.load_from(os.path.join(os.path.dirname(__file__), paramsDataPath), override=True)

    # Check if dictionary is empty
    if not params_to_change:
        for param, val in params_to_change.items():
            Params.loader()[param] = val
        population_loader = PopulationLoader(citiesDataPath,
                                             added_description=Params.loader().description(),
                                             with_caching=False
                                             )
    else:
        population_loader = PopulationLoader(citiesDataPath,
                                             added_description="",
                                             with_caching=False
                                             )
    population_loader.get_world(city_name=city_name, scale=scale)


def generate_all_cities_for_jobs(jobs, cpus_to_use):
    """
    Generate all the population needed before the jobs starts to run, and serialize it. That way,
    The multi processed runs don't try to generate the population at the same time, and can use the cashed results.
    :param jobs: all the jobs to be running
    :param cpus_to_use: the number of cpu cores to use, if grater than 1, the run will be multi processed
    """
    appearing_cities = set((job.city_name, job.scale) for job in jobs)
    appearing_cities_to_params = {city: [] for city in appearing_cities}
    for job in jobs:
        appearing_cities_to_params[(job.city_name, job.scale)] += job.get_all_params_changes()
    for city in appearing_cities:
        appearing_cities_to_params[city] = list(
            map(dict, set(tuple(sorted(d.items())) for d in appearing_cities_to_params[city])))

    appearing_cities_and_params = [
        (city_name, scale, params_to_change)
        for city_name, scale in appearing_cities
        for params_to_change in appearing_cities_to_params[(city_name, scale)]
    ]
    print("Generating all cities...")
    if cpus_to_use == 1:
        for city_name, scale in appearing_cities:
            create_city_and_serialize(city_name, scale, Params.loader())
        for city_name, scale, params_to_change in appearing_cities_and_params:
            create_city_and_serialize(city_name, scale, params_to_change)
    else:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(cpus_to_use)
        futures = []

        for city_name, scale, params_to_change in appearing_cities_and_params:
            futures.append(pool.apply_async(
                create_city_and_serialize, args=(city_name, scale, params_to_change)
            ))
        pool.close()
        pool.join()
        for future in futures:
            future.get()
    print("Done generating cities.")


def create_outdir():
    """
    Creates the output directory for the entire run.
    :return: returns the directory path
    """
    all_outdir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
    curr_outdir = os.path.join(all_outdir, time.strftime('%Y_%m_%d_%H_%M_%S'))
    assert not os.path.exists(curr_outdir), "Directory '%s' already exists!" % curr_outdir
    os.makedirs(curr_outdir)
    print('your outputs will be in: {}'.format(curr_outdir))
    return curr_outdir


def calc_CPU_count(maxCPU: int, percentCPU: float, jobs):
    if len(jobs) > 700:
        percentCPU = 0.75
    if len(jobs) > 850:
        percentCPU = 0.5
    if len(jobs) > 1000:
        percentCPU = 0.25

    return min(int(math.floor(mp.cpu_count() * percentCPU)), maxCPU)


def run(jobs, multi_processed=True, with_population_caching=True, verbosity=True):
    """
    This func handles the user's run of the given simulation jobs.
    The run of the jobs can be multi processed, with each simulation as a unique process, and can use cached population
    to save time.
    """
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path) as json_data_file:
        ConfigData = json.load(json_data_file)
        percentStr = ConfigData['CPU_percent']
        percent = float(percentStr)

    outdir = create_outdir()
    cpus_to_use = calc_CPU_count(30, percent, jobs)

    if cpus_to_use == 0 or not multi_processed:
        cpus_to_use = 1

    tasks_sets = [job.generate_tasks(outdir) for job in jobs]
    finalizers = [job.finalize for job in jobs]
    if cpus_to_use == 1:
        prog_bar = tqdm(total=sum(len(task_set) + 1 for task_set in tasks_sets))
        for task_set, finalizer in zip(tasks_sets, finalizers):
            for task in task_set:
                job_outdir, stop_early = task.params
                task.func(
                    job_outdir,
                    stop_early,
                    with_population_caching=with_population_caching,
                    verbosity=verbosity
                )
                prog_bar.update()
            finalizer(outdir)
            prog_bar.update()
        wandb.finish()
    else:
        if with_population_caching:
            generate_all_cities_for_jobs(jobs, cpus_to_use)
        print('running a pool of {} threads parallelly'.format(cpus_to_use))
        sys.stdout.flush()
        prog_bar = tqdm(total=sum(len(task_set) + 1 for task_set in tasks_sets))
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(cpus_to_use)

        finalize_futures = []

        def get_callback(finalizer, task_set, task):
            def callback(_):
                prog_bar.update()
                task.is_done = True
                if all(t.is_done for t in task_set):
                    finalize_futures.append(pool.apply_async(
                        finalizer, args=(outdir,),
                        callback=lambda _: prog_bar.update()
                    ))

            return callback

        futures = []
        for task_set, finalizer in zip(tasks_sets, finalizers):
            for task in task_set:
                futures.append(pool.apply_async(
                    task.func,
                    args=(*task.params, with_population_caching, verbosity),
                    callback=get_callback(finalizer, task_set, task)
                ))
        for future in futures:
            future.wait()
            future.get()
        pool.close()
        pool.join()
        for future in finalize_futures:
            future.get()
    sys.stderr.flush()
    print('end')
    return outdir


def make_base_infectiousness_to_r_job(scenario_name, city_name, scale, param_range,
                                      interventions=None, num_repetitions=7, initial_num_infected=100, num_rs=0):
    """
    Wraps the inialization of a job that creates a graph of R as a function of base infectiousness value.

    :param scenario_name: name of the run, to use in output directory and output files
    :param city_name: the city name to simulate
    :param scale: city size scale
    :param param_range: list of all the base infectiousness values
    :param interventions: list of interventions to apply during the simulation
    :param num_repetitions: int repetition for each base infectiousness value
    :param num_rs: number of Rs to compute
    :return: ParamChangeRJob
    """
    simple_job = SimpleJob(scenario_name, city_name, scale, interventions=interventions,
                           infection_params=NaiveInitialInfectionParams(initial_num_infected, 0, 1, ORDER.NONE,
                                                                        city_name,
                                                                        InitialImmuneType.GENERAL_POPULATION))
    repeated_job = RepeatJob(simple_job, num_repetitions)
    return ParamChangeRJob(repeated_job, ("person", "base_infectiousness"), param_range, num_rs=num_rs)
