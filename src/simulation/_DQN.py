from __future__ import absolute_import

# == for SimpleJob() functions
from src.util import seed
from src.logs import *
from src.simulation.params import Params
from src.world.population_generation import PopulationLoader
from src.simulation.interventions import *
from src.simulation.interventions.intervention import LockdownIntervention
from src.seir.disease_state import DiseaseState

import os, json
import pandas as pd
import random, re
import numpy as np
import math
from collections import deque
import logging, inspect
from collections import Counter

# -- agent actions by calling interventions
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datetime import timedelta, date, datetime

# !@ ---
import subprocess
import wandb
import tensorflow
from tensorflow import keras
#from tensorflow.keras.callbacks import LearningRateScheduler
#from tensorflow.keras.optimizers import Adam

from keras import layers, models, optimizers, losses
from .. import scenarios

log = logging.getLogger(__name__)

# == MODEL OF ENV
class EpidemicStats:
    '''
    # TODO: update description
    calculate reward and economic index with existed statistics
    '''

    def __init__(self, is_ratio=True):
        self.is_ratio = is_ratio
        self.age_groups = [tuple(x) for x in get_config_param("age_groups")]

        self.susceptible_population = None
        self.vaccinated_population = None
        self.infected_population = None
        self.hospitalized_population = None
        self.recovered_population = None
        self.deceased_population = None
        self.capable_population = None
        self.total_population = None
        self.exposed_population = None
        self.cum_vaccinated = None

        self.chosen_action = 0

        # {'0-18': 0, '18-60': 0, '60-99': 0}
        self.economic_index = 0  # {'0-18': 0, '18-60': 0, '60-99': 0}
        self.reward = 0
        hyper = get_config_param(config_name="dqn_agent")['train_params']["hyper_params"]
        if hyper["constant_flag"]:
            self.alpha = hyper["alpha"]
            self.beta = hyper["beta"]
            self.gamma = hyper["gamma"]
            self.delta = hyper["delta"]
        else:
            self.alpha = 1
            self.beta = 1
            self.gamma = 1
            self.delta = 1
        get_log_history()

    def get_population_statistic(self, simulate_state, vaccinated_seq, num_to_infect, is_init_period, aged_flag=False):
        if aged_flag:
            return self.get_statistics_by_age(simulate_state, vaccinated_seq, num_to_infect, is_init_period)
        else:
            return self.get_total_statistics(simulate_state, vaccinated_seq, num_to_infect, is_init_period)

    def split_population_equally(self, num_to_infect, age_groups):
        try:
            # Determine the number of age groups
            num_groups = len(age_groups)

            # Calculate mean and standard deviation based on the number of age groups
            mean = (num_groups - 1) / 2

            # Dynamically adjust std_dev: a higher value will spread the distribution more
            # You can change the denominator to a higher number for a narrower distribution
            std_dev = num_groups / 6  # Adjust this value as needed

            # Generate random values from a normal distribution
            normal_values = np.random.normal(loc=mean, scale=std_dev, size=num_groups)

            # Ensure all values are non-negative and within the range of indices
            normal_values = np.clip(normal_values, 0, num_groups - 1)

            # Check if the sum of normal_values is zero to prevent division by zero
            sum_normal_values = normal_values.sum()
            if sum_normal_values == 0:
                # Assign default value (e.g., equal distribution)
                normal_splits = np.ones(num_groups) / num_groups
            else:
                # Scale the values to sum to 1
                normal_splits = normal_values / sum_normal_values

            # Scale the splits to the total number of infections
            infections_per_group = (normal_splits * num_to_infect).astype(int)

            # Adjust to ensure the sum equals num_to_infect
            diff = int(num_to_infect - infections_per_group.sum())
            for i in range(diff):
                infections_per_group[i % num_groups] += 1

            # Create the dictionary for infected_population
            infected_population = {age_group: infections_per_group[i] for i, age_group in enumerate(age_groups)}

            return infected_population
        except Exception as e:
            log.error(f"An error occurred in split_population_equally: {e}")
            return {age_group: 0 for age_group in age_groups}

    def get_statistics_by_age(self, simulate_state, regular_immune_seq, num_to_infect, is_init_period):
        self.susceptible_population = {age_group: 0 for age_group in self.age_groups}
        self.vaccinated_population = {age_group: 0 for age_group in self.age_groups}

        if get_config_param("init_infected_equally"):  # same value for each age group. X+X+X
            self.infected_population = {age_group: num_to_infect for age_group in self.age_groups}
        else:  # split initial number to all groups X/3 +X/3+X/3
            self.infected_population = self.split_population_equally(num_to_infect, self.age_groups)

        self.hospitalized_population = {age_group: 0 for age_group in self.age_groups}
        self.recovered_population = {age_group: 0 for age_group in self.age_groups}
        self.deceased_population = {age_group: 0 for age_group in self.age_groups}
        self.capable_population = {age_group: 0 for age_group in self.age_groups}
        self.total_population = {age_group: 0 for age_group in self.age_groups}
        self.exposed_population = {age_group: 0 for age_group in self.age_groups}
        self.vaccinated_population = {age_group: 0 for age_group in self.age_groups}
        self.immuned_by_extension = {age_group: 0 for age_group in self.age_groups}

        self.over_70_population_ratio = 0
        counts = {}
        immuned = {}
        world_ = simulate_state._world.all_people()

        #if self.cum_vaccinated is None:# or isinstance(self.cum_vaccinated, int):
        #    self.cum_vaccinated = 0 #{age_group: 0 for age_group in self.age_groups}  # Initialize as a dictionary of zeros
        self.over_70_population_ratio = len(Counter([person for person in world_ if person.get_age() > 70])) / len(
            world_)
        for age_key in self.capable_population.keys():
            a, b = age_key
            counts[age_key] = Counter([person.get_disease_state() for person in world_ if a < person.get_age() < b])

            regular = Counter([person.get_disease_state() for person in regular_immune_seq if a < person.get_age() < b])
            print(age_key, counts[age_key])
            print('\t ==> EXT:', regular)

            self.immuned_by_extension[age_key] = regular.most_common(1)[0][1] if len(regular) > 0 else 0

            self.total_population[age_key] = len(
                Counter([person for person in world_ if a < person.get_age() < b])) if not self.is_ratio else 1
            cap_tmp = len(
                Counter([person for person in world_ if a < person.get_age() < b])) if self.is_ratio else 1

            for ele in counts[age_key]:
                if DiseaseState.is_susceptible(ele):
                    self.susceptible_population[age_key] = self.to_int(counts[age_key][ele],
                                                                       self.total_population[age_key])
                if DiseaseState.is_infected(ele):
                    self.infected_population[age_key] += self.to_int(counts[age_key][ele],
                                                                     self.total_population[age_key])
                if DiseaseState.is_critical(ele):
                    print('++>> H', age_key, counts[age_key][ele])
                    self.hospitalized_population[age_key] = self.to_int(counts[age_key][ele],
                                                                        self.total_population[age_key])
                if DiseaseState.is_exposed(ele):
                    self.exposed_population[age_key] += self.to_int(counts[age_key][ele],
                                                                    self.total_population[age_key])
                if DiseaseState.is_terminal(ele):
                    if ele.name.lower() == "deceased":
                        self.deceased_population[age_key] = self.to_int(counts[age_key][ele],
                                                                        self.total_population[age_key])
                    elif ele.name.lower() == "immune":
                        self.recovered_population[age_key] = self.to_int(counts[age_key][ele],
                                                                         self.total_population[age_key])
                if DiseaseState.is_vaccinated(ele):
                    self.vaccinated_population[age_key] += self.to_int(counts[age_key][ele],
                                                                       self.total_population[age_key])

            self.total_population[age_key] = cap_tmp
            try:
                self.capable_population[age_key] = (cap_tmp -
                                                    self.deceased_population[age_key] -
                                                    self.hospitalized_population[age_key] -
                                                    self.infected_population[age_key])
            except OverflowError:
                self.capable_population[age_key] = (cap_tmp -
                                                    self.deceased_population[age_key] -
                                                    self.hospitalized_population[age_key])

            self.total_population[age_key] = 1 if self.susceptible_population[age_key] < 1 else self.total_population[
                age_key]

        ''' DEBUG purposes
        a = sum(self.infected_population.values())
        b = sum(self.total_population.values())
        if (a / b) > 1:
            print('?? A/B', a / b)
        '''
        self.cum_vaccinated += sum(self.vaccinated_population.values())
        log.info(f'TO BE IMMUNED BY EXTENSION NEXT DAY {self.immuned_by_extension}')

        #if not is_init_period:
        self.get_economic_index(is_init_period, aged=True)

    def to_int(self, count, total=None):
        if not total:
            x = count / self.total_population
        else:
            x = count / total
        return int(x) if x.is_integer() else x

    def get_total_statistics(self, simulate_state, regular_immune_seq, num_to_infect, is_init_period):
        # TODO: test this type of statistic
        self.susceptible_population = 0
        self.vaccinated_population = 0
        self.infected_population = 0  #num_to_infect if num_to_infect else 0
        self.hospitalized_population = 0
        self.recovered_population = 0
        self.deceased_population = 0
        self.capable_population = 0
        self.total_population = 0
        self.exposed_population = 0
        self.immuned_by_extension = 0
        self.over_70_population_ratio = 0

        world_ = simulate_state._world.all_people()

        counts = Counter([person.get_disease_state() for person in world_])
        regular = Counter([person.get_disease_state() for person in regular_immune_seq])
        self.immuned_by_extension = regular.most_common(1)[0][1] if len(regular) > 0 else 0
        self.over_70_population_ratio = len(Counter([person for person in world_ if person.get_age() > 70])) / len(
            world_)

        # Calculate total population size
        self.total_population = sum(counts.values()) if not self.is_ratio else 1
        cap_tmp = sum(counts.values()) if self.is_ratio else 1

        for disease_state, count in counts.items():
            if DiseaseState.is_susceptible(disease_state):
                self.susceptible_population = self.to_int(count)
            if DiseaseState.is_exposed(disease_state):
                self.exposed_population += self.to_int(count)
            if DiseaseState.is_infected(disease_state):
                self.infected_population += self.to_int(count)
            if DiseaseState.is_critical(disease_state):
                self.hospitalized_population = self.to_int(count)
            if DiseaseState.is_vaccinated(disease_state):
                self.vaccinated_population += self.to_int(count)
            if DiseaseState.is_terminal(disease_state):
                if disease_state.name.lower() == "deceased":
                    self.deceased_population = self.to_int(count)
                elif disease_state.name.lower() == "immune":
                    # self.vaccinated_population = count / total_population
                    self.recovered_population = self.to_int(count)

        self.cum_vaccinated += self.vaccinated_population
        # Calculate capable population
        self.capable_population = (cap_tmp -
                                   self.deceased_population -
                                   self.hospitalized_population -
                                   self.infected_population)
        self.total_population = cap_tmp
        self.total_population = 1 if self.susceptible_population < 1 else self.total_population

        #if not is_init_period:
        self.get_economic_index(is_init_period)

    def get_economic_index(self, is_init_period, aged=False):
        '''
        Economic index from
        "A simulation-deep reinforcement learning (SiRL) approach for epidemic control optimization"

        without V2 (since only V1 in Coderona model)
        without C (since ICU Coderona model hasn't ICU statistics)

        Et = S + V1 + R − α×I − β×H − D
        alpha (incubation rate) = 0.19, 1
        beta (transmission rate) = 0.12, 0.45
        '''
        hyperparam_dict = get_config_param("dqn_agent")['train_params']["actions_weights"]
        if self.chosen_action >= 0 and not is_init_period:
            if aged:
                ec = 0
                for k in self.age_groups:
                    decrease_coef_ = hyperparam_dict[str(self.chosen_action)]
                    if isinstance(decrease_coef_, list):
                        vac_decrease_coef = decrease_coef_[0]
                        decrease_coef = decrease_coef_[1]
                    else:
                        vac_decrease_coef = 1

                    if k == (0, 18):
                        decrease_coef = 1
                        print(f'{k}\t{vac_decrease_coef} {decrease_coef} ')
                        print(self.hospitalized_population[k])
                        vsr = (self.susceptible_population[k] +
                               self.vaccinated_population[k] * vac_decrease_coef +
                               self.recovered_population[k]) * decrease_coef

                    if k == (18, 60):
                        print(f'{k}\t{vac_decrease_coef} {decrease_coef} ')
                        vsr = (self.susceptible_population[k] +
                               self.vaccinated_population[k] * vac_decrease_coef +
                               self.recovered_population[k]) * decrease_coef

                    if k == (60, 99):
                        print(f'{k}\t{vac_decrease_coef} {decrease_coef} ')
                        self.susceptible_population[k] = int(
                            self.susceptible_population[k] * (1 - self.over_70_population_ratio))
                        self.vaccinated_population[k] = int(
                            self.vaccinated_population[k] * (1 - self.over_70_population_ratio))
                        self.recovered_population[k] = int(
                            self.recovered_population[k] *(1 - self.over_70_population_ratio))
                        elder_sus = int(self.susceptible_population[k] * self.over_70_population_ratio)
                        elder_vac = int(self.vaccinated_population[k] * self.over_70_population_ratio)
                        elder_rec = int(self.recovered_population[k] * self.over_70_population_ratio)

                        vsr = (self.susceptible_population[k] +
                               self.vaccinated_population[k] * vac_decrease_coef +
                               self.recovered_population[k]) * decrease_coef + (elder_sus+
                                                                                elder_vac+
                                                                                elder_rec)


                    ec += (self.delta * vsr -
                           self.alpha * self.infected_population[k] - self.beta * self.hospitalized_population[k] -
                           self.gamma * self.deceased_population[k])

                print(f'\t >> ECONOMIC INDEX = {ec}')
                self.economic_index = ec
                #self.economic_index = self.normalize(self.economic_index)
            else:
                self.economic_index = (
                        self.delta * (self.susceptible_population +
                                      self.vaccinated_population +
                                      self.recovered_population) * decrease_coef -
                        self.alpha * self.infected_population - self.beta * self.hospitalized_population -
                        self.gamma * self.deceased_population)
                #import math
                self.economic_index = math.exp(self.economic_index)
                print(f'\t >> ECONOMIC INDEX: {self.economic_index}')
                # TODO: make age separation for decrease_coef
            # self.economic_index = self.normalize(self.economic_index)
        else:
            self.economic_index = -1000000
        get_log_history()

    def normalize(self, value, min_value=0, max_value=1):
        return (value - min_value) / (max_value - min_value)

    def get_state(self):
        bin_width = 1.0 / 100
        bin_index = int(self.susceptible_population / bin_width)

        return np.array([self.susceptible_population, self.vaccinated_population, self.infected_population,
                         self.hospitalized_population, self.recovered_population, self.deceased_population,
                         self.economic_index])  # TODO: add tested and quarantined?
#=========================================================================================================
class Interventions:
    def __init__(self, runner, job):
        self.compliance = runner.compliance
        self.ci_delay = runner.ci_delay
        self.hi_delay = runner.hi_delay
        # INITIAL_DATE = date(year=2020, month=2, day=27)

        self.duration = timedelta(get_config_param("intervention_duration"))
        self.houses_per_day = get_config_param("vaccinate_per_day_household")
        self.vac_per_day = get_config_param("vaccinate_per_day_persons")

        if runner.cur_day == None:
            self.start_date = job.initial_date + timedelta(days=0)
        else:
            self.start_date = job.initial_date + timedelta(days=runner.cur_day)
        self.city_name = runner.city_name

    def immune_by_household(self, min_age:int, max_age:int, immune_type:str="vaccine")->ImmuneByHouseholdIntervention:
        return ImmuneByHouseholdIntervention(start_date=self.start_date,
                                             duration=self.duration,
                                             compliance=self.compliance,
                                             houses_per_day=self.houses_per_day,
                                             min_age=min_age,
                                             max_age=max_age,
                                             immune_type=immune_type
                                             )
    def immune_general(self, min_age:int, max_age:int, immune_type:str="vaccine")->ImmuneGeneralPopulationIntervention:
        return ImmuneGeneralPopulationIntervention(compliance=self.compliance,
                                                   start_date=self.start_date,
                                                   duration=self.duration,
                                                   people_per_day=self.vac_per_day,
                                                   min_age=min_age,
                                                   max_age=max_age,
                                                   immune_type=immune_type
                                                   )

    def lockdown(self, city_name:str="all")->LockdownIntervention:
        return LockdownIntervention(start_date=self.start_date + timedelta(0.0),
                                    duration=self.duration,  # daysdelta(4 * 7),
                                    compliance=self.compliance,
                                    city_name=city_name
                                    )

    def curfew(self)->CityCurfewIntervention:
        return CityCurfewIntervention(self.city_name,
                                      start_date=self.start_date,
                                      duration=self.duration,  # daysdelta(120),
                                      compliance=self.compliance)

    def elderly_quarantine(self, min_age:int=70)->ElderlyQuarantineIntervention:
        return ElderlyQuarantineIntervention(start_date=self.start_date,
                                             duration=self.duration,
                                             compliance=self.compliance,
                                             min_age=min_age
                                             )
    def symptomatic_isolation(self)->SymptomaticIsolationIntervention:
        return SymptomaticIsolationIntervention(start_date=self.start_date,
                                                duration=self.duration,
                                                compliance=self.compliance,
                                                delay=self.ci_delay
                                                )

    def workplace_isolation(self)->WorkplaceClosureIntervention:
        return WorkplaceClosureIntervention(start_date=self.start_date,
                                            duration=self.duration,
                                            compliance=self.compliance
                                            )
    def school_closure(self, proportion_of_envs:float=1.0, city_name:str="all", age_segments:tuple=(3,22))->SchoolClosureIntervention:
        return SchoolClosureIntervention(start_date=self.start_date,
                                         duration=self.duration,
                                         compliance=self.compliance,
                                         proportion_of_envs=proportion_of_envs,
                                         city_name=city_name,
                                         age_segment=age_segments
                                         )
    def social_distance(self, age_range:tuple=(0,99))->SocialDistancingIntervention:
        return SocialDistancingIntervention(start_date=self.start_date,
                                            duration=self.duration,
                                            compliance=self.compliance,
                                            age_range=age_range
                                            )
    def household_isolation(self)->HouseholdIsolationIntervention:
        return HouseholdIsolationIntervention(start_date=self.start_date,
                                              duration=self.duration,
                                              compliance=self.compliance,
                                              delay_on_enter=self.hi_delay
                                              )

class ReplayBuffer:
    MEMORY_SIZE = 900000000
    def __init__(self):
        self.memory = deque(maxlen=self.MEMORY_SIZE)
        self.reward_memory_total = deque(maxlen=self.MEMORY_SIZE)
        self.terminal_memory_total = deque(maxlen=self.MEMORY_SIZE)

        self.day_range = get_config_param("latency days_n")  # initial period for get historical dynamics
        self.intervention_dur = get_config_param("intervention_duration")

        self.train_memory = deque(maxlen=self.MEMORY_SIZE)
        self.terminal_memory = deque(maxlen=self.MEMORY_SIZE)

    def add(self, experience):
        self.train_memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.train_memory, batch_size)

    def size(self):
        return len(self.train_memory)


# DQN Agent
class AgentLSTM:
    def __init__(self, runner, economic, job, model=None, epsilon=None):
        self.replay_buffer = ReplayBuffer()
        self.discount_factor = 0.95
        if epsilon:
            with open(epsilon) as json_data_file:
                ConfigData = json.load(json_data_file)
            self.epsilon = ConfigData['epsilon']
        else:
            self.epsilon = get_config_param("dqn_agent")['train_params']["epsilon"]
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.lr = get_config_param("dqn_agent")['train_params']["learning_rate"]
        #self.loss_fn = losses.MeanSquaredError()
        self.loss_fn = losses.Huber()
        self.optimizer = tensorflow.keras.optimizers.Adam(learning_rate=self.lr)
        self.batch_size = get_config_param("dqn_agent")['train_params']["batch_size"]
        self.target_update_frequency = 10
        self.update_target_counter = 0

        self.time_steps = self.replay_buffer.day_range  # self.day_range - 1
        self.n_outputs = None
        self.input_shape = None

        self.cum_reward_graph = []
        self.model_history = []
        self.mean_action = []
        self.mean_reward = 0

        self.update_act_dict(runner, economic, job)

        self.terminal = 0

        if model:
            if os.path.exists(model):
                log.info(model) # here should be path 
#                log.info(model.summary())
                self.model = keras.models.load_model(model)
                self.target_model = keras.models.load_model(model)
                log.info(f'Loaded model {self.model.summary()}')
                print('~~~~~~~~~~~~~~~~~~~~~~~',self.model.output)
                self.seen_states = {i: 0 for i in range(self.model.output.shape[2])}
               # print(self.seen_states)
        else:
            self.model = self.set_model(economic)
            self.target_model = self.set_model(economic)
        self.update_target_model()

        rew_type = get_config_param("dqn_agent")['train_params']["reward_type"]
        if rew_type['parametrized'] and not rew_type['penalty']:
            self.rew_type = 'parametrized'
        elif not rew_type['parametrized'] and rew_type['penalty']:
            self.rew_type = 'penalty'
        self.reward = 0
        get_log_history()

    def update_act_dict(self, runner, economic, job):
        act = Interventions(runner, job)
        house_interventions = []
        global_interventions = []
        upd_1_7 = []
        upd_2_3 = []
        upd_1_6 = []
        upd_1_5 = []
        x = economic.age_groups
        if len(x) > 0:
            for i in x:
                house_interventions.append(act.immune_by_household(min_age=i[0], max_age=i[1]))
                global_interventions.append(act.immune_general(min_age=i[0], max_age=i[1]))
                upd_2_3.append(act.immune_general(min_age=i[0], max_age=i[1]))
                upd_1_5.append(act.immune_by_household(min_age=i[0], max_age=i[1]))
                upd_1_6.append(act.immune_by_household(min_age=i[0], max_age=i[1]))
                upd_1_7.append(act.immune_by_household(min_age=i[0], max_age=i[1]))

        else:
            house_interventions.append(act.immune_by_household(min_age=0, max_age=99))
            global_interventions.append(act.immune_general(min_age=0, max_age=99))
            upd_2_3.append(act.immune_general(min_age=0, max_age=99))
            upd_1_5.append(act.immune_by_household(min_age=0, max_age=99))
            upd_1_6.append(act.immune_by_household(min_age=0, max_age=99))
            upd_1_7.append(act.immune_by_household(min_age=0, max_age=99))


        house_interventions.append(act.lockdown())
        global_interventions.append(act.curfew())
        upd_2_3.append(act.symptomatic_isolation())
        upd_1_5.append(act.school_closure(age_segments=(3, 22)))
        upd_1_6.append(act.workplace_isolation())
        upd_1_7.append(act.elderly_quarantine(min_age=70))

        self.act_dict = {0: scenarios.Empty_scenario(),
                         1: house_interventions,  # household_vac + lockdown
                         2: global_interventions, # general_vac + curfew
                         3: upd_2_3,  # general + symptomatic
                         4: [act.social_distance(age_range=(0,99))],
                         5: upd_1_5,  # household_vac + school_closure
                         6: upd_1_6,  # household_vac + workpalce_closure
                         7: upd_1_7, # household_vac + elderly_iso
                         8: [act.household_isolation()],
                         9: [act.curfew()],
                         10: [act.lockdown(city_name='all')]
                         }

    def is_terminal(self, terminal):
        #if terminal == -1:
        #    return True
        return False

    def execute_action(self, sim, action, economic, runner, simple_job_obj):
        '''
        Change hyperparameters of the E and Rw equations:
        alpha_ = 1
        beta_ = 1
        lambda_ = 1  # economic aspect first
        mu_ = 1  # infected count aspect first
        ro_ = 1  # deaths aspect first
        pi_ = 1  # vaccination aspect first

        Each of these depends on actions
        :param sim:
        :param action:
        :return:
        '''

        '''
        factor = 0.5
        sim.beta = beta * factor
        sim.sigma = sigma * factor
        sim.gamma = gamma * factor
        sim.mu = mu * factor
        sim.economic_impact_from_actions = factor

        '''
        sim.interventions = self.act_dict[action]
        simple_job_obj.interventions = self.act_dict[action]
        hyper = get_config_param(config_name="dqn_agent")['train_params']["hyper_params"]
        if hyper["constant_flag"]:
            self.lambda_ = hyper['lambda']
            self.mu_ = hyper['mu']
            self.ro_ = hyper['ro']
            self.pi_ = hyper['pi']
            economic.alpha = hyper['alpha']
            economic.beta = hyper['beta']
            economic.gamma = hyper['gamma']
            economic.delta = hyper['delta']
        else:
            raise "There is FALSE flag for dictionary for economic index hyperparameters coefficients"
        get_log_history()
        return simple_job_obj

    def set_model(self, economic):
        stats = [attr for attr in dir(economic) if
                 'population' in attr and not callable(getattr(economic, attr)) and not attr.startswith("__")
                 and not 'total' in attr]
        stats.remove('exposed_population')

        if get_config_param("age_stats"):
            new_stats = []
            for a in get_config_param("age_groups"):
                for s in stats:
                    new_stats.append(f'{s}_{a}')
            stats = new_stats
            state_size = len(stats)
        else:
            state_size = sum(
                len(getattr(economic, i).keys()) if isinstance(getattr(economic, i), dict) else 1 for i in stats)

        start_key = get_config_param("dqn_agent")['train_params']["actions"]["start_key"]
        end_key = get_config_param("dqn_agent")['train_params']["actions"]["end_key"]
        sub_dict = {key: self.act_dict[key] for key in range(start_key, end_key + 1)}
        self.input_shape = state_size + 1  # SEIRDE + Action
        self.n_outputs = len(sub_dict.keys())
        self.seen_states = {i: 0 for i in range(start_key, end_key+1)}
        #TODO: add state_size, action_size, model, target_model

        model = keras.Sequential([
        #keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True),
        #                           input_shape=(self.time_steps, self.input_shape)),
        keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True),
                                   input_shape=(1, self.input_shape)),
        keras.layers.Dropout(0.2),
        keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
        keras.layers.Dropout(0.2),
        keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(self.n_outputs, activation='linear') # softmax
        ])
        #model.add(Dense(action_size, activation='linear'))
        model.compile(loss=self.loss_fn, optimizer=self.optimizer)
        print(model.summary())
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            min_seen = min(self.seen_states.values())
            if get_config_param("dqn_agent")['force_exploring_flag']:
                min_seen_keys = [key for key, value in self.seen_states.items() if value == min_seen]
                x = random.choice(min_seen_keys)
            else:
                x = random.choice(list(self.seen_states.keys()))
            return x, "EPS"
        state = np.expand_dims(state, axis=0)
        print('++==++ act', state.shape)
        print('++==++ act', self.model.inputs)

        #state = np.expand_dims(state, axis=0)  # Shape becomes (1, 22, 1)
        q_values = self.model.predict(state)
        q_values = q_values[0]  # Remove the batch dimension to get shape (11,)
        action = np.argmax(q_values) # Find the action with the maximum Q-value
        return action, "MEMORY"

    def get_reward(self, env, sim):
        '''
        maximize the following multi-objective reward function for
        state θ := [Et, S, I, H, D, R, V1].

        :return:
        '''
        if isinstance(env.infected_population, dict):
            infect = sum(env.infected_population.values())
            dead = sum(env.deceased_population.values())
            sus = sum(env.susceptible_population.values())
            vac = sum(env.vaccinated_population.values())
            hos = sum(env.hospitalized_population.values())
            rec = sum(env.recovered_population.values())
            exp = sum(env.exposed_population.values())
            tot = sum(env.total_population.values())
        else:
            infect = env.infected_population
            dead = env.deceased_population
            sus = env.susceptible_population
            vac = env.vaccinated_population
            hos = env.hospitalized_population
            rec = env.recovered_population
            exp = env.exposed_population
            tot = env.total_population
            # x = sum(env.infected_population.values())

        if self.rew_type == 'parametrized':
            # PREVIOUS VERSION OF REWARD EQUATION (For testing only).
            # This parametrized reward function is NOT the reward used in the actual research.
            # I included it for internal testing and comparison purposes only.
            # The official reward used in experiments is the penalty-based formulation defined below.
            self.reward = (self.lambda_ * env.economic_index -
                           self.mu_ * infect -
                           self.ro_ * dead +
                           self.pi_ * (sus + vac))
        elif self.rew_type == 'penalty':
            # -- PENALTY PART
            # PENALTY REWARD VERSION
            # This is the reward formulation I actually used in the experiments described in the study.
            rt, r0, r_smoothed, r_estimated = sim.get_rt()

            print(f'\t >> RT = {rt}')
            if rt > 2.0:  #or rt == 0:  #!@ TODO: rt==0 is equal to rt is high
                self.reward = get_config_param("dqn_agent")["train_params"]["penalty_reward_params"][
                                  "rt>2.0"] * abs(env.economic_index)  # Severe penalty for high rates
            elif 1.5 < rt <= 2.0:
                self.reward = get_config_param("dqn_agent")["train_params"]["penalty_reward_params"][
                                  "1.5 < rt <= 2.0"] * abs(env.economic_index)  # Mild penalty for moderate rates
            elif 1.25 <= rt <= 1.5:
                self.reward = get_config_param("dqn_agent")["train_params"]["penalty_reward_params"][
                                  "1.25 <= rt <= 1.5"] * env.economic_index  # Reward for relatively stable condition
            elif 1.0 < rt < 1.25:
                self.reward = get_config_param("dqn_agent")["train_params"]["penalty_reward_params"][
                                  "rt < 1.25"] * env.economic_index  # High reward for optimal condition
            elif 0 < rt < 1.0:
                self.reward = get_config_param("dqn_agent")["train_params"]["penalty_reward_params"][
                                  "rt < 1.0"] * env.economic_index

            infection_ratio = infect / tot
            print(f'\t >> infection_ratio = {infection_ratio}')
            infect_rate = get_config_param("dqn_agent")["train_params"]["penalty_reward_params"]["infect_rate"]
            if infection_ratio > 0.01:
                self.reward -= get_config_param("dqn_agent")["train_params"]["penalty_reward_params"][
                                   "ratio > infect_rate"] * (infection_ratio / infect_rate)
            else:
                self.reward += get_config_param("dqn_agent")["train_params"]["penalty_reward_params"][
                                   "ratio < infect_rate"] * (infect_rate - infection_ratio)

            if self.replay_buffer.memory[-1][-1] < 0:  # if action < 0, or not action
                self.reward = -1000000
            print(f'\t >> REWARD = {self.reward}')
        else:
            raise f"You must specify ony one reward type to be active. Current settings is {rew_type}"
        get_log_history()

    def remember(self, state, action, reward, next_state):
        self.replay_buffer.add((state, action, reward, next_state))

    def replay(self, runner):
        if self.replay_buffer.size() < self.batch_size:
            return
        minibatch = self.replay_buffer.sample(self.batch_size)

        #minibatch = [self.replay_buffer[i] for i in indices]
        states = np.array([e[0] for e in minibatch])
        actions = np.array([e[1] for e in minibatch])
        rewards = np.array([e[2] for e in minibatch])
        next_states = np.array([e[3] for e in minibatch])
       # terminal = [terminal[i] for i in indices]

        target = self.model.predict(states)
        target_next = self.target_model.predict(next_states)
        print(target.shape)
        print('====')
        for i in range(self.batch_size):
            current_Q_value = target[i][0][actions[i]]

            if self.is_terminal(runner):
                target[i][0][actions[i]] = rewards[i]  # Terminal state, no future rewards
            else:
                # Non-terminal state, update Q-value using the classic rule
                best_next_Q_value = np.amax(target_next[i])  # max Q-value for the next state
                target[i][0][actions[i]] = (
                        current_Q_value +
                        self.lr * (
                                rewards[i] + self.discount_factor * best_next_Q_value - current_Q_value
                        )
                )

        history = self.model.train_on_batch(states, target, return_dict=True)
        self.model_history.append(history['loss'])

        if self.epsilon > self.epsilon_min:
            epsilon = get_config_param("dqn_agent")['train_params']["epsilon"]
            k = 0.01
            t = runner.episode
            self.epsilon = self.epsilon_min + (epsilon - self.epsilon_min) * math.exp(-k * t)


        self.update_target_counter += 1
        if self.update_target_counter % self.target_update_frequency == 0:
            self.update_target_model()
        print('>!>!>@>@!#>@!>#!@#>@! END oF REPLAY')

#=========================================================================================================
# == AGENT ROUTINES


class Runner:
    '''
    class for encapsulating reinforcement loop
    '''

    def __init__(self, experiment_name, condition_strategy=None, episode = None):
        self.city_name = experiment_name.split(',')[0].split('_')[-1]
        self.simulation_type = 'basic' if not 'force' in experiment_name else 'reinforcement'  # !@
        if episode:
            with open(episode) as json_data_file:
                ConfigData = json.load(json_data_file)
            self.episode = ConfigData['episode']

        else:
            self.episode = 0
        self.cum_reward_graph = []
        self.model_history = []
        self.strategy = condition_strategy
        self.cur_day = None
        self.incubation_day_counter = 0
        self.cur_action = None
        self.history_df = pd.DataFrame()
        self.init_infect = None
        self.cum_reward = None
        get_log_history()
        self.runner_chosen_action = None  # stuff
        self.history_actions_counter = {}
        self.history_actions_reward = {}
        self.history_memory_access = None

    def get_compliance(self, experiment_name):
        # TODO: not sure that i use this
        comp_pattern = r'comp=([\d.]+)'
        ci_delay_pattern = r',ci_delay=(\d+)'
        hi_delay_pattern = r',hi_delay=(\d+)'

        # Find matches using regular expressions
        comp_match = re.search(comp_pattern, experiment_name)
        ci_delay_match = re.search(ci_delay_pattern, experiment_name)
        hi_delay_match = re.search(hi_delay_pattern, experiment_name)

        # Extract values
        self.compliance = float(comp_match.group(1)) if comp_match else None
        self.ci_delay = float(ci_delay_match.group(1)) if ci_delay_match else None
        self.hi_delay = float(hi_delay_match.group(1)) if hi_delay_match else None
        get_log_history()

    def population_init(self, job_obj, path, with_population_caching=True, verbosity=False):
        '''
            Re-using of one of main routines in demography initializing process
            :param job_obj:
            :param with_population_caching:
            :param verbosity:
            :return:
        '''
        seed.set_random_seed()
        config_path = os.path.join(path, "config.json")
        with open(config_path) as json_data_file:
            ConfigData = json.load(json_data_file)
            citiesDataPath = ConfigData['CitiesFilePath']
            paramsDataPath = ConfigData['ParamsFilePath']
            Extensionslst = ConfigData['ExtensionsNamelst']
        Params.load_from(os.path.join(path, paramsDataPath), override=True)
        for param, val in job_obj.params_to_change.items():
            Params.loader()[param] = val
        DiseaseState.init_infectiousness_list()

        citiesDataPath = citiesDataPath

        population_loader = PopulationLoader(
            citiesDataPath,
            added_description=Params.loader().description(),
            with_caching=with_population_caching,
            verbosity=verbosity
        )

        world = population_loader.get_world(city_name=job_obj.city_name,
                                            scale=job_obj.scale,
                                            is_smart=True)

        ExtensionType = None
        get_log_history()
        return world, Extensionslst

    def train_stop_condition(self, economic, runner, rt):
        '''
        return number of persons for several conditions
        (A, B, ...) - where:
            A is percent of infected
            B is percent of vaccinated
            C is percent of # TODO: add something
        :param economic:
        :param strategy:
        :return:
        '''
        if isinstance(self.strategy, dict):
            if self.strategy['days_bound'] > 0:
                if runner.cur_day < self.strategy['days_bound']:
                    return True
                else:
                    return False
            else:
                st = [attr for attr in dir(economic) if attr in list(self.strategy.keys())]

                out = []
                stats = []
                if len(st) > 0:
                    if isinstance(st[0], dict):
                        for i in st:
                            stats.append(sum(getattr(economic, i).values()))
                        stats.append(sum(economic.total_population.values()))
                    else:
                        for i in st:
                            stats.append(getattr(economic, i))
                        stats.append(economic.total_population)

                for s, p in zip(stats[:-1], self.strategy.values()):
                    out.append(s < stats[-1] * p)  # stats[-1] = total_population

                return np.all(out)
        else:
            return rt > self.strategy

    def memorize(self, agent, economic, action, is_init_period):
        to_memory = []
        stats = [economic.susceptible_population,
                 economic.vaccinated_population,
                 economic.infected_population,
                 economic.hospitalized_population,
                 economic.recovered_population,
                 economic.deceased_population,
                 economic.capable_population]

        # discretization
        num_bins = get_config_param(config_name="dqn_agent")['train_params']["discretization_bins_num"]
        bin_width = 1.0 / num_bins
        if num_bins > 1:
            stats = [int(s / bin_width) for s in stats]

        if isinstance(economic.susceptible_population, dict):
            for stat in stats:
                for a in economic.age_groups:
                    to_memory.append(stat[a])
        else:
            for stat in stats:
                to_memory.append(stat)

        to_memory.append(action) if action is not None else to_memory.append(-1)
        #to_memory.append(agent.terminal)
        agent.replay_buffer.memory.append(to_memory)

        return agent

    def calculate_mean_actions(self, actions, window_size):
        mode_actions = []
        mean_actions = []
        sliding_window = deque(maxlen=window_size)
        for action in actions:
            sliding_window.append(action)
            if len(sliding_window) == window_size:
                # mode_action = Counter(sliding_window).most_common(1)[0][0]
                # mode_actions.append(mode_action)
                mean_action = np.mean(sliding_window)
                mean_actions.append(mean_action)
        # return mode_actions
        return mean_actions

    def day_sim(self, sim, extensions):
        for ext in extensions:
            ext.start_of_day_processing()

        sim.simulate_day()
        # Call Extension function at the end of the day
        for ext in extensions:
            regular_immune_seq = ext.end_of_day_processing()
        get_log_history()
        return regular_immune_seq

    def get_trajectory(self, info=()):
        sim, agent, economic, runner, extensions, simple_job_obj = info
        self.action_hist = []
        self.susceptible_hist = []
        self.vaccinated_hist = []
        self.infected_hist = []
        self.recovered_hist = []
        self.death_hist = []
        self.capable_hist = []
        self.reward_hist = []
        # extensions = runner.extenstions
        agent.epsilon = 0  # always use model
        runner.cur_day = 1
        agent.input_shape = agent.model.input_shape
        agent.n_outputs = agent.model.output_shape[-1]
        while runner.train_stop_condition(economic, runner, sim.get_rt()[0]):
            vaccinated_seq = self.day_sim(sim, extensions)
            economic.get_population_statistic(sim, vaccinated_seq, 0, (runner.cur_day, agent.day_range),
                                              aged_flag=get_config_param("age_stats"))
            if runner.cur_day < agent.day_range:
                # print('@@@@@@@', runner.cur_day, economic.infected_population)
                runner.cur_action = 0  # some basic action - NO ACTIONS
                #                agent = runner.memorize(agent, economic, runner.cur_action)
                economic.chosen_action = runner.cur_action
                action = 0

            else:
                current_states = agent.get_current_states()
                action = agent.epsilon_greedy_policy(current_states)
                economic.chosen_action = runner.cur_action # TODO: is neceseral?

            self.susceptible_hist.append(economic.susceptible_population)
            self.vaccinated_hist.append(economic.vaccinated_population)
            self.infected_hist.append(economic.infected_population)
            self.recovered_hist.append(economic.recovered_population)
            self.death_hist.append(economic.deceased_population)
            self.capable_hist.append(economic.capable_population)

            agent = self.memorize(agent, economic, action)
            #agent = runner.memorize(agent, economic, runner.cur_action)
            agent.get_reward(economic, sim)

            self.reward_hist.append(agent.reward)
            #agent.execute_action(sim, action, economic, runner)
            simple_job_obj = agent.execute_action(self, runner.cur_action, economic, runner, simple_job_obj)
            runner.cur_day += 1
        get_log_history()
#        sim.save_stats(None, 'TEST', info=(runner))

def wandb_routine(flag=False, project_name="coderona", team_name='mine', log_params=None,
                  name=(None, None)):
    runs_name, runs_flag = name
    os.environ['WANDB_MODE'] = get_config_param("wandb")['mode']
    checkpoint_date = get_config_param("dqn_agent")["checkpoint_date"]
    if runs_name is not None:
        runs_name = runs_name.replace("\n", "")
        index = runs_name.find('_')
        if index != -1:  # Ensure underscore is found
            runs_name = runs_name[index + 1:]
    wandb_ = get_config_param("wandb")
    get_log_history()

    if flag:
        wandb.login(key=wandb_['api_key'])
        wandb.init(
            project=project_name,
            entity=team_name,
            config=wandb.config,
            name='template',
            reinit=True,
            mode=get_config_param("wandb")['mode'],
            #resume="allow" if checkpoint_date else None,
            #id = checkpoint_date.replace('-', '_') if checkpoint_date else None
        )

    if (not wandb.run or wandb.run.name == 'template' or wandb.run.name != runs_name) and not runs_flag:
        if wandb.run is not None and wandb.run.name != 'template':  # Finish the current run if it exists
            wandb.finish()

            wandb.init(
                project=project_name,
                entity=team_name,
                config=wandb.config,
                name=runs_name,
                reinit=True,
                #resume="allow" if checkpoint_date else None,
                #id=checkpoint_date.replace('-', '_')
            )
        if runs_name:
            wandb.run.name = f'{runs_name}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
            wandb.run.save()
#            checkpoint_path = wandb.restore(checkpoint_filename).name if wandb.restore(checkpoint_filename) else None

        runs_flag = True

    if log_params:
        economic, sim, agent, action, runner = log_params
        values = {}
        rt, r0, r_smoothed, r_estimated = sim.get_rt()
        #is_age = get_config_param("age_stats")
        for v in wandb_["log_vars"]:
            if v == 'rt':
                values[v] = rt
            elif v == "reward":
                print('~~~~~~~~~~ SAVE REWARD TO WANDB', agent.reward)
                values[v] = agent.reward
            elif v == "mean_reward":
                values[v] = agent.mean_reward
            elif v == "cumulative_reward":
                values[v] = runner.cum_reward
            elif v == 'action':
                values[v] = action
            elif v == "mean_action":
                values[v] = agent.mean_action[-1]
            elif v == 'epsilon':
                values[v] = agent.epsilon
            elif v == 'economic_index':
                values[v] = economic.economic_index
            elif v == "init_infected":
                values[v] = runner.init_infect
            elif v == 'susceptible_population':
                values[v] = economic.susceptible_population
            elif v == 'recovered_population':
                 values[v] = economic.recovered_population
            elif v == "capable_population":
                values[v] = economic.capable_population
            elif v == 'exposed_population':
                 values[v] = economic.exposed_population
            elif v == "infected_population":
                values[v] = economic.infected_population
            elif v == "hospitalized_population":
                values[v] = economic.hospitalized_population
            elif v == "deceased_population":
                values[v] = economic.deceased_population
            elif v == "loss_":
                if agent.model_history != []:
                    values[v] = agent.model_history[-1]
                else:
                    values[v] = 0
            elif v == "action_max_reward":
                values[v] = {key: max(values) if len(values) > 0 else 0 for key, values in runner.history_actions_reward.items()}
            elif v == 'action_select_counter':
                values[v] = runner.history_actions_counter
            elif v == "memory_access":
                values[v] = runner.history_memory_access
            elif v == 'r0':
                values[v] = r0
            elif v == "r_smoothed":
                values[v] = r_smoothed
            elif v == "r_estimated":
                values[v] = r_estimated
            elif v == 'compliance':
                values[v] = runner.compliance
            elif v == 'vaccinated_population':
                values[v] = economic.vaccinated_population
            elif v == "level_2.0":
                values[v] = 2.0
            elif v == "level_1.5":
                values[v] = 1.5
            elif v == "level_1.25":
                values[v] = 1.25
            elif v == "level_1.0":
                values[v] = 1.0
            elif v == 'periods':
                if runner.cur_day % agent.replay_buffer.day_range == 0:
                    values[v] = sum(economic.total_population.values())
                else:
                    values[v] = 0
            elif v == "level_2.0":
                values[v] = 2.0
            elif v == "level_1.5":
                values[v] = 1.5
            elif v == "level_1.25":
                values[v] = 1.25
            elif v == "level_1.0":
                values[v] = 1.0
            elif v == 'periods':
                if runner.cur_day % agent.day_range == 0:
                    values[v] = sum(economic.total_population.values())
                else:
                    values[v] = 0

        if isinstance(values[next(iter(values))], dict):
            to_log = {}

            flag = False
            if get_config_param('wandb')["summing_in"]['flag']:
                params = {}
                for p in get_config_param('wandb')["summing_in"]['stat_list']:
                    if isinstance(values[p], dict):
                        params[p] = 0
                    else:
                        params[p] = values[p]
                flag = True

            to_log = {}
            flag = True

            for logs_name, log_value in values.items():
                if logs_name == "action_max_reward":
                    if len(log_value)>0:
                        for k, v in log_value.items():
                            to_log[f'{k}_action_max_reward'] = v

                elif logs_name == 'action_select_counter':
                    if len(log_value)>0:
                        for k, v in log_value.items():
                            to_log[f'{k}_action_select_count'] = v
                elif isinstance(log_value, dict):
                    for age, value in log_value.items():
                        to_log[f'{logs_name}_{age}'] = value
                        if flag:
                            params[logs_name] += value
                else:
                    to_log[logs_name] = log_value

            # Adding the summed values to to_log if flag is True
            if flag:
                for key, total in params.items():
                    to_log[key] = total

            wandb.log(to_log)
            # TODO: make runner.history_df analysis for multiaged
        else:
            wandb.log(values)
            # TODO: fix or implement same for wandb
            ''' CODE for peaks and valleys findings
            alias = f"ep_{runner.episode}_day_{runner.cur_day}"
            temp_df = pd.DataFrame([values], index=[alias])
            print(temp_df.shape)
            runner.history_df = pd.concat([runner.history_df, temp_df]).drop_duplicates()
            
            save_to_file(runner.history_df, filename=f'{runner.city_name}.csv')
            '''
    return runs_flag


def save_to_file(data, filename='data.csv'):
    data.to_csv(filename, mode='a', header=not pd.io.common.file_exists(filename))


def get_log_history():
    frame = inspect.currentframe().f_back
    function_name = frame.f_code.co_name
    class_name = None
    module_name = frame.f_globals['__name__']

    # Traverse the outer frames to find the class context
    for outer_frame in inspect.getouterframes(frame):
        if 'self' in outer_frame.frame.f_locals:
            class_name = outer_frame.frame.f_locals['self'].__class__.__name__
            break

    log = logging.getLogger(module_name)
    if class_name:
        log.info(f"{class_name}.{function_name} starting")
    else:
        log.info(f"{function_name} starting")


def get_config_param(config_name="people_per_day"):
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = f'{parent_dir}/config.json'
    with open(config_path) as json_data_file:
        ConfigData = json.load(json_data_file)

    return ConfigData[config_name]
