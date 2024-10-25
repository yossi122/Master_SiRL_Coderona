from copy import deepcopy
from collections import Counter
from datetime import timedelta
from enum import Enum
from functools import cmp_to_key
import os
import random as random
import logging
import statistics

from src.seir import seir_times
from src.seir.disease_state import DiseaseState
from src.simulation.event import DayEvent
from src.logs import Statistics, DayStatistics
from src.world import Person
from src.world.environments import InitialGroup, Household
from typing import Dict

from src.simulation._DQN import wandb_routine, get_log_history, get_config_param
import matplotlib.pyplot as plt
from src.logs.r0_data import calculate_r0_instantaneous
import math
import numpy as np
import json

log = logging.getLogger(__name__)


class ORDER(Enum):
    NONE = 0,
    ASCENDING = 1,
    DESCENDING = 2,

    def __str__(self):
        return self.name


class Simulation(object):
    """
    An object which runs a single simulation, holding a world,
    calling events and propagating infections throughout environments
    day by day.
    """
    __slots__ = (
        '_verbosity',
        '_world',
        '_date',
        '_initial_date',
        'interventions',
        '_events',
        'stats',
        'stop_early',
        'last_day_to_record_r',
        'num_r_days',
        'first_infectious_people',
        'initial_infection_doc',
        'num_days_to_run',
        '_extension_params',
        'outdir'
    )

    def __init__(self, world, initial_date, interventions=None, stop_early=None, verbosity=False,
                 outdir=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs'),
                 extension_params=None):
        """
        :param world: The World object that this simulation will run on
        :param initial_date: The starting date for the simulation
        :param interventions: A list of the interventions applied in this simulation
        :param stop_early: An object that represent a condition which,
        when holds, causes the simulation to stop prematurely.
        Currently only one type of early stop supported, meant to help compute
        R0, R1, ..., Rk efficiently - stopping when all people infected in the
        first k days have recovered.
        :param verbosity: Whether or not this simulation should print debug info
        :param outdir: The path of the directory output files
        should be written into
        """
        if extension_params is None:
            extension_params = {}
        if interventions is None:
            interventions = []
        self._verbosity = verbosity
        self._world = world
        self._date = initial_date
        self._initial_date = deepcopy(initial_date)
        # attributes relevant for computing R data
        self.stop_early = stop_early
        self.last_day_to_record_r = None
        self.num_r_days = None
        if self.stop_early is not None:
            name_stop, self.num_r_days = self.stop_early
            self.last_day_to_record_r = initial_date + timedelta(days=self.num_r_days)
            assert name_stop == "r", "Other premature stops are not yet supported"

        self.outdir = outdir

    def update_events(self, interventions, extension_params):
        self.interventions = interventions
        self._events = {}
        self.stats = Statistics(self.outdir, self._world)
        self._extension_params = extension_params

        # It's important that we sign people up before we init interventions!
        self._world.sign_all_people_up_to_environments()
        for intervention in interventions:
            self.stats.add_intervention(intervention)

        self.first_infectious_people = set()
        self.initial_infection_doc = None
        self.num_days_to_run = None

        # save all the events that create the interventions behavior on the simulation
        for inter in self.interventions:
            self.register_events(inter.generate_events(self._world))

    def simulate_day(self):
        """
        Simulate one day of the simulation. Does this in four steps:
        1. Apply or remove registered events
        (either applying intervention effects or
        advancing the disease states of people)
        2. register people who changed weights to their environments
        3. spread the infection throughout the environments
        4. register the changes to the Statistics object
        """
        if self._date in self._events:
            self._events[self._date].apply(self)
            del self._events[self._date]

        changed_population = [
            person for person in self._world.all_people() if person._changed
        ]

        for individual in changed_population:
            individual.register_to_daily_environments()

        for env in self._world.all_environments:
            self.register_events(env.propagate_infection(self._date))

        changed_population = [
            person for person in self._world.all_people() if person._changed
        ]

        if self._verbosity and self._date.weekday() == 6:
            log.info("------ day-{}: disease state ------------".format(self._date))
            log.info(Counter([person.get_disease_state() for person in self._world.all_people()]))
            log.info("------ Infected by environments ----------")
            log.info(Counter([person.get_infection_data().environment.name for person in self._world.all_people() if
                              person.get_disease_state().is_infected() and person.get_infection_data()]))

        daily_data = DayStatistics(
            self._date,
            changed_population
        )
        self.stats.add_daily_data(daily_data, self._world)
        for person in changed_population:
            person.save_state()

        if self.last_day_to_record_r is not None and self._date <= self.last_day_to_record_r:
            for person in changed_population:
                if person.is_infected:
                    self.first_infectious_people.add(person)
        self._date += timedelta(days=1)

    def register_event_on_day(self, event, date):
        """
        hook the given event to the given date, so in that day this event will happen.
        :param event: Event
        :param date: datetime Date
        """
        if date not in self._events:
            self._events[date] = DayEvent(date)
        self._events[date].hook(event)

    def register_events(self, event_list):
        """
        Add all the given events to their dates on the simulation.
        This applies only to DayEvents that need to be triggered on a specific date.
        :param event_list: list of Event objects
        """
        if not isinstance(event_list, list):
            event_list = [event_list]
        for event in event_list:
            assert isinstance(event, DayEvent), \
                'Unexpected event type: {}'.format(type(event))
            self.register_event_on_day(event, event._date)

    def infect_random_set(self, num_infected: int, infection_doc: str, per_to_immune=0.0, Immune_compliance: float = 1,
                          order: ORDER = ORDER.NONE, city_name=None, min_age=0, people_per_day=1):
        """
        Infect a uniformly random initial set,
        so that the disease can spread during the simulation.
        :param num_infected: int number of infected to make
        :param per_to_immune: percent from the total population that should get immuned
        :param infection_doc: str to doc the infection data
        (written to the inputs.txt file)
        param oredr: specify if we order asending to descending the ages oof the persons that will get immuned 
        :param city_name: the name of the city to infect
        (if left None, infects people from all around the World)
        :param min_age: int specify the min age from which we start to infect population
        if the value is 0 we infect all the population \
        :param: Immune_compliance float. Simulate the state in which we aske some percentage of the population
        to get immune but only some of them agreed
        """
        assert Immune_compliance >= 0, "Immune_compliance can not be negative"
        assert isinstance(num_infected, int)
        assert self.initial_infection_doc is None
        self.initial_infection_doc = infection_doc
        if per_to_immune is None:
            per_to_immune = 0.0
        if city_name is not None:
            population = [p for p in self._world.all_people() \
                          if (p.get_city_name() == city_name.lower())]
        else:
            population = [p for p in self._world.all_people()]

        # Doing best effort to infect and immune the people in our world
        # after talking to Noam we first infect the ones we can and immune the rest
        num_infected = min(num_infected, len(population))
        adults = [p for p in population if p.get_age() > min_age]
        tmp_num_immuned = int(round(len(adults) * per_to_immune * Immune_compliance))
        num_immuned = min(len(population) - num_infected, tmp_num_immuned)
        assert len(population) >= num_infected + num_immuned \
            , "Trying to immune:{} infect:{} people out of {}".format(num_immuned, num_infected, len(population))
        used_adults = 0
        used_persons = {}
        # First set the people that aren't immune to be infected
        while num_infected > 0:
            Selected_persons = random.sample(population, num_infected)
            for p in Selected_persons:
                if (p.get_id() not in used_persons) and (p.get_disease_state() == DiseaseState.SUSCEPTIBLE):
                    self.register_events(p.infect_and_get_events(self._date, InitialGroup.initial_group()))
                    num_infected = num_infected - 1
                    used_persons[p.get_id()] = p
                    if p.get_age() > 0:
                        used_adults += 1

        num_immuned = min(len(adults) - used_adults, num_immuned)
        if order == ORDER.ASCENDING:
            adults = sorted(adults, key=cmp_to_key(Person.person_comperator_ASCENDING))
        elif order == ORDER.DESCENDING:
            adults = sorted(adults, key=cmp_to_key(Person.person_comperator_DESCENDING))
        else:
            adults = random.sample(adults, len(adults))
        # Second set- immune persons that are above min_age and we are able to immune
        Immuned_until_now = 0
        while Immuned_until_now < num_immuned:  # we start to count from zero therefor we need one more person
            Selected_persons = adults[Immuned_until_now:num_immuned]
            delta_days = 0
            immuned_today = 0
            for p in Selected_persons:
                if (p.get_id() not in used_persons):
                    #print(p._disease_state)
                    ok, events = p.immune_and_get_events(start_date=self._date, delta_time=timedelta(days=delta_days))
                    self.register_events(events)
                    # print("immuning id:{} on {}".format(p.get_id(),self._date + timedelta(days =delta_days)))
                    used_persons[p.get_id()] = p
                    if ok:
                        immuned_today += 1
                        num_immuned = num_immuned - 1
                        Immuned_until_now += 1
                    if immuned_today == people_per_day:
                        delta_days += 1
                        immuned_today = 0

    def immune_households_infect_others(self, num_infected: int, infection_doc: str, per_to_immune=0.0,
                                        Immune_compliance: float = 1, Sort_order: ORDER = ORDER.NONE, city_name=None,
                                        min_age=0, people_per_day=0):
        """
        Immune some percentage of the households in the population and infectimg a given percentage of the population
        so that the disease can spread during the simulation.
        :param num_infected: int number of infected to make
        :param infection_doc: str to document the infection data
        (written to the inputs.txt file)
        :param city_name: the name of the city to infect
        (if left None, infects people from all around the World)
        :param min_age: specify the min age from which we start to infect population
        if the value is 0 we infect all the population 
        :per_to_immune: percentage of the population that we are going to immune by housholds
        :people_per_day: how much houses per day we should immune
        :param: Immune_compliance float. Simulate the state in which we aske some percentage of the population
        to get immune but only some of them agreed
        """
        if not isinstance(num_infected, int):
            num_infected = int(num_infected)
        assert isinstance(num_infected, int)
        assert self.initial_infection_doc is None
        self.initial_infection_doc = infection_doc
        if per_to_immune is None:
            per_to_immune = 0.0
        if city_name is not None:
            tmp_households = [h for h in self._world.get_all_city_households() if h._city == city_name]
        else:
            tmp_households = [h for h in self._world.get_all_city_households()]

        households = []
        adults_cnt = 0
        for h in tmp_households:
            cnt = len([p for p in h.get_people() if p.get_age() > min_age])
            if cnt > 0:
                households.append(h)
                adults_cnt += cnt
        if Sort_order == ORDER.NONE:
            households = random.sample(households, len(households))
        elif Sort_order == ORDER.ASCENDING:
            households = sorted(households, key=cmp_to_key(Household.house_comperator_ASCENDING))
        elif Sort_order == ORDER.DESCENDING:
            households = sorted(households, key=cmp_to_key(Household.house_comperator_DESCENDING))

        num_infected = min(self._world.num_people(), num_infected)
        # Immune only some percentage of adults, that agreed to be immuned
        cnt_people_to_immun = int(self._world.num_people() * per_to_immune * Immune_compliance)
        used_persons = {}
        household_index = 0
        days_delta = 0

        if num_infected > 0:

            UnsafePersons = [p for p in self._world.all_people() if p._disease_state.name == 'SUSCEPTIBLE']
            people_to_infect = random.sample(UnsafePersons, min(len(UnsafePersons), num_infected))
            for person in people_to_infect:
                # print("calling infect_and_get_events from immune_households_infect_others for id:{}".format(person.get_id()))
                #print(person._disease_state.name )
                self.register_events(person.infect_and_get_events(self._date, InitialGroup.initial_group()))
                used_persons[person.get_id()] = person
                # print("Infecting person id:{} on date:{}".format(person.get_id(),self._date))

        while (cnt_people_to_immun > 0) and (people_per_day > 0) and (household_index < len(households)):
            cnt_people_to_immun_today = people_per_day
            while (cnt_people_to_immun_today > 0) and (household_index < len(households)):
                persons_to_immune = [p for p in households[household_index].get_people() \
                                     if (p.get_age() >= min_age) and (p.get_id() not in used_persons)]
                if Sort_order == ORDER.NONE:
                    cnt_immune_in_house = 0
                    for i in range(min(len(persons_to_immune), cnt_people_to_immun_today)):
                        #print("immune_households_infect_others", persons_to_immune[i]._disease_state)
                        ok, events = persons_to_immune[i].immune_and_get_events(start_date=self._date,
                                                                                delta_time=timedelta(days=days_delta))
                        self.register_events(events)
                        # Dror
                        used_persons[persons_to_immune[i].get_id()] = persons_to_immune[i]
                        # print("Immune person id:{} date:{}".format(persons_to_immune[i].get_id(),self._date + timedelta(days=days_delta)))
                        if ok:
                            cnt_people_to_immun_today -= 1
                            cnt_people_to_immun -= 1
                        cnt_immune_in_house += 1
                    if cnt_immune_in_house == len(persons_to_immune):
                        household_index += 1
                elif Sort_order in [ORDER.ASCENDING, ORDER.DESCENDING]:
                    i = 0
                    cnt_immune_in_house = 0
                    while i < min(len(persons_to_immune), cnt_people_to_immun_today):
                        #print(persons_to_immune[i]._disease_state)
                        ok, events = persons_to_immune[i].immune_and_get_events(start_date=self._date,
                                                                                delta_time=timedelta(days=days_delta))
                        self.register_events(events)
                        used_persons[persons_to_immune[i].get_id()] = persons_to_immune[i]
                        # print("Immune person id:{} date:{}".format(persons_to_immune[i].get_id(),self._date + timedelta(days=days_delta)))
                        if ok:
                            cnt_people_to_immun_today -= 1
                            cnt_people_to_immun -= 1
                        cnt_immune_in_house += 1
                        for j in range(i + 1, min(len(persons_to_immune), cnt_people_to_immun_today)):
                            if (persons_to_immune[j] not in used_persons) and (
                                    (persons_to_immune[i].get_age() // 10) == (persons_to_immune[j].get_age() // 10)):
                                #print(persons_to_immune[j]._disease_state)
                                ok, events = persons_to_immune[j].immune_and_get_events(start_date=self._date,
                                                                                        delta_time=timedelta(
                                                                                            days=days_delta))
                                self.register_events(events)
                                used_persons[persons_to_immune[j].get_id()] = persons_to_immune[j]
                                # print("Immune person id:{} date:{}".format(persons_to_immune[i].get_id(),self._date + timedelta(days=days_delta)))
                                if ok:
                                    cnt_people_to_immun_today -= 1
                                    cnt_people_to_immun -= 1
                                cnt_immune_in_house += 1
                                i += 1
                        if cnt_immune_in_house == len(persons_to_immune):
                            household_index += 1
            days_delta += 1

    def first_people_are_done(self):
        """
        chacks whether the people infected on the first “num_r_days” days
        are infected. We use this in simulations in which we try to compute R.
        When these people recover, we stop the simulation.
        """
        if self.stop_early is None:
            return False
        return all((not person.is_infected) for person in self.first_infectious_people)

    def infect_chosen_set(self, infection_datas, infection_doc):
        """
        Infect a chosen and specific set of people, given to the function, and register the events.
        :param infection_datas: list of (id, date, seit_times) for each person to infect
        :param infection_doc: str to doc the infection for inputs file
        """
        assert self.initial_infection_doc is None
        self.initial_infection_doc = infection_doc
        for person_id, infection_date, seir_times in infection_datas:
            p = self._world.get_person_from_id(person_id)
            events = p.infect_and_get_events(infection_date, InitialGroup.initial_group(), seir_times=seir_times)
            p.get_infection_data().date = None  # To avoid being asked to plot this date, which is out of our range
            self.register_events(events)

        original_date = self._date
        for date in sorted(self._events.keys()):
            if date < original_date:
                self._date = date
                self._events[date].apply(self)
                del self._events[date]
        self._date = original_date

    def print_hh_statistics(self):
        all_households = [h for h in self._world.get_all_city_households()]

        elderly_with_children = 0
        more_than_2_parents = 0
        more_than_2_elderly = 0
        only_children = 0
        city_name = ""
        for h in all_households:
            children = [p for p in h.get_people() if p.get_age() <= 16]
            elderly = [p for p in h.get_people() if p.get_age() >= 60]
            parents = [p for p in h.get_people() if 60 > p.get_age() > 18]
            city_name = h._city.get_name()
            if len(elderly) > 0 and len(children) > 0:
                elderly_with_children += 1
            if len(parents) > 3:
                more_than_2_parents += 1
            if len(elderly) > 2:
                more_than_2_elderly += 1
            if len(elderly) == 0 and len(parents) == 0 and len(children) > 0:
                only_children += 1

        print(f"print_hh_statistics() \n\t"
              f"city={city_name}\n\t"
              f"all_households={len(all_households)}\n\t"
              f"elderly_with_children={elderly_with_children} ({elderly_with_children * 100 / len(all_households):.1f}%)\n\t"
              f"more_than_2_parents={more_than_2_parents} ({more_than_2_parents * 100 / len(all_households):.1f}%)\n\t"
              f"more_than_2_elderly={more_than_2_elderly} ({more_than_2_elderly * 100 / len(all_households):.1f}%)\n\t"
              f"only_children={only_children} ({only_children * 100 / len(all_households):.1f}%)\n"
              )

    def get_rt(self):
        population = self._world.all_people()
        cur_date = self._date
        rt = calculate_r0_instantaneous(population=population, today=cur_date)
        print(f'\t >> {cur_date} RAW RT {rt}')

        if self._date - self._initial_date > timedelta(6):
            return rt, 0, 0, 0  # TODO: r0, r_smoothed, r_estim - didn't calculate as useless
        else:
            return 0, 0, 0, 0

    def run_simulation(self, num_days, name, datas_to_plot=None, run_simulation=None,
                       extensionsList=None, info=()):
        runner, agent, economic, sim, extensions, simple_job_obj, outdir, ext_param = info
        # day = 0
        log.info('=' * 100)
        get_log_history()

        runner.cum_reward = 0
        runner.cur_day = 0
        runner.incubation_day_counter = 0
        #  runner.cur_action = -1  # Initialize with no action taken
        is_init_period = True
        action_type = 'eps'

        # == compose cur_state ==
        regular_immune_seq = runner.day_sim(self, extensions)
        economic.get_population_statistic(self, regular_immune_seq, 0, is_init_period,
                                          aged_flag=get_config_param("age_stats"))
        agent.terminal = 0
        agent = runner.memorize(agent, economic, runner.cur_action, is_init_period)

        cur_state = np.array(agent.replay_buffer.memory[-1])

        if action_type.lower() == 'eps':
            runner.history_memory_access = 0
            # runner.history_memory_access['MEMORY'] = False
        elif action_type.lower() == 'memory':
            runner.history_memory_access = 1
            # runner.history_memory_access['MEMORY'] = True

        shape = cur_state.shape

        cur_state = np.reshape(cur_state, (1, shape[0]))
        #cur_state = np.reshape(cur_state, (1, 1, cur_state.shape[-1]))

        runner.cur_day += 1
        print(runner.episode)

        while runner.cur_day < runner.strategy['days_bound']:
            print(runner.cur_day, '!~!' * 25)
            log.info('#' * 50)
            print('PREV A: ', runner.cur_action)
            print('&&', cur_state.shape)
            #print(f'prev acion is {runner.cur_action}', end='\t')

            if runner.cur_day < agent.replay_buffer.day_range:
                # The initial waiting period before the first action is taken
                #print(runner.cur_day, 'WAIT', runner.cur_action)
                pass
            else:
                # Check if it's time to change the action (only three times)
                if abs(runner.cur_day - agent.replay_buffer.day_range) % agent.replay_buffer.intervention_dur == 0:
                    is_init_period = False
                    print('???? intervention dur', cur_state.shape)
                    runner.cur_action, action_type = agent.act(cur_state)  # Agent chooses an action
                    print(runner.cur_day, 'NEW ACTION IS: ', runner.cur_action)
                    if runner.cur_action >= 0:
                        simple_job_obj = agent.execute_action(self, runner.cur_action, economic, runner, simple_job_obj)
                        sim.update_events(simple_job_obj.interventions, ext_param)
                        agent.seen_states[runner.cur_action] += 1
                        runner.history_actions_counter = {int(k): v for k, v in runner.history_actions_counter.items()}
                        runner.history_actions_counter[runner.cur_action] += 1

            print('NEXT A: ', runner.cur_action)
            # == next state ==
            economic.chosen_action = runner.cur_action # !!!!
            next_immune_seq = runner.day_sim(self, extensions)
            runner.cur_day += 1
            economic.get_population_statistic(self, next_immune_seq, 0, is_init_period,
                                              aged_flag=get_config_param("age_stats"))
            agent.terminal = 0
            agent = runner.memorize(agent, economic, runner.cur_action, is_init_period)
            next_state = np.array(agent.replay_buffer.memory[-1])

            next_state = np.reshape(next_state, [1, next_state.shape[0]])
            #next_state = np.reshape(next_state, (1, 1, next_state.shape[-1]))
            log.info(f'~~~~ Cur {cur_state.shape} vs Next {next_state.shape}')

            agent.get_reward(economic, sim)

            if runner.cur_action >= 0:
                agent.remember(cur_state, runner.cur_action, agent.reward, next_state)

            cur_state = next_state

            agent.replay(runner)
            wandb_routine(log_params=(economic, sim, agent, runner.cur_action, runner),
                          name=(runner.city_name, True))

            agent.update_act_dict(runner, economic, simple_job_obj)

            if action_type.lower() == 'eps':
                runner.history_memory_access = 0
                #runner.history_memory_access['MEMORY'] = False
            elif action_type.lower() == 'memory':
                runner.history_memory_access = 1
                #runner.history_memory_access['MEMORY'] = True

            log.info(
                f'EP:{runner.episode}\tday:{runner.cur_day}\treward:{agent.reward:.5f}\taction:{runner.cur_action}\n')

            log.info('#'*50)
        #agent.terminal.append(1)
        #agent.replay_buffer.terminal_memory.append(agent.terminal)

        return extensions, simple_job_obj

    def get_extensions(self, num_days, datas_to_plot,
                       name, extensionsList):
        assert self.num_days_to_run is None
        self.num_days_to_run = num_days
        if datas_to_plot is None:
            datas_to_plot = dict()
        log.info("Starting simulation :\n" + name)

        extensions = []
        if extensionsList != None:
            for ExtName in extensionsList:
                mod = __import__('src.extensions.' + ExtName, fromlist=[ExtName])
                ExtensionType = getattr(mod, ExtName)
                assert ExtensionType(self)
                extensions += [ExtensionType(self)]
        return datas_to_plot, extensions

    def save_stats(self, datas_to_plot, name, info=''):
        # TODO: move calculation of statistics inside of loop to get dynamics after each 15 day
        age_groups = []

        #print(age_groups)

        '''
        age_groups = (None,
                      (0, 20),
                      (21, 40),
                      (41, 60),
                      (61, 80),
                      (81, 99))
        '''
        save_to = self.stats._output_path
        if info != '':
            for i in datas_to_plot['amit_graph']:
                if i.age_group not in age_groups:
                    age_groups.append(i.age_group)
            if not os.path.exists(save_to):
                os.mkdir(save_to)
            runner, agent = info

            fn = (f"{save_to}/{runner.city_name}_model.weights.h5")
            agent.model.build(input_shape=agent.model.input_shape)
            agent.model.save(fn)

            with open(f"{save_to}/params.json", "w") as f:
                json.dump({"episode": runner.episode,
                                "epsilon": agent.epsilon,
                                "action_select_counter": runner.history_actions_counter
                           }, f)

        if name.lower() == 'test':
            runner = info
            for age in runner.age_groups:

                fig, ax = plt.subplots()
                ax.grid()
                ax.margins(0)

                ax.plot(runner.susceptible_hist, label='susceptible')
                ax.plot(runner.vaccinated_hist, label='vaccinated')
                ax.plot(runner.infected_hist, label='infected')
                ax.plot(runner.recovered_hist, label='recovered')
                ax.plot(runner.death_hist, label='deaths')
                ax.plot(runner.capable_hist, label='economy')
                ax.legend(loc='best')
                ax.set_ylim([-0.05, 1.05])
                plt.savefig(f'{save_to}/trajectory')

                x_new_start = 0
                for i in range(len(runner.action_hist) - 1):
                    current = runner.action_hist[i]
                    next = runner.action_hist[i + 1]
                    # print(current)

                    color_map = {
                        0: 'blue',
                        1: 'green',
                        2: 'yellow',
                        3: 'red',
                        4: 'orange',
                        5: 'purple',
                        6: 'pink',
                        7: 'brown',
                        8: 'gray',
                        9: 'black',
                        10: 'aqua'
                    }

                    # Get the color corresponding to 'current' value
                    color = color_map.get(current, 'default_color')

                    ax.axvspan(i, i + 1, facecolor=color, alpha=0.4)

                ax.legend()
                plt.savefig(f'{save_to}/actions_{age}')
                # plt.show()

                fig, ax = plt.subplots()
                ax.plot(runner.reward_hist)
                plt.savefig(f'{save_to}/reward_by_actions_{age}')
