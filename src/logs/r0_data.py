from datetime import timedelta, date
import numpy as np
import pickle
import os
import sys
import time
from collections import namedtuple
from functional import seq

from typing import List
from src.world import Person
from src.world.infection_data import InfectionData
from src.w import W

# a debug flag to save the population to be able to reload it and call calculate_r0_data directly
debug_save_population_to_file = True


def save_population_to_file(population: List[Person]) -> None:
    filename = f'population-{time.strftime("%Y%m%d-%H%M%S")}.pkl'
    if not os.path.exists(filename):
        with open(filename, 'wb') as f:
            recursion_limit = sys.getrecursionlimit()
            print(f"save_population_to_file() saving to {filename}... (recursion_limit={recursion_limit})")
            sys.setrecursionlimit(40000)
            pickle.dump(filename, f)
            sys.setrecursionlimit(recursion_limit)
            print(f"save_population_to_file() saved {filename} DONE")


def load_population_to_file() -> List[Person]:
    with open('population.pkl', 'r') as f:
        population = pickle.load(f)
        return population


def calculate_r0_instantaneous(population: List[Person], today: date, look_back_days: int = 7) -> float:
    """calculates the instantaneous r based on a look back of look_back_days days"""
    infected_per_day_dict = {}

    def infected_recently(p: Person, today: date, look_back_days: int = 7):
        infection_data = p.get_infection_data()
        return infection_data is not None and \
               infection_data.date is not None and \
               (today >= infection_data.date >= today - timedelta(days=look_back_days))

    def add_to_dict(x):
        infected_per_day_dict[x[0]] = len(x[1])

    infected_per_day = seq(population).filter(lambda p: infected_recently(p, today, look_back_days)).group_by(
        lambda p: p.get_infection_data().date)
    infected_per_day.for_each(add_to_dict)

    if infected_per_day.len() == 0:
        return 0

    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7325187/
    w = W()
    look_back_days = w.w_len()

    instantaneous_reproductive = infected_per_day_dict.get(today, 0) \
                                 / max(seq([w.get_w(i) *
                                        infected_per_day_dict.get(today-timedelta(days=i), 0)
                                for i in range(1, look_back_days)]).sum(), 1)

    return instantaneous_reproductive


def calculate_r0_data(population: List[Person], max_date=None, min_age=0, max_age=100):
    """
    Calculates daily avg r data.
    people can infect others only if their age is between min_age to max_age. but they can infect anyone
    Returns a dictionary that contains an array of dates, an array of avg r values,
    and an array of smoothed avg values.
    If no valid infections were made, returns None
    population: List[Person]
    """

    '''
    class R0Statistics:
        got_infected_today: int = 0
        infected_by_ones_who_got_infected_today: int = 0

    p = seq(population)
    min_infection_date = p.filter(p.get_infection_data() is not None and p.get_infection_data().date is not None) \
        .map(p.get_infection_data().date).min()
    max_infection_date = p.filter(p.get_infection_data() is not None and p.get_infection_data().date is not None) \
        .map(p.get_infection_data().date).max()
    r0_by_infection_date_pre_division = {
        min_infection_date + timedelta(days=i): R0Statistics()
        for i in range((max_infection_date - min_infection_date).days + 1)
    }

    def sum_sick_per_day(person: Person):
        r0_by_infection_date_pre_division[person.get_infection_data().date].got_infected_today += 1
        r0_by_infection_date_pre_division[person.get_infection_data().date].infected_by_ones_who_got_infected_today += \
            1 if p.get_infection_data().transmitter is not None else 0

    p.for_each(sum_sick_per_day)
    seq(r0_by_infection_date_pre_division.items())
    '''

    # number_of_people_before = len(population)
    # population = seq(population).filter(lambda p:
    #                         (p.get_infection_data() is None or p.get_infection_data().transmitter is None) or
    #                         (min_age <= p.get_infection_data().transmitter.get_age() <= max_age)
    #                        ).to_list()
    # print(f"calculate_r0_data() filtered the list from {number_of_people_before} to {len(population)} (-{number_of_people_before-len(population)}) people for {min_age}<ages<{max_age} ")

    # rs is dictionary for each person how many he has infected every day
    # a dictionary where the key is a Person and the
    # value is the [infection date, number-of-people-that-this-person-infected]
    # note: rs is calculated even for dates after "max_date"
    RS = namedtuple('RS', 'infection_date people_infected_by_me')
    # rs = {p: [None, 0] for p in population}
    rs = {p: RS(None, 0) for p in population}
    # rs_original = {p: [None, 0] for p in population}
    for p in population:
        infection_data: InfectionData = p.get_infection_data()
        if (infection_data is not None) and (infection_data.date is not None):
            # rs_original[p][0] = infection_data.date
            rs[p] = RS(infection_data.date, rs[p].people_infected_by_me)
            if infection_data.transmitter is not None:
                # rs_original[infection_data.transmitter][1] += 1
                rs[infection_data.transmitter] = \
                    RS(rs[infection_data.transmitter].infection_date,
                       rs[infection_data.transmitter].people_infected_by_me + 1)
                # if rs_original[infection_data.transmitter][1] != rs[infection_data.transmitter].people_infected_by_me:
                #    print("ERRORRRRRRRR people_infected_by_me!")
                # if rs_original[infection_data.transmitter][0] is not None and rs_original[infection_data.transmitter][0] != rs[infection_data.transmitter].infection_date:
                #    print("ERROR!!!!!DATE !")

    valid_infections = any(p.get_infection_data().date is not None for p in population
                           if (p.get_infection_data() is not None)
                           )
    if not valid_infections:
        return None

    # The last date someone got infected
    max_infection_date = max(p.get_infection_data().date for p in population
                             if (p.get_infection_data() is not None) and
                             (p.get_infection_data().date is not None)
                             )
    # The first date someone got infected
    min_infection_date = min(p.get_infection_data().date for p in population
                             if (p.get_infection_data() is not None) and
                             (p.get_infection_data().date is not None)
                             )

    # Generating dictionary date:num of infectors, num of infected, how much every person contributed to the infection every day
    InfectionByDate = namedtuple('InfectionByDate',
                                 'infected_today infected_by_someone_who_got_infected_today smoothed_infected_by_someone_who_got_infected_today')
    # r0_by_infection_date_pre_division_original = {
    #     min_infection_date + timedelta(days=i): [0, 0, 0]
    #     for i in range((max_infection_date - min_infection_date).days + 1)
    # }
    r0_by_infection_date_pre_division = {
        # for every date [number-of-people-got-infected-today,
        #                 total-number-of-people-infected-by-someone-who-got-sick-today]
        min_infection_date + timedelta(days=i): InfectionByDate(0, 0, 0)
        # min_infection_date + timedelta(days=i): [0, 0, 0]
        for i in range((max_infection_date - min_infection_date).days + 1)
    }

    # TODO: r0_by_infection_date_pre_division[date] may get dates out of the range:
    #          min_infection_date..max_infection_date
    # TODO: also, the "infections" may count people who got infected After the max_infection_date
    # TODO: Ask what's the difference between: r0_by_infection_date_pre_division[date][1]
    #       and r0_by_infection_date_pre_division[date][2] ?
    # ???? TODO should it be "for data, infected_by in rs.values():" ????
    for infected_by in rs.values():
        if infected_by.infection_date is not None:
            # the date is the date where the person got sick
            # r0_by_infection_date_pre_division_original[infected_by.infection_date][0] += 1  # how many people who got-infected today
            # r0_by_infection_date_pre_division_original[infected_by.infection_date][1] += infected_by.people_infected_by_me  # sum of all of the people who were
            i = r0_by_infection_date_pre_division[infected_by.infection_date]
            r0_by_infection_date_pre_division[infected_by.infection_date] = InfectionByDate(
                i.infected_today + 1,
                i.infected_by_someone_who_got_infected_today + infected_by.people_infected_by_me,
                0
            )
            # if r0_by_infection_date_pre_division_original[infected_by.infection_date][0] != r0_by_infection_date_pre_division[infected_by.infection_date].infected_today:
            #    print("ERRORRRR infected_today")
            # if r0_by_infection_date_pre_division_original[infected_by.infection_date][1] != r0_by_infection_date_pre_division[infected_by.infection_date].infected_by_someone_who_got_infected_today:
            #     print("ERRORRRR infected_by_someone_who_got_infected_today")

            # infected by a person who got sick today

    for p in population:
        if (p.get_infection_data() is not None) and (p.get_infection_data().date is not None):
            # r0_by_infection_date_pre_division_original[p.get_infection_data().date][2] += p._num_infections
            i = r0_by_infection_date_pre_division[p.get_infection_data().date]
            r0_by_infection_date_pre_division[p.get_infection_data().date] = InfectionByDate(
                i.infected_today,
                i.infected_by_someone_who_got_infected_today,
                i.smoothed_infected_by_someone_who_got_infected_today + p._num_infections)
            # if r0_by_infection_date_pre_division_original[p.get_infection_data().date][2] != r0_by_infection_date_pre_division[p.get_infection_data().date].smoothed_infected_by_someone_who_got_infected_today:
            #     print("ERROR!!!! smoothed_infected_by_someone_who_got_infected_today")

    '''
    Incubating post Latent:
    - p=0.1  w=0.1/sum
    - 0.1
    - 0.1
    Symptomatic post Incubating:
    - 0.3
    - 0.3
    - 0.3
    - 0.3
    Immune post Symptomatic:
    
    find w according to params.json
    
    Gamma(duration of Incubating post Latent) --> sample 1000 times --> 2
    Gamma(duration of Symptomatic post Incubating) --> sample 1000 times --> 4
    
    p[1..2] = probability of Infectiousness during Incubating post Latent
    p[3..6] = probability of Infectiousness during Symptomatic post Incubating
    
    w[i] = p[i]/sum(p)
    '''

    max_date = max_infection_date if not max_date else max_date
    r0_by_infection_date = [
        (date, (
            # total_infected / num_infecting if num_infecting > 0 else np.NaN,
            # smoothed_num_infected / num_infecting if num_infecting > 0 else np.NaN
            infections.infected_by_someone_who_got_infected_today / infections.infected_today
            if infections.infected_today > 0 else np.NaN,
            infections.smoothed_infected_by_someone_who_got_infected_today / infections.infected_today
            if infections.infected_today > 0 else np.NaN
        ))
        # TODO: seems like the first item is counting the "infected" and not the "infecting" right?
        # for date, [num_infecting, total_infected, smoothed_num_infected]
        for date, infections
        in r0_by_infection_date_pre_division.items()
        if date <= max_date
    ]

    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7325187/
    w = W()
    look_back_days = w.w_len()
    infected_per_day = np.array([
        infections.infected_today
        for date, infections
        in r0_by_infection_date_pre_division.items()
        if date <= max_date
    ])
    instantaneous_reproductive = [
        infected_per_day[i] / max(seq([w.get_w(j) * infected_per_day[i - j] for j in range(1, look_back_days)]).sum(),
                                  1)
        for i in range(look_back_days, len(infected_per_day))
    ]
    instantaneous_reproductive = [0.0] * look_back_days + instantaneous_reproductive

    dates = [c[0] for c in r0_by_infection_date]
    smoothed_avg_r0 = [c[1][1] for c in r0_by_infection_date]
    avg_r0 = [c[1][0] for c in r0_by_infection_date]
    population_size = len(population)
    # R-Effective = R0 * (suceptible / population-size)
    # estimated_r0 = [(r / (susceptibles.get(d) / population_size)) for r, d in zip(avg_r0, dates)]
    return {'dates': dates, 'smoothed_avg_r0': smoothed_avg_r0, 'avg_r0': avg_r0, 'estimated_r0': [],
            'instantaneous_r': instantaneous_reproductive}
