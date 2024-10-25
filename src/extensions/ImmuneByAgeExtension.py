import math
import random
from datetime import timedelta, date
from functional import seq
from src.seir import DiseaseState
from src.simulation.initial_infection_params import InitialImmuneType
from src.simulation.simulation import Simulation
from src.world import Person
from src.world.environments.neighborhood import NeighborhoodCommunity
from src.w import W
from src.simulation._DQN import get_config_param

import logging
log = logging.getLogger(__name__)

class NeighborhoodStatistics:
    def __init__(self):
        self._look_back_days = 7
        self._sick_recently = {}

    def push(self, neighborhood_id: int, item: int) -> None:
        if neighborhood_id not in self._sick_recently:
            self._sick_recently[neighborhood_id] = seq([])

        self._sick_recently[neighborhood_id] = seq(item) + self._sick_recently[neighborhood_id]
        if self._sick_recently[neighborhood_id].len() > self._look_back_days:
            self._sick_recently[neighborhood_id] = self._sick_recently[neighborhood_id].take(self._look_back_days)

    def average(self, neighborhood_id: int) -> int:
        return self._sick_recently[neighborhood_id].average()


class ImmuneStrategy:
    NONE: int = 0,
    ASCENDING: int = 1,
    DESCENDING: int = 2,

    def __init__(self, order: int, immune_by_households: bool, immune_everybody_in_the_house: bool,
                 immune_by_neighborhood: bool):
        """
        if self.immune_by_households then if immune_everybody_in_the_house is True as well,
        then if vaccinating one person at home, we will vaccinate ALL the household
        """
        self.order = order
        self.immune_by_households = immune_by_households
        self.immune_everybody_in_the_house = immune_everybody_in_the_house
        self.immune_by_neighborhood = immune_by_neighborhood

    def get_order(self):
        return self.order

    def get_description(self, days_since_start: int = None):
        by_household = "Random"
        order = self.order
        immune_by_neighborhood_overide = self.immune_by_neighborhood and days_since_start is not None and days_since_start <= 7
        if immune_by_neighborhood_overide:
            order = self.DESCENDING
        if self.immune_by_neighborhood:
            if not immune_by_neighborhood_overide:
                by_household = "By Neighborhood"
        elif self.immune_by_households:
            if self.immune_everybody_in_the_house:
                by_household = "By Household, All or nothing"
            else:
                by_household = "By Household"
        if order == self.ASCENDING:
            return "ASCENDING" + ", " + by_household
        elif order == self.DESCENDING:
            return "DESCENDING" + ", " + by_household
        else:
            return "ANY_AGE" + ", " + by_household

    def __str__(self):
        return self.get_description()


class ImmuneByAgeExtension(Simulation):
    def __init__(self, parent: Simulation):
        # super().__init__(world, initial_date)
        self.finished = False
        extension_parameters = parent._extension_params["ImmuneByAgeExtension"]
        # print(f"ImmuneByAgeExtension() extension_parameters {str(extension_parameters)}", flush=False)
        # change the following parameters to affect the vaccination flow
        self.days_since_start = 0
        # self.target_immune_percentage = extension_parameters.per_to_Immune
        self.target_immune_percentage = [
            0,  # AGE 0-9
            0.7,  # AGE 10-19
            0.73,  # AGE 20-29
            0.79,  # AGE 30-39
            0.82,  # AGE 40-49
            0.86,  # AGE 50-59
            0.88,  # AGE 60-69
            0.94,  # AGE 70-79
            0.92,  # AGE 80-89
            0.91  # AGE 90-99
        ]

        self.min_age_to_immune = extension_parameters.min_age
        self.max_age_to_immune = get_config_param("linked_immune_age")[0]["max_age"]
        self.max_people_to_immune_a_day = extension_parameters.people_per_day
        self.carry_over_vaccinations = 0
        self.historical_neighborhood_data = NeighborhoodStatistics()
        self.completed_neighborhood = []
        self.immune_strategy: ImmuneStrategy = ImmuneStrategy(
            order=extension_parameters.order.value,
            immune_by_households=extension_parameters.immune_source in
                                 [InitialImmuneType.HOUSEHOLDS, InitialImmuneType.HOUSEHOLDS_ALL_AT_ONCE],
            immune_everybody_in_the_house=extension_parameters.immune_source == InitialImmuneType.HOUSEHOLDS_ALL_AT_ONCE,
            immune_by_neighborhood=extension_parameters.immune_source == InitialImmuneType.BY_NEIGHBORHOOD)

        if extension_parameters.per_to_Immune is not None:
            self.target_immune_percentage = [extension_parameters.per_to_Immune] * len(self.target_immune_percentage)

        # internal state. do not change!
        self.parent = parent
        if self.immune_strategy.immune_by_neighborhood:
            self.state_max_age_to_immune = math.floor((self.max_age_to_immune + 5) / 10) * 10
            self.state_min_age_to_immune = self.state_max_age_to_immune - 10
        else:
            if self.get_order() == ImmuneStrategy.DESCENDING:
                self.state_max_age_to_immune = math.floor((self.max_age_to_immune + 5) / 10) * 10
                self.state_min_age_to_immune = self.state_max_age_to_immune - 10
            elif self.get_order() == ImmuneStrategy.ASCENDING:
                self.state_min_age_to_immune = self.min_age_to_immune
                self.state_max_age_to_immune = math.floor((self.min_age_to_immune + 15) / 10) * 10
            else:
                self.state_min_age_to_immune = self.min_age_to_immune
                self.state_max_age_to_immune = self.max_age_to_immune

    def __str__(self):
        return f"ImmuneByAgeExtension() immune_percentage={self.target_immune_percentage}\n" + \
               f"min_age={self.min_age_to_immune}, max_age={self.max_age_to_immune}\n" + \
               f"self.max_people_to_immune_a_day={self.max_people_to_immune_a_day}\n" + \
               f"carry_over_vaccinations_from_yesterday={self.carry_over_vaccinations}\n" + \
               f"strategy={str(self.immune_strategy.get_description(self.days_since_start))}"

    def is_by_households(self):
        return self.immune_strategy.immune_by_households

    def is_by_neighborhood(self):
        # in the first Week of a neighborhood Immune strategy, we would like to vaccinate the elderly people
        # then, in the next weeks, continue with the neighborhood Immune strategy
        return self.immune_strategy.immune_by_neighborhood and self.days_since_start > 7

    def get_order(self):
        if self.immune_strategy.immune_by_neighborhood and self.days_since_start <= 7:
            return ImmuneStrategy.DESCENDING
        else:
            return self.immune_strategy.get_order()

    def is_immune_everybody_in_the_house(self):
        return self.immune_strategy.immune_everybody_in_the_house

    def _register_events(self, person: Person):
        ok, events = person.immune_and_get_events(start_date=self.parent._date, delta_time=timedelta(days=0))
        if ok:
            self.parent.register_events(events)

    def can_immune(self, state: DiseaseState) -> bool:
        if state in (DiseaseState.LATENT,
                     DiseaseState.INCUBATINGPOSTLATENT,
                     DiseaseState.ASYMPTOMATICINFECTIOUS,
                     DiseaseState.SUSCEPTIBLE):
            return True
        else:
            return False

    def start_of_day_processing(self):
        pass

    def count_sick_people(self, people):
        c = seq(people).count(lambda p: p.get_disease_state() in [DiseaseState.SYMPTOMATICINFECTIOUS])
        return c

    def get_age_category(self, p: Person):
        return p.get_age_category()

    def end_of_day_processing(self):
        """
        if portion of population that is immune > self.ImmunePortion Do nothing
        else immune by the selected ImmuneStrategy
        """
        self.days_since_start += 1
        immuned = seq(self.parent._world.all_people()).count(
            lambda p: p.get_disease_state() == DiseaseState.IMMUNE and
                      (self.min_age_to_immune < p.get_age_category() <= self.max_age_to_immune)
        )
        population = seq(self.parent._world.all_people()).count(
            lambda p: (p.get_disease_state() != DiseaseState.DECEASED) and
                      (self.min_age_to_immune < p.get_age_category() <= self.max_age_to_immune)
        )
        immuned_percentage = (immuned / population) if population > 0 else 1.0

        people_vaccinated_today = []

        target_percentage_index = max(0, int(self.state_min_age_to_immune / 10))
        target_percentage_index = min(len(self.target_immune_percentage) - 1, target_percentage_index)
        target_percentage = self.target_immune_percentage[target_percentage_index]

        # print(f"Immune: self.state_min_age_to_immune={self.state_min_age_to_immune}, "
        #       f"target={target_percentage}, "
        #       f"immuned_percentage={immuned_percentage}", flush=False)
        if not self.finished:
            if self.is_by_neighborhood():
                self.state_min_age_to_immune = self.min_age_to_immune
                self.state_max_age_to_immune = self.max_age_to_immune

                neighborhoods = seq(self.parent._world.all_environments).filter(
                    lambda e: isinstance(e,
                                         NeighborhoodCommunity) and e.get_neighborhood_id() not in self.completed_neighborhood)

                if neighborhoods.len() == 0:
                    print(f"No more neighborhoods to vaccinate! (day {self.days_since_start})", flush=False)
                    self.finished = True
                    return

                r_per_neighborhood = neighborhoods.map(
                    lambda neighborhood: self.calculate_r0_instantaneous_per_neighborhood(self.parent._date,
                                                                                          neighborhood, 7))

                index_of_max = r_per_neighborhood.to_list().index(r_per_neighborhood.max())

                # print(f"Immune neighborhood {index_of_max}, neighborhood_id={neighborhoods[index_of_max].get_neighborhood_id()}, sick_per_neighborhood={all_people_to_consider}...", flush=False)
                # print(f"Immune neighborhood {index_of_max}, neighborhood_id={neighborhoods[index_of_max].get_neighborhood_id()}, r_per_neighborhood={r_per_neighborhood}...", flush=False)

                people_vaccinated_today = [p for p in neighborhoods[index_of_max].get_people()
                                           if (self.can_immune(p.get_disease_state())) and
                                           (p.get_age_category() > self.min_age_to_immune) and
                                           (p.get_age_category() <= self.max_age_to_immune)]
                people_vaccinated_today.sort(key=self.get_age_category,
                                             reverse=self.get_order() == ImmuneStrategy.DESCENDING)
                # are we done with this neighborhood?
                if len(people_vaccinated_today) < self.max_people_to_immune_a_day + self.carry_over_vaccinations:
                    # print(f"completed neighborhood {neighborhoods[index_of_max].get_neighborhood_id()}", flush=False)
                    self.completed_neighborhood += [neighborhoods[index_of_max].get_neighborhood_id()]

            elif self.is_by_households():
                households = [h for h in self.parent._world.get_all_city_households()]
                for h in households:
                    if self.is_immune_everybody_in_the_house():
                        # if one person can be vaccinated in this household, vaccinate ALL the eligible people
                        if seq(h.get_people()).count(
                                lambda p: self.can_immune(p.get_disease_state()) and
                                          (p.get_age_category() > self.state_min_age_to_immune) and
                                          (p.get_age_category() <= self.state_max_age_to_immune)) > 0:
                            people_vaccinated_today += [p for p in h.get_people()
                                                        if (self.can_immune(p.get_disease_state())) and
                                                        (p.get_age_category() >= 18) and
                                                        (p.get_age_category() <= self.state_max_age_to_immune)]
                    else:
                        people_vaccinated_today += [p for p in h.get_people()
                                                    if (self.can_immune(p.get_disease_state())) and
                                                    (p.get_age_category() > self.state_min_age_to_immune) and
                                                    (p.get_age_category() <= self.state_max_age_to_immune)]
                    # do we have enough people to vaccinate today?
                    if len(people_vaccinated_today) > self.max_people_to_immune_a_day + self.carry_over_vaccinations:
                        break
            else:
                people_vaccinated_today = [p for p in self.parent._world.all_people()
                                           if (self.can_immune(p.get_disease_state())) and
                                           (p.get_age_category() > self.state_min_age_to_immune) and
                                           (p.get_age_category() <= self.state_max_age_to_immune)]
            count_to_percentage = math.floor(max(0, target_percentage * population - immuned))
            print(f"ImmuneByAgeExtension({str(self.immune_strategy.get_description(self.days_since_start))}, (day {self.days_since_start}), " +
                  f"age {self.state_min_age_to_immune}-{self.state_max_age_to_immune}, immune {self.max_people_to_immune_a_day} a day, " +
                  f"carry_over_vaccinations={self.carry_over_vaccinations}, " +
                  f"immune % = {immuned}/{population} = {immuned_percentage * 100.0:.1f} count_to_percentage={count_to_percentage}",
                  flush=False)
            num_can_be_vaccinated_today = min(count_to_percentage,
                                              self.max_people_to_immune_a_day + self.carry_over_vaccinations)

            log.info(f"VACCINATION EXTENSION == Day {self.days_since_start}: num_can_be_vaccinated_today={num_can_be_vaccinated_today}, "
                  f"people_vaccinated_today.len()={len(people_vaccinated_today)}")

            people_vaccinated_today = seq(people_vaccinated_today).take(num_can_be_vaccinated_today)
            people_vaccinated_today.for_each(lambda person: self._register_events(person))

            # advance to the next age group only if you covered the current age group
            if people_vaccinated_today.len() < (self.max_people_to_immune_a_day + self.carry_over_vaccinations) or \
                    immuned_percentage >= target_percentage:
                log.info(f"VACCINATION EXTENSION == VacToday {people_vaccinated_today.len()} < {(self.max_people_to_immune_a_day + self.carry_over_vaccinations)}")
                if self.is_by_neighborhood():
                    pass
                elif self.get_order() == ImmuneStrategy.DESCENDING:
                    if self.state_min_age_to_immune <= self.min_age_to_immune:
                        self.finished = True
                    self.state_min_age_to_immune = max(self.state_min_age_to_immune - 10, self.min_age_to_immune)
                    self.state_max_age_to_immune = self.state_min_age_to_immune + 10
                elif self.get_order() == ImmuneStrategy.ASCENDING:
                    if self.state_max_age_to_immune >= self.max_age_to_immune:
                        self.finished = True
                    self.state_max_age_to_immune = min(self.state_max_age_to_immune + 10, self.max_age_to_immune)
                    self.state_min_age_to_immune = self.state_max_age_to_immune - 10
                elif self.get_order() == ImmuneStrategy.NONE:
                    pass  # do nothing
                else:
                    assert False, "immune_strategy is Out Of Range!"

            self.carry_over_vaccinations = (self.max_people_to_immune_a_day + self.carry_over_vaccinations -
                                           people_vaccinated_today.len())
            if self.carry_over_vaccinations < 0:
                print(
                    f"!!!!!!!!!!!! self.carry_over_vaccinations={self.carry_over_vaccinations} is NEGATIVE !!!!!!!!!!")

            # else:
            #     print(f"IMMUNE NOTHING people_to_immune={len(people_to_immune)} while self.max_people_to_immune_a_day={self.max_people_to_immune_a_day}\n"
            #           f"immuned_percentage={immuned_percentage}, target_percentage={target_percentage}", flush=False)
            return people_vaccinated_today
    def calculate_r0_instantaneous_per_neighborhood(self, today: date,
                                                    neighborhood: NeighborhoodCommunity,
                                                    look_back_days: int = 7) -> float:
        """calculates the instantaneous r based on a look back of look_back_days days"""

        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7325187/
        w = W()
        look_back_days = w.w_len()

        infected_today = self.parent.stats.get_neiborhood_data(today - timedelta(days=1),
                                                               neighborhood.get_neighborhood_id())
        weighted_infected_last_week = max(seq([w.get_w(i) *
                                               self.parent.stats.get_neiborhood_data(today - timedelta(days=i + 1),
                                                                                     neighborhood.get_neighborhood_id())
                                               for i in range(1, look_back_days)]).sum(), 1)

        if infected_today == 0 or weighted_infected_last_week == 0:
            return 0

        instantaneous_reproductive = infected_today / weighted_infected_last_week

        return instantaneous_reproductive
