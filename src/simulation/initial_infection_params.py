from src.simulation.mock_simulation import InitialStateSimulation
from src.simulation.simulation import ORDER
from enum import Enum


class InitialInfectionParams(object):
    """
    Base abstract class that defines the behavior of infecting an initial set of people at the start of the simulation.
    Each subclass has to implement a way to infect an initial set in a simulation object.
    """

    def __init__(self):
        pass

    def infect_simulation(self, sim, outdir):
        raise NotImplementedError()


class InitialImmuneType(Enum):
    """
    The initial immune can come from 2 sources:
    1. Immuning certain percentage of the population randomally
    2. Immuning certain percentage of households randomally and infect random people from the rest of the population.
    """
    GENERAL_POPULATION = 1
    HOUSEHOLDS = 2
    HOUSEHOLDS_ALL_AT_ONCE = 3
    BY_NEIGHBORHOOD = 4

    def __str__(self):
        return self.name


class NaiveInitialInfectionParams(InitialInfectionParams):
    """
    Infect num_to_infect random people in city_name_to_infect
    or in the entire country if city_name_to_infect is not given
    """

    def __init__(self, num_to_infect, per_to_Immune, Immune_compliance, order, city_name_to_infect, immune_source,
                 min_age=0, people_per_day=0):
        super(NaiveInitialInfectionParams, self).__init__()
        self.num_to_infect = num_to_infect
        self.city_name_to_infect = city_name_to_infect.lower()
        self.per_to_Immune = per_to_Immune
        self.immune_source = immune_source
        self.min_age = min_age
        self.people_per_day = people_per_day
        self.Immune_compliance = Immune_compliance
        self.order = order

    def infect_simulation(self, sim, outdir):
        # TODO to skip the immune code here and only use the one in the extension
        per_to_Immune = 0 # self.per_to_Immune
        if self.immune_source == InitialImmuneType.GENERAL_POPULATION:
            sim.infect_random_set(self.num_to_infect, str(self), per_to_Immune, self.Immune_compliance, self.order, \
                                  self.city_name_to_infect, self.min_age, people_per_day=self.people_per_day)
        elif self.immune_source in [InitialImmuneType.HOUSEHOLDS, InitialImmuneType.HOUSEHOLDS_ALL_AT_ONCE, InitialImmuneType.BY_NEIGHBORHOOD]:
            sim.immune_households_infect_others(self.num_to_infect, str(self), per_to_Immune,
                                                self.Immune_compliance, self.order, \
                                                self.city_name_to_infect, self.min_age,
                                                people_per_day=self.people_per_day)
        else:
            print(f"infect_simulation() ERROR!! {self.immune_source} is not handled!!!")
            exit(-100)

    def __str__(self):
        if self.city_name_to_infect is None:
            return "{}(num_to_infect={})".format(self.__class__.__name__, self.num_to_infect)
        return "{}(num_to_infect={}, per_to_Immune={}, city_name_to_infect={}, per_to_Immune={},immune_source={})".format(
            self.__class__.__name__, self.num_to_infect, self.per_to_Immune, self.city_name_to_infect, self.per_to_Immune,
            self.immune_source
        )


class SmartInitialInfectionParams(InitialInfectionParams):
    """
    This subclass defines a "smart" way to infect - runs a "mock" simulation until the requested amount of 'symptomatic or post-symptomatic' are reached.

    initial_symptomatic_num_in_each_city is a dict from a city to the number of 'symptomatic or post-symptomatic' people that city should have.
    The simulation will infect initial_random_set_for_mock and then wait until the appropriate numbers infected in each city.
    It will 'copy' the disease states in each city corresponding to that city's
    point in time (when the 'symptomatic or post-symptomatic' number was right).
    """

    def __init__(self, initial_symptomatic_num_in_each_city, initial_random_set_for_mock):
        super(SmartInitialInfectionParams, self).__init__()
        self.initial_symptomatic_num_in_each_city = initial_symptomatic_num_in_each_city
        self.initial_random_set_for_mock = initial_random_set_for_mock

    def infect_simulation(self, sim, outdir):
        mock_sim = InitialStateSimulation(
            sim._world,
            sim._initial_date,
            None,
            verbosity=False,
            outdir=outdir
        )
        mock_sim.infect_random_set(self.initial_random_set_for_mock, "N/A", None)
        initial_infections = mock_sim.get_initial_infections(
            {city: self.initial_symptomatic_num_in_each_city for city in sim._world.get_all_city_names()}
        )
        for e in sim._world.all_environments:
            e.clear()

        sim._world.sign_all_people_up_to_environments()
        sim.infect_chosen_set(initial_infections, str(self))

    def __str__(self):
        return "{}(initial_symptomatic_num_in_each_city={}, initial_random_set_for_mock={})".format(
            self.__class__.__name__, self.initial_symptomatic_num_in_each_city, self.initial_random_set_for_mock
        )
