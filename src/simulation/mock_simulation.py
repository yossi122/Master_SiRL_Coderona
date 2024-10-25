import os
from collections import namedtuple

from src.simulation.simulation import Simulation
from src.simulation.event import DiseaseStateChangeEffect
from src.seir import DiseaseState
from src.world import World


InfectionAndSeirData = namedtuple("InfectionAndSeirData", ["person_id", "infection_date", "seir_times"])


class InitialStateSimulation(Simulation):
    """
    This class defines a mock simulation, that is used to simulate an initial infection spread,
    before the "real" simulation begins. This implementation is a simple one, that only simulation the infection,
    and stops when there are enough symptomatic or after number of days.
    """
    def __init__(
            self,
            world,
            initial_date,
            interventions=None,
            verbosity=False,
            outdir=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
    ):
        for e in world.all_environments:
            e.clear()
        super(InitialStateSimulation, self).__init__(
            World(
                [p.make_eventless_copy() for p in world.all_people()],
                world.all_environments,
                world._generating_city_name,
                world._generating_scale
            ),
            initial_date,
            interventions=interventions,
            verbosity=verbosity,
            outdir=outdir
        )
        # structure - a dict from city to a dict of person to a list of pairs of disease states and change dates
        # {
        #   city:
        #       {
        #           person: InfectedPersonData
        #       }
        # }
        self._infected_people_seir_times_per_city = {}
        self._original_initial_date = initial_date

    class InfectedPersonData(object):
        """
        Tracks the infection data of each infected person in the city/ simulation.
        """
        __slots__ = ("symptomatic", "symptomatic_date", "infection_date", "states_and_dates")

        def __init__(self, symptomatic, symptomatic_date, infection_date, states_and_dates):
            self.symptomatic = symptomatic
            self.symptomatic_date = symptomatic_date
            self.infection_date = infection_date
            self.states_and_dates = states_and_dates

        def __repr__(self):
            return "InfectedPersonData(symp={}, symp_date={}, infection_date={}, states_and_dates={})".format(
                self.symptomatic, self.symptomatic_date, self.infection_date, self.states_and_dates
            )

    def get_initial_infections(self, num_symptomatic_in_each_city_dict, max_num_days=150):
        """
        The main loop of the simulation.
        runs for each day, until max_num_days or num_symptomatic_in_each_city_dict is satisfied.
        At the end, returns all the infected data.
        """
        enough_infected = False
        num_days = 0
        while not (enough_infected or num_days >= max_num_days):
            self.simulate_day()
            enough_infected = True
            for city, infected_people in self._infected_people_seir_times_per_city.items():
                num_symptomatic = 0
                for p, p_data in infected_people.items():
                    if p_data.symptomatic and p_data.symptomatic_date <= self._date:
                        num_symptomatic += 1
                if num_symptomatic < num_symptomatic_in_each_city_dict[city]:
                    enough_infected = False
                    break
            num_days += 1

        assert num_days < max_num_days, \
            "Not enough infections achieved to meet requirements in {} days".format(max_num_days)

        rv = []
        for city, infected_people in self._infected_people_seir_times_per_city.items():
            city_infections_at_initial_date = self.generate_single_city_initial_state(
                city, num_symptomatic_in_each_city_dict[city]
            )
            for person, p_data in city_infections_at_initial_date:
                rv.append(self.infection_and_seir_data_from_person_data(person, p_data))
        return rv

    def calibrate_state_change_times(self, person_data, target_date):
        date_offset = target_date - self._original_initial_date
        person_data.states_and_dates = [
            (c[0] - date_offset, c[1], c[2])
            for c in person_data.states_and_dates
        ]
        person_data.infection_date -= date_offset
        if person_data.symptomatic:
            person_data.symptomatic_date -= date_offset

    def infection_and_seir_data_from_person_data(self, person, person_data):
        """
        Get the infection data from the infected person, by the format that is defined by InfectionAndSeirData
        :param person: Person
        :param person_data: InfectedPersonData
        :return: InfectionAndSeirData
        """
        states_and_dates = person_data.states_and_dates
        assert all(states_and_dates[i][2] == states_and_dates[i + 1][1] for i in range(len(states_and_dates) - 1))
        seir_times = []
        for i, [d, _, new_state] in enumerate(states_and_dates):
            if i == len(states_and_dates) - 1:
                length = None
            else:
                assert new_state == states_and_dates[i + 1][1]
                length = states_and_dates[i + 1][0] - d
            seir_times.append((new_state, length))
        assert seir_times[-1][0].is_terminal()
        return InfectionAndSeirData(
            person_id=person.get_id(),
            infection_date=person_data.infection_date,
            seir_times=seir_times
        )

    def generate_single_city_initial_state(self, city, number_of_infections_to_take):
        """
        Gets a certain number of symptomatic people from the given infected city, for the real simulation.
        :param city: city name
        :param number_of_infections_to_take: int of desired symptomatic people
        :return: list of infected data
        """
        if number_of_infections_to_take == 0:
            return []
        assert city in self._infected_people_seir_times_per_city
        infected_people = list(self._infected_people_seir_times_per_city[city].items())
        infected_people.sort(key=lambda x: x[1].infection_date)
        symptomatic_people = sorted(
            [p for p in infected_people if p[1].symptomatic],
            key=lambda p: p[1].symptomatic_date
        )
        target_date = symptomatic_people[number_of_infections_to_take - 1][1].symptomatic_date
        all_infected_for_initial_state = [p for p in infected_people if p[1].infection_date <= target_date]

        for p in all_infected_for_initial_state:
            self.calibrate_state_change_times(p[1], target_date)
            p[1].states_and_dates.sort(key=lambda x: x[0])  # Sort by date
        return all_infected_for_initial_state

    def register_event_on_day(self, event, date):
        for e in [event, *event.hooks]:
            if isinstance(e.effect, DiseaseStateChangeEffect):
                person = e.effect.get_person()
                if person.get_disease_state() != DiseaseState.IMMUNE:
	                city = person.get_city_name()
	                if city not in self._infected_people_seir_times_per_city:
	                    self._infected_people_seir_times_per_city[city] = {}

	                person_data = self._infected_people_seir_times_per_city[city].setdefault(
	                    person,
	                    self.InfectedPersonData(
	                        symptomatic=False,
	                        symptomatic_date=None,
	                        infection_date=self._date,
	                        states_and_dates=[(self._date, None, DiseaseState.LATENT)]
	                    )
	                )
	                if e.effect.get_states()[1] == DiseaseState.SYMPTOMATICINFECTIOUS:
	                    person_data.symptomatic = True
	                    person_data.symptomatic_date = date
	                person_data.states_and_dates.append((date, *e.effect.get_states()))

        super(InitialStateSimulation, self).register_event_on_day(event, date)
