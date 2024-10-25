import random
from math import exp

from src.world.environments.environment import Environment
from src.util import Distribution


class HomogeneousEnvironment(Environment):
    """
    A group of people in the simulation.
    People arrive each day to their environments and each two people in the same environment can infect each other.
    Most interventions changes the probability that a person will go to the environment.
    contact_prob_between_each_two_people = the probability that two persons will meet in the environment
    """

    __slots__ = (
        '_person_dict',
        '_infectious_people_and_weights',
        '_contact_prob_between_each_two_people'
    )

    def __init__(self, contact_prob_between_each_two_people : float, full_name=None):
        if full_name is None:
            full_name = self.name
        super(HomogeneousEnvironment, self).__init__(full_name)
        self._person_dict = {}
        self._infectious_people_and_weights = {}
        self._contact_prob_between_each_two_people = \
            contact_prob_between_each_two_people

    def sign_up_for_today(self, person, weight):
        """
        Change the amount of time that a person will stay in the environment
        This amount (weight) is based to 1 and may change with interventions.
        If the weight is zero, this person won't go to this rnviroment until a further change.
        """
        if person.is_dead:
            self._person_dict.pop(person, None)
            self._infectious_people_and_weights.pop(person, None)
            return

        if person.is_infectious:
            total_weight = person.get_prob_to_infect_on_contact() * weight
            self._infectious_people_and_weights[person] = total_weight
        else:
            self._infectious_people_and_weights.pop(person, None)

        self._person_dict[person] = weight

    def clear(self):
        """
        Kick all the peole out of the environment
        """
        self._person_dict = {}
        self._infectious_people_and_weights = {}

    def propagate_infection(self, date):
        """
        Simulate contacts within this environment for today,
        Return the new infection events.
        The model: the number of times person i and person j meet
        has a Poisson distribution with mean w_i * w_j,
        so 'the number of times the infection passes from i to j' is Poisson(contact_prob * w_i * w_j * inf_i),
        so the probability the infection doesn't pass from i to j is exp(-contact_prob*w_i*w_j*inf_i),
        hence the probability for person j to not get infected is exp(-sum_{i}{contact_prob * w_i * w_j * inf_i})
        which is exp(-sum_{i}{contact_prob * w_i * inf_i})**w_j. We define the 'weightess non infection prob'
        as exp(-sum_{i}{contact_prob * w_i * inf_i}).
        See the specification document for more details.
        """
        if len(self._infectious_people_and_weights) == 0:
            return []

        total_infected_weights = sum(self._infectious_people_and_weights.values(), 0.0)

        log_weightless_non_infection_prob = \
            - self._contact_prob_between_each_two_people * total_infected_weights
        weightless_non_infection_prob = exp(log_weightless_non_infection_prob)

        infection_source_distribution = Distribution(
            list(self._infectious_people_and_weights.keys()),
            list(self._infectious_people_and_weights.values()),
            check_sum=False,
            is_singleton=True
        )
        new_events = []
        num_infections = 0
        for person, weight in self._person_dict.items():
            if person.is_susceptible:
                infection_prob = 1 - (weightless_non_infection_prob ** weight)
                if random.random() < infection_prob:
                    num_infections += 1
                    infection_source = infection_source_distribution.sample()
                    # print("propagate_infection calling infect_and_get_events for id:{} date:{}".format(person.get_id(),date))
                    curr_events = person.infect_and_get_events(date, self, infection_source)
                    new_events += curr_events

        if num_infections > 0:
            for p, weight in self._infectious_people_and_weights.items():
                p._num_infections += weight * num_infections / total_infected_weights

        return new_events

    def get_people(self):
        """
        Returns an iterator of the people that come to the environment
        """
        return self._person_dict.keys()
