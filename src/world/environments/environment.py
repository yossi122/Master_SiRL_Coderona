from random import randint

class EnvironmentalAttribute(object):
    """
    used for PartialEnvironmentIntervention, and represents some attribute environments have (for instance, last name).
    """
    __slots__ = ('name', 'num_possibilities', 'environment_name')

    def __init__(self, environment_name, name, num_possibilities):
        """
        :param environment_name: The name of the environment type that holds it (e.g. ‘househeold’)
        :param name: The name of the attribute (e.g. ‘last name’)
        :param num_possibilities: The number of possible values the attribute can accept (e.g. 2)
        only when an environment is queries for an attribute the actual value of that attribute is randomly chosen.
        """
        self.name = name
        self.num_possibilities = num_possibilities
        self.environment_name = environment_name

    def __call__(self, person):
        return person.get_environment(self.environment_name).get_attribute(self)

class Environment(object):
    """
    Represents a place where a group of people can interact and infect each other.
    """
    __slots__ = ('_id', '_attributes', '_full_name')
    num_total_environments = 0

    def __init__(self, full_name):
        """
        initialized the environment
        :param full_name: str that is the full name of the environment, e.g household or workplace
        """
        self._id = Environment.num_total_environments  # Used only for debugging purposes
        self._attributes = {}
        self._full_name = full_name
        Environment.num_total_environments += 1

    def __repr__(self):
        return str(self.__class__) + "(id=%d)" % self._id

    def sign_up_for_today(self, person, weight):
        """
        Needs to be implemented by all subclasses.
        the logic of the person arriving (or not) each day to the environment
        :param person: Person
        :param weight: float, between 0 to 1. The wight affects the probability of the person infecting / being infected
        """
        raise NotImplementedError()

    def propagate_infection(self, date):
        """
        Needs to be implemented for every subclass
        Simulate contacts within this environment for today
        Returns the new infection events
        :param date: the current date
        :return: list of Event objects
        """
        raise NotImplementedError()

    def get_attribute(self, attribute):
        """
        Returns the environment attribute value
        :param attribute: the name of the desired attribute
        :return: randomly selected value, from available values
        """
        if attribute.name not in self._attributes:
            self._attributes[attribute.name] = randint(0, attribute.num_possibilities - 1)
        return self._attributes[attribute.name]
