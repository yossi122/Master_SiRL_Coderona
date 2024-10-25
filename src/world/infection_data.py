class InfectionData(object):
    """
    Hold the data relevant to the infection of some person.
    We use it to track the infection source and the environment where each person got infected.
    """
    __slots__ = ("date", "environment", "transmitter", "infected")

    def __init__(self, infected, date, environment, transmitter):
        """
        Save the infection data
        :param infected: Person, the one that got infected
        :param date: datetime Date object that states the date of the infection
        :param environment: Environment where the person got infected
        :param transmitter: Person who infected
        """
        self.date = date
        self.environment = environment
        self.transmitter = transmitter
        self.infected = infected

    def __eq__(self, other):
        if other is None:
            return False
        return self.date == other.date and self.environment == other.environment and self.transmitter == other.transmitter and self.infected == other.infected

    def get_stats(self):
        """
        Get more detailed dict with data about the infection, like age and names.
        :return: dict
        """
        # env, transmitter age, transmitter disease state
        if self.transmitter is None:
            return {
                "infection_env_short": self.environment.name,
                "infection_env_long": self.environment._full_name,
                "infected_age": self.infected.get_state().age,
                "infector_age": None,
                "infector_disease_state": None,
                "disease_state": self.infected.get_disease_state()
            }
        return {
            "infection_env_short": self.environment.name,
            "infection_env_long": self.environment._full_name,
            "infected_age": self.infected.get_state().age,
            "infector_age": self.transmitter.get_state().age,
            "infector_disease_state": self.transmitter.get_state().disease_state,
            "disease_state": self.infected.get_disease_state()
        }

    @staticmethod
    def get_keys():
        """
        :return: get the keys of the detailed infection data dict
        """
        return ("infection_env_short", "infected_age", "infector_age", "infector_disease_state")
