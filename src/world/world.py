
class World(object):
    """
    The World class holds all the people and their environments for the simulation.
    """
    __slots__ = ('_people_dict', 'all_environments', '_city_name_to_env', '_generating_city_name', '_generating_scale')

    def __init__(self, all_people, all_environments, generating_city_name, generating_scale):
        """
        Init World object with all population.
        :param all_people: list of all the people
        :param all_environments: list of all the environments
        :param generating_city_name: the city name (str) that is generated in the simulation of this world
        :param generating_scale: (float) from 0 to 1,
        the scale of the given city, i.e 0.1 generates a similar city which is smaller by a factor of 10,
        and all the environments shrinks as well.
        """
        self._people_dict = {p.get_id(): p for p in all_people}
        self.all_environments = all_environments
        self._init_city_name_to_env_dict()
        self._generating_city_name = generating_city_name.lower()
        self._generating_scale = generating_scale

    def _init_city_name_to_env_dict(self):
        """
        Save all the city environments by the city name
        """
        self._city_name_to_env = {}
        for env in self.all_environments:
            if env.name == "city_community":
                name = env._city.english_name.lower()
                assert name not in self._city_name_to_env, "Got the city '%s' multiple times!" % name
                self._city_name_to_env[name] = env

    def get_city_community(self, city_name):
        """
        :param city_name: str city name
        :return: the city environment
        """
        lowered_city_name = city_name.lower()
        assert lowered_city_name in self._city_name_to_env, "No city named '%s'" % city_name
        return self._city_name_to_env[lowered_city_name]

    def sign_all_people_up_to_environments(self):
        """
        For all the people in the world, register to their environments.
        The registration happens according to the person's routine
        This is needed for the infection to spread.
        """
        for person in self.all_people():
            person.register_to_daily_environments()

    def get_all_city_communities(self):
        """
        return all the city environments in the world
        :return: list of CityEnvironment
        """
        return list(self._city_name_to_env.values())
    
    def get_all_city_households(self):
        households  = [ p for p in self.all_environments if p.name == 'household']
        return households

    def get_all_city_names(self):
        """
        get all the cities' names in this world
        :return: list of str
        """
        return list(self._city_name_to_env.keys())

    def num_people(self):
        """
        number (int) of people in the world
        """
        return len(self._people_dict)

    def all_people(self):
        """
        return a list of all persons in the world
        :return: list of Person
        """
        return list(self._people_dict.values())

    def get_person_from_id(self, person_id):
        """
        return person that has the given id
        :param person_id: int id
        :return: Person by id
        """
        return self._people_dict[person_id]
