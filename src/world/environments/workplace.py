from src.world.environments.homogeneous_environment import HomogeneousEnvironment


class Workplace(HomogeneousEnvironment):
    __slots__ = ('_city', '_age_segment', '_city_env')
    name = "workplace"

    def __init__(self, city, contact_prob_between_each_two_people, age_segment, full_name=None):
        super(Workplace, self).__init__(contact_prob_between_each_two_people, full_name=full_name)
        self._city = city
        self._city_env = None
        self._age_segment = age_segment

    def set_city_env(self, city_env):
        self._city_env = city_env
