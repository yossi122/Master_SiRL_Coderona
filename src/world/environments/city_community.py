from collections import defaultdict
import random

from src.world.environments.homogeneous_environment import HomogeneousEnvironment


class CityCommunity(HomogeneousEnvironment):
    __slots__ = ('_city', '_sub_environments_dict', '_already_sorted_envs')
    name = "city_community"

    def __init__(self, city, contact_prob_between_each_two_people):
        super(CityCommunity, self).__init__(contact_prob_between_each_two_people)
        self._city = city
        self._sub_environments_dict = defaultdict(list)
        self._already_sorted_envs = set()

    def add_environment(self, env):
        assert not (env.name in self._already_sorted_envs), "Can't add an anvironment after sorting them!"
        self._sub_environments_dict[env.name].append(env)

    def get_sorted_environments(self, env_name):
        assert env_name in self._sub_environments_dict, "Don't have environments of name '%s'" % env_name
        if env_name in self._already_sorted_envs:
            return self._sub_environments_dict[env_name]
        random.shuffle(self._sub_environments_dict[env_name])
        self._already_sorted_envs.add(env_name)
        # Tuple since we don't want anyone changing it
        return tuple(self._sub_environments_dict[env_name])
