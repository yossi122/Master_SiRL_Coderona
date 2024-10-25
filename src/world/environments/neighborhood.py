from src.world.environments.homogeneous_environment import HomogeneousEnvironment


class NeighborhoodCommunity(HomogeneousEnvironment):
    '''
    Represt neighborhood ini a city which contains a  group of houses
    _id = Each NeighborhoodCommunity has it's own id number which is a specific identifier.
    '''
    __slots__ = ('_city', '_city_env','_hood_id')
    name = "neighborhood_community"
    num_total_hoods = 0

    def __init__(self, city, contact_prob_between_each_two_people):
        super(NeighborhoodCommunity, self).__init__(contact_prob_between_each_two_people)
        self._city = city
        self._city_env = None
        self._hood_id = NeighborhoodCommunity.num_total_hoods
        NeighborhoodCommunity.num_total_hoods +=1
        
    def get_neighborhood_id(self):
        return self._hood_id

    def set_city_env(self, city_env):
        self._city_env = city_env
