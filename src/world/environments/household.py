from functools import cmp_to_key
from src.world.environments.homogeneous_environment import HomogeneousEnvironment
from src.world.person import Person

class Household(HomogeneousEnvironment):
    __slots__ = ('_city', '_city_env')
    name = "household"

    def __init__(self, city, contact_prob_between_each_two_people):
        super(Household, self).__init__(contact_prob_between_each_two_people)
        self._city = city
        self._city_env = None

    def set_city_env(self, city_env):
        self._city_env = city_env
    
    @classmethod
    def house_comperator_ASCENDING(cls, a,b):
        """
        Compare households by their youngest person
        """
        #Get the persons we whish to sort 
        a_people = a.get_people()
        b_people = b.get_people()
        #Sort the persons by their age in ascending way 
        a_people = sorted(a_people,key = cmp_to_key(Person.person_comperator_ASCENDING))
        b_people = sorted(b_people,key = cmp_to_key(Person.person_comperator_ASCENDING))
        #compare the youngest persons of each house
        return a_people[0].get_age() - b_people[0].get_age()

    @classmethod
    def house_comperator_DESCENDING(cls,a,b):
        """
        Compare households by their youngest person
        """
        #Get the persons we whish to sort 
        a_people = a.get_people()
        b_people = b.get_people()
        #Sort the persons by their age in ascending way 
        a_people = sorted(a_people,key = cmp_to_key(Person.person_comperator_DESCENDING))
        b_people = sorted(b_people,key = cmp_to_key(Person.person_comperator_DESCENDING))
        #compare the youngest persons of each house
        return b_people[0].get_age() - a_people[0].get_age()
