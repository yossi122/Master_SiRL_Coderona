# This code creates the households of a city and divides them into
# neighborhoods. It is naive, a far better implementation exists in
# smart_population_generation.py

import warnings
from src.world.city_data import City
from src.world import Person
from src.world.environments import Household, NeighborhoodCommunity
from src.simulation.params import Params

def generate_household_in_city(city):
    """
    Generates a househeold from a given city (which holds relevant distributions)
    :param city: A City object corresponding to where we want to generate a household.
    :return: A list of Person objects corresponding to a random household in that city.
    """
    assert isinstance(city, City)
    household_size = city.household_size_distribution.sample()
    return [Person(city.age_distribution.sample()) for i in range(household_size)]

def generate_all_households_and_communities_of_city_naive(city, scaling=1.0):
    """
    Generates all households of some city and divides them into neighborhoods.
    :param city: A City object corresponding to where we want to generate a population.
    :param scaling: A factor to multiply the city.population by (in order to simulate toy examples)
    :return: A tuple of 3 lists:
    (a list of all generated people (Person objects),
     a list of all generated neighborhoods (lists of Person objects),
     a list of all generate environments (NeighborhoodCommunity objects))
    """
    #Amit confirmed that we should always use this class
    #warnings.warn("DEPRECATED CODE! naive city generation should not be used!", DeprecationWarning)
    assert isinstance(city, City)
    curr_community = []
    all_people = []
    all_communities = []
    all_environments = []
    min_num_people = int(city.population * scaling)
    params = Params.loader()['population']
    while len(all_people) < min_num_people:
        curr_household = generate_household_in_city(city)
        curr_household_env = Household(city, params['household_avg_daily_contacts'] / (max(len(curr_household) - 1, 1)))
        all_environments.append(curr_household_env)
        for person in curr_household:
            all_people.append(person)
            person.add_environment(curr_household_env)
            curr_community.append(person)
        if (len(curr_community) >= params['community_approx_size']) or (len(all_people) >= min_num_people):
            all_communities.append(curr_community)
            curr_community_env = NeighborhoodCommunity(
                city,
                params['community_avg_daily_contacts'] / float(max(len(curr_community), 1))
            )
            all_environments.append(curr_community_env)
            for person in curr_community:
                person.add_environment(curr_community_env)
            curr_community = []
    assert len(curr_community) == 0
    return all_people, all_communities, all_environments
