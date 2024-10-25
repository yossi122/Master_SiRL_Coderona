# This code creates the households of a city and divides them into
# neighborhoods. It is the smarter alternative to naive_population_generation.py
# The generation code itself lies in household_generation.py, this code
# validates the output and wraps it to the needed format,

from src.world.population_generation.household_generation import sim_houses

from src.world.city_data import City
from src.world import Person
from src.world.environments import Household, NeighborhoodCommunity
from src.simulation.params import Params
from src.util import divide_weighted_array
import math
import random

def get_distribution_distance(samples, distribution):
    """
    Computes and returns the distance between two distributions
    (one given as a distribution and the other as samples),
    defined as the angle between the vector they create (normalized to be between 0 and 1).
    :param samples: A series of samples which we want to test against a distribution
    :param distribution: A distribution
    :return: The angle between the distribution given by the samples and the given distribution
    """
    n = max(max(samples)+1, len(distribution))
    sample_distribution = [0 for i in range(n)]
    for s in samples:
        sample_distribution[s] += 1 / len(samples)
    # The formula is angle(x, y) = acos(<x,y>/(|x|*|y|)). We normalize
    prod = sum([x*y for (x, y) in zip (sample_distribution, distribution)])
    norms = (sum([x**2 for x in sample_distribution])**0.5, sum([x**2 for x in distribution])**0.5)
    return math.acos(prod / (norms[0] * norms[1])) * 2 / math.pi, sample_distribution

def generate_all_households_and_communities_of_city_smart(city, scaling=1.0, verbosity=False):
    """
    Generates all of the households of some city and divides them into neighborhoods.
    :param city: A City object corresponding to where we want to generate a population.
    :param scaling: A factor to multiply the city.population by (in order to simulate toy examples)
    :param verbosity: If 'True', outputs some data about the distance
    between the requested and achieved distributions.
    :return: A tuple of 3 lists:
    (a list of all generated people (Person objects),
     a list of all generated neighborhoods (lists of Person objects),
     a list of all generate environments (NeighborhoodCommunity objects))
    """
    assert isinstance(city, City)
    all_people = []
    all_environments = []
    params = Params.loader()['population']
    if verbosity:
        print("Generating city '%s'" % city.english_name)
    # Calling the actual code that generates houses
    households_ages = sim_houses(
        city.age_data,
        city.household_size_data,
        city.percentage_of_households_with_65_plus,
        city.percentage_of_households_with_17_minus,
        int(city.total_households * scaling),
        int(city.population * scaling),
        city.percentage_of_households_with_single_parent,
        city.percentage_of_children_with_single_parent_in_household,
        city.kids_per_household_data
    )
    # Converting houses from "lists of ages" to "lists of Person objects"
    households = [list(map(Person, ha)) for ha in households_ages]
    # The following is here because the algorithm generates them in a very ordered way,
    # and we don't want it to cause skews
    # (this is redundant since the addition of divide_weighted_array, but better safe than sorry)
    random.shuffle(households)
    # Computing the distance between the requested and achieved age distribution
    all_ages_groups = []
    for household in households:
        for person in household:
            all_ages_groups.append(min(person.get_age() // 5, 15))
    age_distribution_dist, sampled_age_distribution = get_distribution_distance(all_ages_groups, [t[0] for t in city.age_data])
    if age_distribution_dist > 0.15:
        print("WARNING probabilities far from original in city '%s'" % city.english_name)
    # Computing the distance between the requested and achieved household size distribution
    all_household_sizes = list(map(lambda l: min(len(l), len(city.household_size_data) - 1), households_ages))
    household_size_distribution_dist, sampled_household_size_distribution = get_distribution_distance(
        all_household_sizes, city.household_size_data)
    if household_size_distribution_dist > 0.15:
        print("WARNING probabilities far from original in city '%s'" % city.english_name)
    num_houses_got = len(households_ages)
    num_people_got = len(all_ages_groups)
    if verbosity:
        print("Was looking for age distribution of")
        print([t[0] for t in city.age_data])
        print("And got")
        print(sampled_age_distribution)
        print("Age distribution distance of", age_distribution_dist)
        print("Was looking for")
        print(city.household_size_data)
        print("And got")
        print(sampled_household_size_distribution)
        print("Household size distribution distance of", household_size_distribution_dist)
        print("Relative error in number of houses of", num_houses_got / int(city.total_households * scaling) - 1)
        print("Was looking for", int(city.total_households * scaling))
        print("And got", num_houses_got)
        print("Relative error in number of people of", num_people_got / int(city.population * scaling) - 1)
        print("Was looking for", int(city.population * scaling))
        print("And got", num_people_got)
        print("----------------------------------------------------")

    # Divide households to neighborhoods
    all_neighborhood_households = divide_weighted_array(
        [(household, len(household)) for household in households],
        params['community_approx_size']
    )

    # Transfer the format to the requested output format
    all_neighborhoods = []
    for curr_neighborhood_households in all_neighborhood_households:
        curr_neighborhood = []
        for curr_household in curr_neighborhood_households:
            curr_household_env = Household(city, params['household_avg_daily_contacts'] / max((len(curr_household) - 1, 1)))
            all_environments.append(curr_household_env)
            for person in curr_household:
                all_people.append(person)
                curr_neighborhood.append(person)
                person.add_environment(curr_household_env)
        all_neighborhoods.append(curr_neighborhood)
        curr_community_env = NeighborhoodCommunity(
            city,
            params['community_avg_daily_contacts'] / float(max(len(curr_neighborhood), 1))
        )
        all_environments.append(curr_community_env)
        for person in curr_neighborhood:
            person.add_environment(curr_community_env)
    assert sum([len(neighborhood) for neighborhood in all_neighborhoods]) == len(all_people), \
        "Internal error in population generation"
    return all_people, all_neighborhoods, all_environments
