# We are given the following data:
# * The nafa that each city belongs to
# * The percentage of people in each city which work inside their own city
# * The percentage of people from each nafa that work in any other nafa
# We create, from this data, the distribution of workplace city
# for people living in each origin city, in the following way:
# Say we want to compute the probability that a person from city X
# will work in city Y.
# * If X=Y, this is given.
# * If nafa(X)!=nafa(Y), we assume that the person works in nafa(Y)
#   with the given (nafa-wide) distribution, and inside each city in nafa(Y)
#   with probability proportional to the city population.
# * If nafa(x)=nafa(Y), we assume that the same holds for
#   all cities in nafa(X) except for X (i.e. the ratio of probabilities for
#   Y1,Y2 in nafa(X) is the same as the ratio of the populations of Y1, Y2).
#
# One additional detail - since we are throwing out cities that are too small
# after rescaling, we might end up with weird situations.
# If a nafa completely empties out, we throw it out from all distributions
# and normalize. If a nafa is left with one city, then the data of
# 'how many people live work within the city' and
# 'how many people live and work within the nafa' refer to the same thing -
# so in that case we only use the latter.

from src.world.city_data.city import City
from src.util import DiscreteDistribution

def make_city_work_destination_distributions(city_list):
    """
    Define, for each city in city_list, its workplace_city_distribution.
    See the documentation in the beginning of the file for details.
    :param city_list: The list of cities which are taken in the simulation.
    :return: void, but all the city.workplace_nafa_distribution-s are updated.
    """
    # Compute the list of cities in each nafa
    nafa_to_cities = {}
    for city in city_list:
        assert isinstance(city, City)
        if city.nafa not in nafa_to_cities:
            nafa_to_cities[city.nafa] = []
        nafa_to_cities[city.nafa].append(city)
    # Compute the total population in each nafa
    total_nafa_populations = {nafa: sum([city.population for city in cities])
                              for (nafa, cities) in nafa_to_cities.items()}
    # Compute the nafas which appear at least once in our list
    existing_nafas = list(nafa_to_cities.keys())
    # Compute the new workplace_nafa_distribution to account for
    # nafa-s that did not appear
    for city in city_list:
        tot_existing_nafa_prob = sum([city.workplace_nafa_distribution[t] for t in existing_nafas])
        city.workplace_nafa_distribution = {nafa: prob / tot_existing_nafa_prob
                                            for (nafa, prob) in
                                            city.workplace_nafa_distribution.items()}
    # Compute the workplace_city_distribution of all cities
    for origin_city in city_list:
        # This is the correction we referred to at the beginning of the document.
        if origin_city.population == total_nafa_populations[origin_city.nafa]:
            # This might happen since we're throwing out small cities
            origin_city.proportion_working_in_city = origin_city.workplace_nafa_distribution[origin_city.nafa]
        curr_probs = {}
        for other_city in city_list:
            # Compute the probability of someone from origin_city
            # to work in other_city (3 cases as detailed above)
            if other_city.nafa != origin_city.nafa:
                curr_probs[other_city.english_name] = origin_city.workplace_nafa_distribution[other_city.nafa] *\
                                         other_city.population / total_nafa_populations[other_city.nafa]
            elif other_city != origin_city:
                curr_probs[other_city.english_name] = (origin_city.workplace_nafa_distribution[other_city.nafa] - origin_city.proportion_working_in_city) *\
                                         other_city.population / \
                                         (total_nafa_populations[other_city.nafa] - origin_city.population)
            else:
                curr_probs[other_city.english_name] = origin_city.proportion_working_in_city
        # This also asserts that the probabilities sum to 1.
        origin_city.workplace_city_distribution = DiscreteDistribution(city_list, [curr_probs[city.english_name] for city in city_list])
