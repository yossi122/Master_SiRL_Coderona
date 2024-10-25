# This file generates all environments of a city using the model explained
# in the specification document.

import random as _random
from collections import namedtuple

from src.world.city_data import City
from src.world.environments import CityCommunity, School, Workplace
from src.world.population_generation.smart_population_generation import generate_all_households_and_communities_of_city_smart
from src.world.population_generation.naive_population_generation import generate_all_households_and_communities_of_city_naive
from src.simulation.params import Params
from src.util import divide_array
from src.world.world import World

# A 'cookbook' for environments of people in a certain age group
CrossEnvironmentData = namedtuple(
    'CrossEnvironmentData',
    [
        'env_name', # The name of this type of environment (e.g. 'high_school')
        'env_type', # The subclass of Environment that corresponds to it (e.g. School)
        'age_segment', # The ages that attend these environments
        'father_name', # The name of the father-node of this node (for instance - a School would be a father of a Classroom)
        'size', # The maximum allowed size of such an environment
        'average_daily_contacts', # The average number of contacts a person has in this environment each day
        'is_per_age' # Is this environment age-segregated (for instance, classrooms are)
    ]
)

# The params.json describe all cross-environments except for the neighborhood and city,
# which are computed a little differently
# (households are generated in a city, and whole households are divided to neighborhoods).
# We therefore define them as hardcoded-environments, with many redundant parameters as None.

NEIGHBORHOOD = "neighborhood"
NEIGHBORHOOD_NODE = CrossEnvironmentData(
    env_name=NEIGHBORHOOD,
    env_type=None,
    age_segment=(0, 99),
    father_name=None,
    size=None,
    average_daily_contacts=None,
    is_per_age=False
)
CITY = "city"
CITY_NODE = CrossEnvironmentData(
    env_name=CITY,
    env_type=None,
    age_segment=(0, 99),
    father_name=None,
    size=None,
    average_daily_contacts=None,
    is_per_age=False
)

# This translates the env_type as written in params.json to a real type.
_env_typename_to_types = {
    "School": School,
    "Workplace": Workplace
}


def load_cross_environment_data_from_json(json_dict):
    """
    :param json_dict: a dict obtained from the 'city_environments' section
    of the params.json file
    :return: a CrossEnvironmentData object
    """
    assert json_dict['env_type'] in _env_typename_to_types
    env_type = _env_typename_to_types[json_dict['env_type']]
    return CrossEnvironmentData(
        env_name=json_dict['env_name'],
        env_type=env_type,
        age_segment=json_dict['age_range'],
        father_name=json_dict['father_name'],
        size=json_dict['size'],
        average_daily_contacts=json_dict['average_daily_contacts'],
        is_per_age=json_dict['is_per_age']
    )

def node_name_per_age(name, age):
    """
    Expand the name of a per-age environment something that depends on the age
    :param name: the original environment name (e.g. 'high_school')
    :param age: the environment age (e.g. 17)
    :return: (e.g. 'high_school_age_17')
    """
    return "{}_age_{}".format(name, age)

def expand_per_age_environments(node_name_list, env_node_dict):
    """
    Expand a list of CrossEnvironmentData nodes to another list that is already
    completely age stratified.
    For instance, if neighborhood->elementary_school->classroom->social_group
    where the first two are not age dependent and the last two are,
    each of the last two will be expanded to multiple nodes (one per age),
    where the father nodes will be determined correctly
    (elementary_school->classroom_age_i->social_group_age_i).
    :param node_name_list: A list of the names of the environments
    (given in topological order)
    :param env_node_dict: A python dict whose keys are node names
    and whose values are the actual nodes.
    Note that it is important that the node_name_list is ordered topologically,
    this is way we even need it (and not use env_node_dict.keys()). dict.keys()
    preserves insertion order in python3.7, but we did not want to create this
    compatibility constraint.
    :return: A pair (new_node_name_list, new_env_node_dict)
    """
    rv_dict = {}
    rv_name_list = []
    for name in node_name_list:
        node = env_node_dict[name]
        assert isinstance(node, CrossEnvironmentData)
        if node.father_name is None:
            rv_dict[name] = node
            rv_name_list.append(name)
            continue

        orig_father_node = env_node_dict[node.father_name]
        if not node.is_per_age:
            assert not orig_father_node.is_per_age, \
                "Env {} is per age, but father env {} is per age".format(name, orig_father_node.env_name)
            assert name not in rv_dict
            rv_dict[name] = node
            rv_name_list.append(name)
            continue

        for age in list(range(node.age_segment[0], node.age_segment[1] + 1)):
            if orig_father_node.is_per_age:
                new_father_name = node_name_per_age(orig_father_node.env_name, age)
            else:
                new_father_name = orig_father_node.env_name
            assert new_father_name in rv_dict
            new_name = node_name_per_age(name, age)
            new_node = CrossEnvironmentData(
                env_name=new_name,
                env_type=node.env_type,
                age_segment=(age, age),
                father_name=new_father_name,
                size=node.size,
                average_daily_contacts=node.average_daily_contacts,
                is_per_age=False
            )
            assert new_name not in rv_dict
            rv_dict[new_name] = new_node
            rv_name_list.append(new_name)
    return rv_name_list, rv_dict


def divide_people_to_environments(people, max_env_size, env_type, city, avg_num_daily_contacts, age_segment, name=None):
    """
    Divide a set of people evenly into environments
    :param people: The set of people to divide
    :param max_env_size: The maximum allowed environment size
    :param env_type: The type of the environment we wish to create
    (its constructor is called to create each instance)
    :param city: The city object (to be passed to the constructor)
    :param avg_num_daily_contacts: The average number of daily contacts each
    person should have in this environment (contact_prob is computed from it)
    :param age_segment: The age_segment of this environment
    (to be passed on to the constructor)
    :param name: The full name of the environment
    (to be passed on to the constructor)
    :return: A pair
    (all environments created, the subsets of people that comprise them)
    """
    _random.shuffle(people)
    all_environments = []
    people_segments = divide_array(people, max_env_size)
    for curr_people in people_segments:
        contact_prob = avg_num_daily_contacts / (max(len(curr_people) - 1, 1))
        # Create an instance of the environment
        curr_env = env_type(
            city=city,
            contact_prob_between_each_two_people=contact_prob,
            age_segment=age_segment,
            full_name=name
        )
        all_environments.append(curr_env)
        for person in curr_people:
            person.add_environment(curr_env)
    return all_environments, people_segments


def make_cross_environments(people, cross_env_list, city, initial_divisions):
    """
    Divides the population randomly into the given cross environments,
    adds the environments to the corresponding people and returns the new environments.
    :param people: The list of people to be divided
    :param cross_env_list: The list of cross environments to divide by
    :param city: The city (passed as a parameter to the constructor of the environments)
    :param initial_divisions: {NEIGHBORHOOD: list of lists of people, CITY: list of list of people}
    :return: The newly created environments
    """
    all_environments = []
    cross_env_list = [CITY_NODE, NEIGHBORHOOD_NODE] + cross_env_list
    name_list = [e.env_name for e in cross_env_list]
    env_dict = {e.env_name: e for e in cross_env_list}
    name_list, env_dict = expand_per_age_environments(name_list, env_dict)
    for env_name in name_list:
        if env_name in [NEIGHBORHOOD, CITY]:
            continue
        node = env_dict[env_name]
        father_node_envs = initial_divisions[node.father_name]
        divisions_at_node = []
        for single_env_population in father_node_envs:
            new_environments, people_division = divide_people_to_environments(
                [p for p in single_env_population if node.age_segment[0] <= p.get_age() <= node.age_segment[1]],
                node.size,
                node.env_type,
                city,
                avg_num_daily_contacts=node.average_daily_contacts,
                age_segment=node.age_segment,
                name=node.env_name
            )
            divisions_at_node += people_division
            for env in new_environments:
                all_environments.append(env)
        initial_divisions[env_name] = divisions_at_node
    return all_environments


def generate_city(city, is_smart_household_generation, internal_workplaces=True, scaling=1.0, verbosity=False, to_world=True):
    """
    Generates the population of a city and divides them into environments
    (as described in the specification document).
    :param city: The City object corresponding to the city we wish to create
    :param is_smart_household_generation:
    Should we use smart_population_generation or naive_population_generation
    (this should always be naive, unless testing this particular thing)
    :param internal_workplaces: Should the workplace be generated
    (from their CrossEnvironmentDatas nodes) or should they not be generated
    (so they may be generated between different cities).
    This should be True when generating a model of a single city
    and False when generating a model for multiple cities.
    :param scaling: The scale we wish to multiply the city by
    (see the specification document for details)
    :param verbosity: Whether or not this should print debug data
    :param to_world: Should we return a World object or a pair
    (all_people, all_environments)?
    This should be True when generating a single city and False when generating
    multiple cities (so they may be added together before constructing World)
    :return: Eiteher a World object or a pair (all_people, all_environments),
    depending on the to_world parameter
    """
    assert isinstance(city, City)
    if is_smart_household_generation:
        all_people, all_neighborhoods, all_environments = \
            generate_all_households_and_communities_of_city_smart(city, scaling, verbosity=verbosity)
    else:
        all_people, all_neighborhoods, all_environments = \
            generate_all_households_and_communities_of_city_naive(city, scaling)
    city_env_params = Params.loader()['city_environments']
    cross_environments = []
    for env_params in city_env_params:
        if (env_params["env_name"] == 'workplace') and not internal_workplaces:
            continue
        cross_environments.append(load_cross_environment_data_from_json(env_params))

    initial_division = {NEIGHBORHOOD: all_neighborhoods, CITY: [all_people]}
    new_environments = make_cross_environments(all_people, cross_environments, city, initial_division)
    for env in new_environments:
        all_environments.append(env)
    city_community = CityCommunity(
        city,
        Params.loader()['population']['city_avg_daily_contacts'] / \
        float(len(all_people))
    )
    for environment in all_environments:
        city_community.add_environment(environment)
        environment.set_city_env(city_community)
    for person in all_people:
        person.add_environment(city_community)
    all_environments.append(city_community)
    if to_world:
        assert internal_workplaces
        return World(all_people, all_environments, city.english_name, scaling)
    assert not internal_workplaces
    return all_people, all_environments


def generate_entire_country(city_list, is_smart_household_generation=True, scaling=1.0, verbosity=False):
    """
    Generates the World of a model of the entire country, in the same way as
    was detailed in the specification document.
    :param city_list: The list of cities which comprise this model
    (workplace_city_distribution should already be initialized)
    :param is_smart_household_generation: Are we using
    smart_population_generation or naive_population_generation.
    This should only be False for testing/debugging purposes.
    :param scaling: The scale that should multiply all of the cities involved.
    :param verbosity: Whether or not we should print debug information.
    :return: A world object corresponding to a model of all of these cities.
    """
    all_people = []
    all_environments = []
    city_to_workers = {city: [] for city in city_list}
    workplace_node = None
    city_env_params = Params.loader()['city_environments']
    for env_params in city_env_params:
        if env_params["env_name"] == 'workplace':
            workplace_node = load_cross_environment_data_from_json(env_params)
    assert workplace_node is not None, "Could not find workplace in params.json!"
    for city in city_list:
        city_people, city_environments = generate_city(
            city,
            is_smart_household_generation=is_smart_household_generation,
            internal_workplaces=False,
            scaling=scaling,
            verbosity=verbosity,
            to_world=False
        )
        for person in city_people:
            all_people.append(person)
            if workplace_node.age_segment[0] <= person.get_age() <= workplace_node.age_segment[1]:
                workplace_city = city.workplace_city_distribution.sample()
                city_to_workers[workplace_city].append(person)
        for environment in city_environments:
            all_environments.append(environment)
    for city, workers in city_to_workers.items():
        workplaces_in_city, people_segments = divide_people_to_environments(
            workers,
            workplace_node.size,
            Workplace,
            city,
            workplace_node.average_daily_contacts,
            age_segment=workplace_node.age_segment,
            name=workplace_node.env_name
        )
        for workplace in workplaces_in_city:
            all_environments.append(workplace)
    return World(all_people, all_environments, 'all', scaling)
