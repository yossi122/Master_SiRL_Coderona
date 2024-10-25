import os
import pickle
import logging
from src.world.population_generation import generate_city, generate_entire_country
from src.world.city_data import get_city_list_from_dem_xls
from src.world.population_generation.yomemut import make_city_work_destination_distributions
from src.simulation.params import Params

log = logging.getLogger(__name__)

OUTPUT_DIR_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'serializing')
OUTPUT_NAME = 'population'

MIN_CITY_SIZE = 1500

class PopulationLoader(object):
    """
    An object which wraps the generation and caching of World objects.
    """
    __slots__ = ('m_all_cities','city_smart_scale_to_world', 'output_dir', 'verbosity', 'with_caching', 'added_description')

    def __init__(self,filePath, added_description="", with_caching=True, output_dir=OUTPUT_DIR_PATH, verbosity=False):
        """
        :param added_description: A string to concatenate to the end of saved filenames.
        :param with_caching: Should the loader read from files and use internal caching to lower run times
        :param output_dir: The directory of the loaded/saved serialized files
        :param verbosity: Should the world-generation algorithm print debug info
        """
        self.m_all_cities = get_city_list_from_dem_xls(filePath)
        self.city_smart_scale_to_world = {}
        self.output_dir = output_dir
        self.verbosity = verbosity
        self.with_caching = with_caching
        self.added_description = added_description

    def _get_filepath(self, city_name, is_smart, scale):
        """
        Gets the path in which a certain generated World should be saved.
        The parameters correspond to those needed in order to
        generate the World.
        :param city_name: The name of the city to generate
        :param is_smart: Are we using smart generation
        :param scale: The scale by which to multiply the city size
        :return: The path that this serialized file should be saved to
        """
        description = self.added_description.replace(' ', '').replace(',','_').replace('[','').replace(']', '')
        filename = "%s_%s_%s_%s%s.pkl" % (OUTPUT_NAME, city_name, is_smart, scale, description)
        return os.path.join(self.output_dir, filename)

    def _save_to_file(self, world, is_smart):
        """
        Serializes and saves a generated World to a file
        :param world: The World to save
        :param is_smart: Was the world generated using smart world generation
        :return: None
        """
        filepath = self._get_filepath(world._generating_city_name, is_smart, world._generating_scale)
        assert not os.path.exists(filepath), "File '%s' already exists!" % filepath
        log.info("Saving the new results to {} ...".format(filepath))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        with open(filepath, 'wb') as f:
            pickle.dump((world, Params.loader()), f)

    def _save_on_me(self, world, is_smart):
        """
        Saves a generated World to a local dict on this object
        :param world: The World we wish to save
        :param is_smart: Was the world generated using smart world generation
        :return: None
        """
        tup = (world._generating_city_name, is_smart, world._generating_scale)
        assert tup not in self.city_smart_scale_to_world, "Dumping an existent city"
        self.city_smart_scale_to_world[tup] = (world, Params.loader())

    @staticmethod
    def assert_params_are_valid(params):
        """
        Makes sure the params that appear in a serialized file are the same as
        the current ones we are using
        (otherwise there will be compatibility errors)
        :param params: The params that should be checked against the curret ones
        :return: None (throws an exception of it does not hold)
        """
        assert params == Params.loader(), \
            "Trying to load a file corresponding to different params!"

    def try_deserialize(self, city_name, is_smart, scale):
        """
        Tries to load a World from a file (if it exists)
        :param city_name: The city we wish to load
        :param is_smart: Whether or not we wish to use smart world generation
        :param scale: The scale by which to multiply the city size
        :return: Either a loaded World or None if there was no saved version
        """
        filepath = self._get_filepath(city_name, is_smart, scale)
        if not os.path.exists(filepath):
            return None
        with open(filepath, 'rb') as f:
            log.info("Loading data")
            world, params = pickle.load(f)
            self.assert_params_are_valid(params)
            return world

    def try_load(self, city_name, is_smart, scale):
        """
        Tries to load a World, either from the local dictionary or from a file
        :param city_name: The city we wish to load
        :param is_smart: Whether or not we wish to use smart world generation
        :param scale: The scale by which to multiply the city size
        :return: Either a loaded World or None if there was no cached/saved version
        """
        tup = (city_name, is_smart, scale)
        if tup in self.city_smart_scale_to_world:
            world, params = self.city_smart_scale_to_world[tup]
            self.assert_params_are_valid(params)
            return world
        ret = self.try_deserialize(city_name, is_smart, scale)
        if ret is None:
            return None
        world = ret
        self._save_on_me(world, is_smart)
        return world

    def get_city_by_name(self, city_name):
        """
        Return a city object corresponding to a certain city name
        (calls the parsing of the demographics table)
        :param city_name: The english name of the city to load
        :return: The requested city
        (throws an exception if there was none/multiple matches)
        """
        
        possible_cities = [c for c in self.m_all_cities if c.get_name() == city_name.lower()]
        if len(possible_cities) == 0:
            raise Exception("Found no city named '%s'" % city_name)
        if len(possible_cities) > 2:
            raise Exception("Ambiguous city name '%s'" % city_name)
        return possible_cities[0]

    def get_all_large_enough_cities(self, scale):
        """
        Returns all city object corresponding to cities that after rescaling
        by 'scale' are still large enough to be generated
        :param scale: The scale by which we will multiply the city
        :return: A list of all corresponding City objects
        """

        m_all_cities = [city for city in self.m_all_cities if city.population * scale > MIN_CITY_SIZE]
        log.info("Generating %d cities with a total population of %d" %
                 (len(m_all_cities), sum(city.population for city in m_all_cities)))
        make_city_work_destination_distributions(m_all_cities)
        return m_all_cities

    def get_world(self, city_name='all', scale=1.0, is_smart=True):
        """
        Gets a World object corresponding to the given parameters.
        If self.with_caching is True, tries to load the world from
        a cached/saved version and generates a new one only if it fails,
        in which case it saves cached/saved versions of the world
        :param city_name: The name of the city to generate ('all' for entire country)
        :param scale: The scale by which to multiply the city size
        :param is_smart: Whether or not we wish to use smart population generation
        :return: A World object corresponding to the given parameters
        """
        city_name = city_name.lower()
        if self.with_caching:
            ret = self.try_load(city_name, is_smart, scale)
            if ret:
                return ret
            log.info("Could not load population of '%s'" % city_name)
        log.info("Generating city '%s'..." % city_name)
        if city_name == 'all':
            city_list = self.get_all_large_enough_cities(scale)
            world = generate_entire_country(
                city_list, is_smart, scale, verbosity=self.verbosity
            )
            log.info("Generated a total of %d people in %d environments" % (len(world.all_people()), len(world.all_environments)))
        else:
            city = self.get_city_by_name(city_name)
            world = generate_city(city, is_smart, scaling=scale, verbosity=self.verbosity)
            assert world._generating_scale == scale, \
                "Internal error: %s != %s" % (world._generating_scale, scale)
            assert world._generating_city_name == city_name, \
                "Internal error: %s != %s" % (world._generating_city_name, city_name)
        if self.with_caching:
            self._save_on_me(world, is_smart)
            self._save_to_file(world, is_smart)
        return world
