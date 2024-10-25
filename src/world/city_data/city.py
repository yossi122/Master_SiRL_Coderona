# This file defines a 'City' class, which is just a mutable namedtuple
# with the attributes described below

# The attributes of City
CityFields = [
    'town_symbol',
    'hebrew_name',
    'english_name',
    'region',
    'nafa',
    'population',
    'location',
    'density',
    'num_jobs',
    'age_distribution',
    'age_data',  # For Yossi's household_generation
    'total_households',
    'household_size_distribution',
    'household_size_data',  # For Yossi's household_generation
    'kids_per_household_distribution',
    'kids_per_household_data',  # For Yossi's household_generation
    'percentage_of_households_with_65_plus',
    'percentage_of_households_with_17_minus',
    'percentage_of_households_with_single_parent',
    'percentage_of_children_with_single_parent_in_household',
    'proportion_working_in_city',
    'workplace_nafa_distribution',
    'workplace_city_distribution'
]


# We do this without namedtuple since we want it to be mutable!
# We need it to be mutable since we only compute all of the
# workplace_city_distribution-s after generating all cities.
class City(object):
    """
    An object containing all of the demographic data of a city.
    This is basically an implementation of namedtuple, only this one
    is mutable.
    """

    def __init__(self, *args, **kwargs):
        assert len(args) + len(kwargs) == len(CityFields), "Expected %d parameters, got %d" % \
                                                           (len(CityFields), len(args) + len(kwargs))
        for i in range(len(args)):
            self.__dict__[CityFields[i]] = args[i]
        for key, value in kwargs.items():
            assert key not in self.__dict__, "Got two values for '%s'" % key
            self.__dict__[key] = value
        
    def get_name(self):
        return self.english_name.lower()

    def __repr__(self):
        return "City(" + ", ".join(["%s=%s" % (key, self.__dict__[key]) for key in CityFields]) + ")"
