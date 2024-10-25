import os
from functools import lru_cache
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
import warnings
from numpy import sqrt, nanmean, nanstd, NaN, isnan, array, logical_or
import csv
import logging
from collections import Counter, namedtuple, defaultdict
from itertools import cycle
from datetime import timedelta
from itertools import combinations

from src.simulation.interventions import *
from src.world import InfectionData
from src.logs.summary import make_summary_by_age_table, TableFormat
from src.simulation.params import Params
from src.seir import DiseaseState
from src.world import RedactedPersonAndEnv,World
from src.logs.r0_data import calculate_r0_data

from typing import List
from src.world import Person
from src.simulation.interventions.intervention import Intervention, LockdownIntervention
import datetime
from functional import seq
from collections import namedtuple

logging.getLogger('matplotlib.font_manager').disabled = True

BackgroundStripe = namedtuple("BackgroundStripe", ("start", "end", "color", "label"))

INTERVENTION_TYPE_TO_COLOR = {
    SocialDistancingIntervention: 'green',
    ElderlyQuarantineIntervention: 'yellow',
    SchoolClosureIntervention: 'orange',
    SchoolIsolationIntervention: 'brown',
    SymptomaticIsolationIntervention: 'red',
    CityCurfewIntervention: 'gray',
    HouseholdIsolationIntervention: 'blue',
    WorkplaceClosureIntervention: 'teal',
    LockdownIntervention: 'pink'
}

INTERVENTION_TYPE_TO_LABEL = {
    CityCurfewIntervention: 'city curfew intervention',
    ElderlyQuarantineIntervention: 'Elderly Quarantine',
    HouseholdIsolationIntervention: 'household isolation',
    LockdownIntervention: 'lockdown isolation',
    SocialDistancingIntervention: 'social distancing',
    SchoolClosureIntervention: 'school closure',
    SchoolIsolationIntervention: 'school isolation',
    SymptomaticIsolationIntervention: 'symptomatic isolation',
    WorkplaceClosureIntervention: 'workplace closure'
}

class DataToPlot(object):
    """
    An object representing a graph to plot from the statistics (there could be multiple DataToPlot in one figure)
    property_to_count = a lambda that takes a redacted person or redacted infection data and returns a boolean (e.g. a person is in some age group).
    props = visual properties dictionary for the graph (e.g. its color, shape)
    is_integral = draw the number of people satisfying that property (like number of people in some disease state)
    or the difference in that number
    infection_data = None if property_to_count takes redacted people,
    otherwise a tuple of the properties of infection data that property_to_count uses.
    See the design document for more details.
                     we save only projections of up two properties in order to save memory.
    """
    def __init__(self, property_to_count, props, is_integral=True, infection_data=None):
        self.property_to_count = property_to_count
        self.props = props
        self.is_integral = is_integral
        self.infection_data = infection_data


def integral_list(l):
    """
    Returns the list of cumulative sums of the given list.
    For instance, integral_list([1, 2, 3]) would return [1, 3, 6]
    """
    res = [0.]
    for elem in l:
        res.append(res[-1] + elem)
    return res[1:]


class DayStatistics(object):
    """
    A summary of the changes that happened at a certain date.
    self.person_count saves the age-stratified changes in the disease states of of all people
    self.infection_data_projection saves the data of this date's infections (see InfectionData for details).


    self.person_count is a counter object which saves, for each state and age, the number of new people in that state
    minus the number of people which are no longer in that state.

    self.infection_data_projection is a dictionary from subsets of the properties it saves to a counter
    (not of complete InfectionData objects in order to save memory).
    Currently it saves single properties and pairs of properties.
    """
    __slots__ = ('date', 'person_count', 'infection_data_projection', 'diff_infect', 'infected_today')

    def __init__(self, date, changed_population):
        self.date = date
        self.person_count = Counter(
            person.get_state()
            for person in changed_population
        )
        self.person_count.subtract(
            person.get_last_state()
            for person in changed_population
        )
        infected_today_stats = list(
            person.get_infection_data().get_stats()
            for person in changed_population
            if (person.get_infection_data() is not None) and (person.get_infection_data().date == date)
        )
        self.infected_today = len(infected_today_stats)
        self.infection_data_projection = {
            key: Counter(stat[key] for stat in infected_today_stats)
            for key in InfectionData.get_keys()
        }
        self.infection_data_projection.update({
            pair: Counter((stat[pair[0]], stat[pair[1]]) for stat in infected_today_stats)
            for pair in combinations(InfectionData.get_keys(), 2)
        })

        self.person_count.pop(None, None)
        self.diff_infect = sum(weight for state, weight in self.person_count.items() if state[1].is_infected())

    def __repr__(self):
        return 'day: ' + repr(self.date) + '\n' + \
            repr(self.person_count)


class Statistics(object):
    """
    The main object documenting the result of a single simulation.
    Saves in self._days_data the DayStatistics object of each day.
    """
    __slots__ = (
        '_output_path',
        '_days_data',
        '_final_state',
        'num_infected',
        '_interventions',
        '_r0_data',
        'min_date',
        'max_date',
        '_params_at_init',
        'all_environment_names',
        'full_env_name_to_short_env_name',
        '_hood_data'
    )

    def __init__(self, output_path, world):
        self._output_path = output_path
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        self._days_data = []
        self._final_state = None
        self._interventions: List[Intervention] = []
        self._r0_data = None
        self.num_infected = 0
        self.min_date = None
        self.max_date = None
        self._hood_data= {}
        self.all_environment_names = set([env._full_name for env in world.all_environments])
        self.all_environment_names.add('initial_group')
        self.full_env_name_to_short_env_name = {'initial_group': 'initial_group'}
        for env in world.all_environments:
            if env._full_name in self.full_env_name_to_short_env_name:
                assert self.full_env_name_to_short_env_name[env._full_name] == env.name
            self.full_env_name_to_short_env_name[env._full_name] = env.name
        self._params_at_init = Params.loader()

    def add_daily_data(self, daily_data: DayStatistics,world:World):
        """
        Register the data of this day
        """
        assert self._final_state is None, "Can't add daily data after marked ending!"
        self._days_data.append(daily_data)
        self.num_infected += self._days_data[-1].diff_infect
        self.update_date_range(daily_data.date)
        #Counts how many are sick in each neighborhood
        hood_data = {}
        for person in world.all_people():
            if person.get_disease_state() == DiseaseState.SYMPTOMATICINFECTIOUS:
                hoodID = person.get_neighberhood().get_neighborhood_id()
                if hoodID not in hood_data:
                    hood_data[hoodID] = 1 
                else:
                    hood_data[hoodID] +=1
        self._hood_data[daily_data.date] = hood_data

    def get_neiborhood_data(self,date,hood_id):
        '''
        Returns the number of infected people which lives in certain neighborhood
        '''
        if self._hood_data == None:             return 0
        if date not in self._hood_data:         return 0 
        if hood_id not in self._hood_data[date]:return 0
        return  self._hood_data[date][hood_id]
            
    def mark_ending(self, all_people):
        """
        Mark this as a complete simulation and save more detailed data
        of the final states of people in the simulation.
        """
        assert self._final_state is None, "Can't mark ending twice!"
        all_redacted_people = []
        for person in all_people:
            if person.get_infection_data() is not None:
                all_redacted_people.append(
                    RedactedPersonAndEnv(person.get_age(), person.get_disease_state(),
                                         person.get_infection_data().environment._full_name
                                         )
                )
            else:
                all_redacted_people.append(
                    RedactedPersonAndEnv(person.get_age(), person.get_disease_state(), None)
                )
        self._final_state = Counter(all_redacted_people)

    def update_date_range(self, date):
        """
        Updates the min_date and max_date of this simulation
        after introducing a new date.
        :param date: The new date to introduce
        :return: None
        """
        if self.min_date is None:
            self.min_date = date
        if self.max_date is None:
            self.max_date = date
        self.min_date = min(self.min_date, date)
        self.max_date = max(self.max_date, date)

    def add_intervention(self, intervention: Intervention):
        """
        Adds an intervention to the documented list of interventions
        :param intervention: an intervention to add
        :return: None
        """
        self._interventions.append(intervention)

    def get_days_data(self) -> List[DayStatistics]:
        """
        Return the list of DayStatistics held on this object
        :return: self._days_data
        """
        return self._days_data

    def get_dates(self):
        """
        :return: The list of all dates of the DayStatistics held on this object
        """
        return [d.date for d in self._days_data]

    def get_r0_data(self):
        return self._r0_data

    def is_static(self):
        """
        Return whether the disease has gone extinct
        """
        if len(self._days_data) < 5:
            return False
        return self.num_infected == 0

    def plot_daily_sum(self, image_path, datas_to_plot):
        """
        Plot and save a graph with curves corresponding to given DataToPlot objects.
        :param image_path: The (formatless) path in which to save the csv and svg files
        :param datas_to_plot: A list of DataToPlot objects correspondingt o the requested curves
        :return: None
        """
        datas = []
        dates = self.get_dates()
        for data_to_plot in datas_to_plot:
            datas.append({
                'data': self.sum_days_data(
                    data_to_plot.property_to_count, data_to_plot.is_integral, data_to_plot.infection_data),
                'props': data_to_plot.props
            })
        self.plot(
            os.path.join(self._output_path, image_path),
            dates,
            datas,
            self.make_background_stripes(),
            y_axes_label="infections #"
        )

    def clip_date_to_time_frame(self, date):
        """
        :param date: Some data
        :return: The clipped date, moved the minimum amount to fit in the time frame of this object
        """
        return min(max(date, self.min_date), self.max_date)

    def make_background_stripes(self):
        """
        Makes the background stripes corresponding to the interventions
        that this object holds.
        :return: A list of BackgroundStripe objects, one for each intervention.
        """
        ret = []
        for intervention in self._interventions:
            if intervention.start_date is not None:
                assert type(intervention) in INTERVENTION_TYPE_TO_COLOR, \
                    "Intervention of type '%s' has no defined color!" % type(intervention)
                ret.append(BackgroundStripe(
                    self.clip_date_to_time_frame(intervention.start_date),
                    self.clip_date_to_time_frame(intervention.start_date + intervention.duration),
                    INTERVENTION_TYPE_TO_COLOR[type(intervention)],
                    INTERVENTION_TYPE_TO_LABEL[type(intervention)]
                ))
        return ret

    def calculate_susceptible(self, today_date: datetime.date, population: List[Person]):
        count = 0
        for person in population:
            if person.is_susceptible:
                count += 1
        self._susceptibles[today_date] = count

    def calc_r0_data(self, population: List[Person], max_date=None, min_age=0, max_age=100):
        """
        Calculate the R data of the given list of people
        (i -> the average number of people a person on day i infected)
        and save it on this object.
        :param population: A list of Person objects on which to calculate the data
        :param max_date: The last date of the graph
        :return: None, the data is saved on self
        """
        if max_date is None:
            max_date = self.max_date
        self._r0_data = calculate_r0_data(population, max_date, min_age, max_age)

    def plot_r0_data(self, image_path, avg_props=None, smoothed_props=None):
        """
        Plots the R data of this simulation (computed in advance).
        :param image_path: The (formatless) path to which to save the resulting csv, svg
        :param avg_props: Visual properties that should be added to the non-smoothed graph
        :param smoothed_props: Visual properties that should be added to the smoothed graph
        :return: None (saves result to files)
        """
        if avg_props is None:
            avg_props = {}
        if 'label' not in avg_props:
            avg_props['label'] = "base reproduction number r"

        if smoothed_props is None:
            smoothed_props = {}
        if 'label' not in smoothed_props:
            smoothed_props['label'] = "smoothed base reproduction number r"

        r_effective_props = {}
        if 'label' not in r_effective_props:
            r_effective_props['label'] = "r effective"

        instantaneous_r_props = {}
        if 'label' not in instantaneous_r_props:
            instantaneous_r_props['label'] = "instantaneous r"

        datas = [
            {'data': self._r0_data['smoothed_avg_r0'], 'props': smoothed_props},
            {'data': self._r0_data['avg_r0'], 'props': avg_props},
            # {'data': self._r0_data['estimated_r0'], 'props': r_effective_props},
            {'data': self._r0_data['instantaneous_r'], 'props': instantaneous_r_props},
        ]
        self.plot(os.path.join(self._output_path, image_path), self._r0_data['dates'], datas, self.make_background_stripes(), y_axes_label="r")

    def sum_days_data(self, property_to_count, is_integral, infection_data=None):
        """
        :param property_to_count: A function that takes a redacted person or
        redacted infection data and returns a boolean
        (whether or not the person satisfied the property).
        :param is_integral: Should the function compute the number of people
        satisfying the property each day or the daily change in that number
        :param infection_data: A string or tuple of strings (or None) referring
        to the type of data that property_to_count accepts
        (see the design document for more details)
        :return: A list of the number of people satisfying the property each day
        (or the daily change in that number, in the case where is_integral is False)
        """
        data = []
        for day in self._days_data:
            if infection_data is None:
                today_data = sum(
                    count
                    for reducted_person, count
                    in day.person_count.items()
                    if property_to_count(reducted_person)
                )
            else:
                today_data = sum(
                    count
                    for reducted_infection, count
                    in day.infection_data_projection[infection_data].items()
                    if property_to_count(reducted_infection)
                )
            data.append(today_data)
        if is_integral:
            data = integral_list(data)
        return data

    def count_property_at_end(self, property_to_count):
        """
        Count the number of people satisfying some property
        at the end of the simulation. Same as taking the last element of
        self.sum_days_data(property_to_count, True), but runs much faster.
        :param property_to_count: A function which accepts a RedactedPerson
        and outputs whether or not the person satisfies that property.
        :return: The number of people satisfying that property.
        """
        return sum(
            count
            for redacted_person, count in self._final_state.items()
            if property_to_count(redacted_person)
        )

    @lru_cache(maxsize=None)
    def get_summary_data_for_age_group(self, age_group, shortened=True):
        """
        Compute and return some interesting properties about the people
        in the given age group in the simulation.
        :param age_group: The age group for which we produce the data.
        :param shortened: Whether we want a shortened version or the full one.
        :return: A dict of the format {property_name: amount} detailing
        the different properties of individuals of the given age group.
        """
        assert self._final_state is not None, "Can't get summary before marked done!"
        if age_group is None:
            person_in_age_group = lambda person: True
        else:
            def person_in_age_group(person):
                return age_group[0] <= person.age <= age_group[1]

        total_people = self.count_property_at_end(person_in_age_group)
        total_deceased = self.count_property_at_end(
            lambda person: person_in_age_group(person) and
                           person.disease_state == DiseaseState.DECEASED
        )
        total_infected = self.count_property_at_end(
            lambda person: person_in_age_group(person) and
                           person.disease_state != DiseaseState.SUSCEPTIBLE
        )
        maximum_infected_simultaneously = max(self.sum_days_data(
            lambda person: person_in_age_group(person) and
                   person.disease_state.is_infected(), True
        ))
        maximum_critical_simultaneously = max(self.sum_days_data(
            lambda person: person_in_age_group(person) and
                           person.disease_state == DiseaseState.CRITICAL, True
        ))

        ret = defaultdict(int)
        ret["Total people"] = total_people
        ret["Total deceased"] = total_deceased
        ret["Total infected"] = total_infected
        ret["Maximum infected simultaneously"] = maximum_infected_simultaneously
        ret["Maximum critical simultaneously"] = maximum_critical_simultaneously

        for env_name in self.all_environment_names:
            if shortened:
                name = self.full_env_name_to_short_env_name[env_name]
            else:
                name = env_name
            key_str = "Total infected in {}".format(name)
            if key_str not in ret:
                ret[key_str] = 0
        for redacted_person, count in self._final_state.items():
            if not person_in_age_group(redacted_person):
                continue
            env_name = redacted_person.infection_env_source
            if env_name is None:
                # Person not infected
                continue
            if shortened:
                name = self.full_env_name_to_short_env_name[env_name]
            else:
                name = env_name
            key_str = "Total infected in {}".format(name)
            ret[key_str] += count
        return ret
    def get_state_stratified_summary_table(self,table_format:TableFormat,is_relative = True):
        """
        Computes and returns summary data which is stratified
        by the given disease states
        :param states: The disease states by which to stratify the data
        :param is_relative: Should the results be relative (i.e. the percentage
        of people satisfying each property) or absolute numbers
        :param shortened: Should this generate a short summary or
        a more detailed one
        :return: A concise readable text containing a table of
        age-stratified results (delta deceased, delta infected, ...)
        """
        #Collecting the data
        lst = []
        for i in range(len(self._days_data)):
            if self._days_data[i].person_count == 0:
                tmp = (self._days_data[i].date,None)
            else:
                d = self._days_data[i].date
                states_col = []
                for reducted_person, count in self._days_data[i].person_count.items():
                    for c in range(count):
                        states_col.append(reducted_person.disease_state.name)
                if self._days_data[i].infected_today > 0:
                    seq(range(self._days_data[i].infected_today)).for_each(lambda x: states_col.append('infected_today'))
                cnt = Counter(states_col)
                tmp = (d,cnt)
            lst.append(tmp)

        
        #Create corresponding string 
        table = [[] for i in range(len(lst)+1)]
        states = [state.name for state in DiseaseState] + ['infected_today', 'critical_today']
        table [0] = ["Dates"] + states
        for i in range(1,len(lst)+1):
            date,data = lst[i-1]
            tmp = [str(data[s]) if s in data else '0' for s in states]
            table[i] = [str(date)] + tmp
        ans = table_format.format(table)
        return ans

    def get_age_stratified_summary_table(
        self,
        table_format,
        age_groups=(
            None,  # None means all ages
            (0, 9),
            (10, 19),
            (20, 29),
            (30, 39),
            (40, 49),
            (50, 59),
            (60, 69),
            (70, 79),
            (80, 89),
            (90, 99),
        ),
        is_relative=True,
        shortened=True
    ):
        """
        Computes and returns summary data which is stratified
        by the given age groups
        :param age_groups: The age groups by which to stratify the data
        :param is_relative: Should the results be relative (i.e. the percentage
        of people satisfying each property) or absolute numbers
        :param shortened: Should this generate a short summary or
        a more detailed one
        :return: A concise readable text containing a table of
        age-stratified results (total deceased, total infected, ...)
        """
        datas = {age_group: [self.get_summary_data_for_age_group(age_group, shortened=shortened)] for age_group in age_groups}
        return make_summary_by_age_table(datas, table_format, is_relative)

    @staticmethod
    def write_multiple_stats_summary_file(
        stats,
        outdir,
        name,
        age_groups=(
            None,
            (0, 9),
            (10, 19),
            (20, 29),
            (30, 39),
            (40, 49),
            (50, 59),
            (60, 69),
            (70, 79),
            (80, 89),
            (90, 99),
        ),
        shortened=True
    ):
        """
        Computes and saves several summary data files of multiple Statistics
        objects, stratified by the given age groups.
        Expectations and standard errors are computed.
        :param stats: A list of Statistics objects to analyze
        :param outdir: The directory into which to save the files.
        :param name: The scenario name, will affect the filenames.
        :param age_groups: The age groups by which to stratify the data
        :param shortened: Should this generate a short summary or
        a more detailed one
        :return: None (output files are saved).
        """
        if shortened:
            outpath = os.path.join(outdir, name + '_summary')
            absolute_table_path = os.path.join(outdir, name + '_absolute_table')
        else:
            outpath = os.path.join(outdir, name + '_summary_long')
            absolute_table_path = os.path.join(outdir, name + '_absolute_table_long')
        datas = {age_group: [
            stat.get_summary_data_for_age_group(age_group, shortened=shortened)
            for stat in stats
        ] for age_group in age_groups}
        for table_format in (TableFormat.TEXTUAL, TableFormat.CSV):
            extension = '.' + table_format.get_file_extension()
            for path in (outpath, absolute_table_path):
                assert not os.path.exists(path + extension), "Failed to create file '%s': file exists!" % (path + extension)
            with open(outpath + extension, 'w') as f:
                f.write(make_summary_by_age_table(datas, table_format))
            with open(absolute_table_path + extension, 'w') as f:
                f.write(make_summary_by_age_table(datas, table_format, False))
    
    def write_daily_delta(self, filename):
        outpath = os.path.join(self._output_path, filename)
        absolute_table_path = os.path.join(self._output_path, 'daily_delta_table')
        for table_format in TableFormat:
            extension = '.' + table_format.get_file_extension()
            for path in (outpath, absolute_table_path):
                assert not os.path.exists(path + extension), "Failed to create file '%s': file exists!" % (path + extension)
            with open(outpath + extension, 'w') as f:
                f.write(self.get_state_stratified_summary_table(table_format=table_format))


    def write_summary_file(self, filename, shortened=True):
        """
        Write a relative and an absolute summary file for the simulation.
        :param filename: The name of the relative summary file
        (the absolute is computed internally)
        :param shortened: Whether we want a shortened or full version.
        :return: None (saves the files).
        """
        outpath = os.path.join(self._output_path, filename)
        if shortened:
            absolute_table_path = os.path.join(self._output_path, 'absolute_table')
        else:
            absolute_table_path = os.path.join(self._output_path, 'absolute_table_long')
        for table_format in (TableFormat.TEXTUAL, TableFormat.CSV):
            extension = '.' + table_format.get_file_extension()
            for path in (outpath, absolute_table_path):
                assert not os.path.exists(path + extension), "Failed to create file '%s': file exists!" % (path + extension)
            with open(outpath + extension, 'w') as f:
                f.write(self.get_age_stratified_summary_table(table_format=table_format, shortened=shortened))
            with open(absolute_table_path + extension, 'w') as f:
                f.write(self.get_age_stratified_summary_table(table_format=table_format, is_relative=False, shortened=shortened))

    def write_params(self):
        """
        Write a params.json file corresponding to the parameters used
        to the output directory (for documentation purposes)
        """
        params_path = os.path.join(self._output_path, 'params.json')
        params = Params.loader()
        assert params == self._params_at_init, "Params changed mid-simulation!"
        params.dump(params_path)

    def write_inputs(self, sim):
        """
        Write an inputs.txt file which documents the parameters which were used
        to generate this simulation.
        :param sim: The Simulation object
        """
        inputs_path = os.path.join(self._output_path, 'inputs.txt')
        with open(inputs_path, 'w') as f:
            f.write('\n'.join([
                "City: {}".format(sim._world._generating_city_name),
                "Scale: {}".format(sim._world._generating_scale),
                "Initial date: {}".format(sim._initial_date),
                "Initial infection: {}".format(sim.initial_infection_doc),
                "Num days: {}".format(sim.num_days_to_run),
                "base_infectiousness: {}".format(self._params_at_init["person"]["base_infectiousness"]),
                "",
                "Interventions:",
                ""
            ]))
            for intervention in sim.interventions:
                f.write("{}\n".format(intervention))

    def write_interventions_inputs_csv(self):
        """
        Writes the parameters used to construct the interventions of
        the simulation to a csv file (for ease of parsing).
        """
        inputs_path = os.path.join(self._output_path, 'interventions_inputs.csv')
        with open(inputs_path, 'w', newline='') as f:
            csv_writer = csv.writer(f)
            last_name = None
            count = 0
            for intervention in sorted(self._interventions, key=lambda x: type(x).__name__):
                curr_name = type(intervention).__name__
                attr = intervention.attributes()
                if curr_name == last_name:
                    count += 1
                else:
                    count = 1
                    last_name = curr_name
                for key, value in attr.items():
                    if isinstance(value, dict):
                        for param_key, param_value in value.items():
                            params = [curr_name, count, "{}.{}".format(key, param_key), param_value]
                            csv_writer.writerow(params)
                    else:
                        params = [curr_name, count, key, value]
                        csv_writer.writerow(params)

    def dump(self, path):
        """
        Save this object to a file, so we may use it to generate more outputs
        or debug errors without having to rerun the simulation every time.
        :param path: The path we wish to save the
        """
        if not os.path.exists(self._output_path):
            os.makedirs(self._output_path)
        outpath = os.path.join(self._output_path, path)
        assert not os.path.exists(outpath), "File %s already exists!" % outpath
        with open(outpath, 'wb') as f:
            pickle.dump(self, f)

    def write_all_person_stats_to_csv(self, persons: List[Person], to_file_name: str = "persons_age_sickness.csv"):
        if not os.path.exists(self._output_path):
            os.makedirs(self._output_path)
        outpath = os.path.join(self._output_path, to_file_name)
        PERSON_STATS = namedtuple('PERSON_STATS', 'age sickness_status')
        seq(persons).map(lambda person: PERSON_STATS(person.get_age(), person.get_disease_state())).to_csv(outpath)

    @staticmethod
    def load(path):
        """
        Load a saved Statistics object from a file
        :param path: The path of the file containing the saved Statistics object
        :return: The Statistics object
        """
        with open(path, 'rb') as f:
            return pickle.load(f)

    def __repr__(self):
        return repr(self._days_data[-1])

    @staticmethod
    def remove_nans_from_plot(x, ys):
        """
        Takes a list of x-values and corresponding lists of y-values,
        removes any x-values for which all y-values are nan and updates
        the y-value lists as well.
        :param x: A list of x-values underlying some graph.
        :param ys: A list of lists of y-values corresponding to these x-values.
        :return: A pair (new_x, new_ys) in the format of the input, only
        without any x-values for which all y-values are nan.
        """
        if len(x) < len(ys[0]):
            # ys = array(ys)[range(len(x))]
            ys = [array(y)[range(len(x))] for y in ys]
        masks = [isnan(y) for y in ys]
        mask = masks[0]
        for m in masks:
            mask = logical_or(mask, m)
        return array(x)[~mask], [array(y)[~mask] for y in ys]

    @staticmethod
    def get_csv_svg_filenames_and_check_they_do_not_exist(clean_filename):
        """
        Takes a filename without any format, adds .csv and .svg and asserts
        that these files do not exist.
        :param clean_filename: A filename without any format.
        :return: A pair of filenames (csv_filename, svg_filename)
        """
        csv_filename = "{}.csv".format(clean_filename)
        svg_filename = "{}.svg".format(clean_filename)
        #assert not os.path.exists(svg_filename), "File {} already exists!".format(svg_filename)
        #assert not os.path.exists(csv_filename), "File {} already exists!".format(csv_filename)
        return csv_filename, svg_filename

    @staticmethod
    def compute_axis_ticks(data, number_of_ticks):
        minimum = min(data)
        maximum = max(data)
        step = (maximum - minimum) / number_of_ticks
        return [(minimum+tick*step) for tick in range(number_of_ticks+1)]

    @staticmethod
    def plot(image_path, dates, datas, background_stripes=None, is_dates=True, y_axes_label=""):
        """
        plot a graph:
        dates = the x-axis range
        datas = list of data to plot
        background_stripes = add shades to mark periods of time (as interventions)
        is_dates = if false, don't treat dates as dates
        """
        if background_stripes is None:
            background_stripes = []
        yscales = []
        for data in datas:
            if 'yscale' in data['props']:
                yscales.append(data['props']['yscale'])
                del data['props']['yscale']
            else:
                yscales.append('linear')
        assert all([yscale == yscales[0] for yscale in yscales]), "All yscales must be equal!"
        yscale = yscales[0]
        plt.clf()
        plt.figure(figsize=(10, 10), dpi=80)
        if is_dates:
            npdates = [mdates.date2num(d) for d in dates]
        else:
            npdates = dates
        drew_anything = False
        for data in datas:
            new_dates, new_data = Statistics.remove_nans_from_plot(npdates, [data['data']])
            if len(new_dates) == 1:
                # Plotting one dot throws an exception in matplotlib
                continue
            drew_anything = True
            new_data = new_data[0]
            plt.rcParams["figure.figsize"] = (20, 3)
            plt.plot(new_dates, new_data, **data['props'])
            plt.xlabel("Date" if is_dates else "????")
            plt.ylabel(y_axes_label)
            plt.title(os.path.basename(image_path))
            plt.grid(color='pink', linestyle='--', linewidth=0.5)
        if not drew_anything:
            warnings.warn(f"Did not draw anything! No good data! for {os.path.basename(image_path)}")
            return
        for stripe in background_stripes:
            plt.axvspan(
                mdates.date2num(stripe.start),
                mdates.date2num(stripe.end),
                facecolor=stripe.color,
                alpha=0.2,
                label=stripe.label
            )
        if is_dates:
            fig = plt.gcf()
            fig.autofmt_xdate()
            fig.set_size_inches(11,8)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%y"))
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fancybox=True, shadow=True)
        plt.yscale(yscale)
        csv_filename, svg_filename = Statistics.get_csv_svg_filenames_and_check_they_do_not_exist(image_path)
        if len(dates) > 1:
            plt.savefig(svg_filename, bbox_inches="tight")
        plt.close('all')
        with open(csv_filename, 'w') as f:
            if is_dates:
                date_strings = [d.strftime("%d/%m/%y") for d in dates]
            else:
                date_strings = ["{:.4f}".format(d) for d in dates]
            f.write(",".join(["Dates:"] + date_strings))
            for data in datas:
                data_strings = [str(val) for val in data['data']]
                f.write("\n")
                f.write(",".join([data['props']['label']] + data_strings))

    @staticmethod
    def plot_with_err(image_path, dates, datas, datas_error, background_stripes=None, is_dates=True, clear_graph=True):
        """
        same as plot, but with datas_error which is an "error" for each data and each date.
        Can be used for plotting std or confidence.
        """
        if background_stripes is None:
            background_stripes = []
        yscales = []
        for data in datas:
            if 'yscale' in data['props']:
                yscales.append(data['props']['yscale'])
                del data['props']['yscale']
            else:
                yscales.append('linear')
        assert all([yscale == yscales[0] for yscale in yscales]), "All yscales must be equal!"
        yscale = yscales[0]
        if clear_graph:
            plt.clf()
            plt.figure(figsize=(10, 10), dpi=80)
        if is_dates:
            npdates = [mdates.date2num(d) for d in dates]
        else:
            npdates = dates
        for data, err in zip(datas, datas_error):
            new_dates, new_data_and_err = Statistics.remove_nans_from_plot(npdates, [data['data'], err['data']])
            new_data, new_err = new_data_and_err
            upper_err_curve = [d + e for d, e in zip(new_data, new_err)]
            lower_err_curve = [d - e for d, e in zip(new_data, new_err)]
            plt.plot(new_dates, new_data, **data['props'])
            plt.plot(new_dates, upper_err_curve, **err['props'])
            plt.plot(new_dates, lower_err_curve, **err['props'])
            plt.fill_between(new_dates, lower_err_curve, upper_err_curve, alpha=0.5)
            plt.grid(color='pink', linestyle='--', linewidth=0.5)
        for stripe in background_stripes:
            plt.axvspan(
                mdates.date2num(stripe.start),
                mdates.date2num(stripe.end),
                facecolor=stripe.color,
                alpha=0.2,
                label=stripe.label
            )
        if is_dates:
            plt.gcf().autofmt_xdate()
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%y"))
        plt.title(os.path.basename(image_path))
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fancybox=True, shadow=True)
        plt.yscale(yscale)
        csv_filename, svg_filename = Statistics.get_csv_svg_filenames_and_check_they_do_not_exist(image_path)
        plt.savefig(svg_filename, bbox_inches="tight")
        with open(csv_filename, 'w') as f:
            if is_dates:
                date_strings = ["Dates:"] + [d.strftime("%d/%m/%y") for d in dates]
            else:
                date_strings = [""] + ["{:.4f}".format(d) for d in dates]
            f.write(",".join(date_strings))
            for data, err in zip(datas, datas_error):
                new_dates, new_data_and_err = Statistics.remove_nans_from_plot(npdates, [data['data'], err['data']])
                new_data, new_err = new_data_and_err
                mean_data_strings = [str(val) for val in new_data]
                stdev_data_strings = [str(val) for val in new_err]
                f.write("\n")
                f.write(",".join(["Mean of " + data['props']['label']] + mean_data_strings))
                f.write("\n")
                f.write(",".join(["Standard deviation of " + data['props']['label']] + stdev_data_strings))

class AgeGroupDiseaseStatePlot(DataToPlot):
    """
    plot all the people in a certain age_group which their disease state in disease_states
    """
    def __init__(self, age_group, disease_states, props, is_integral=True):
        self.age_group = age_group
        self.disease_states = disease_states
        self.props = props
        self.is_integral = is_integral
        self.infection_data = None

    def property_to_count(self, redacted_person):
        if self.age_group is None:
            return (redacted_person.disease_state in self.disease_states)

        return (
            (self.age_group[0] <= redacted_person.age <= self.age_group[1]) and
            (redacted_person.disease_state in self.disease_states)
        )

class InfectionPlot(DataToPlot):
    """
    plot all the infections which attibute_name
    ("infection_env_short", "infection_env_long", "infected_age", "infector_age" or "infector_disease_state")
    in filter_group.
    """
    def __init__(self, attibute_name, filter_group, is_age, props, is_integral=False):
        self.attibute_name = attibute_name
        self.filter_group = filter_group
        self.props = props
        self.is_integral = is_integral
        self.is_age = is_age
        self.infection_data = attibute_name

    def property_to_count(self, att):
        if self.is_age:
            return att is not None and self.filter_group[0] <= att < self.filter_group[1]
        return (att in self.filter_group)

def make_age_and_state_datas_to_plot(
    age_groups=((0, 19), (20, 39), (40, 59), (60, 79), (80, 99)),
    disease_state_groups=(
        ("infected", (
                DiseaseState.SYMPTOMATICINFECTIOUS,
                DiseaseState.INCUBATINGPOSTLATENT,
                DiseaseState.LATENT,
                DiseaseState.CRITICAL,
                DiseaseState.ASYMPTOMATICINFECTIOUS
            )
         ),
        ("susceptible", (DiseaseState.SUSCEPTIBLE,)),
        ("Immune", (DiseaseState.IMMUNE,))
    ),
    additional_props=None,
    is_integral=True
):
    """
    Make a list of 'DataToPlot' objects that define desired graphs
    :param age_groups: The list of age-segments by which to stratify the data (for instance, [(0, 49), (50, 99)] means
    make graphs for people aged 0-49 and graphs for people aged 50-99. If you do not wish to stratify, put [None].
    :param disease_state_groups: The sets of disease states by which to group people.
    :param additional_props: A dict of additional graphical props to send to matplotlib (None means {}).
    For instance: 'yscale': 'log'
    :return: The desired list of 'DataToPlot' objects.
    """
    if additional_props is None:
        additional_props = {}
    colors = ['hotpink', 'lightskyblue', 'mediumpurple', 'mediumturquoise', 'darkorange']
    lines = [(0,(5,10)), (0,(1,1)), (0,(5,1)),(0,(3,1,1,1,1,1)), (0,(10,0))]
    ret = []
    color_cycle = cycle(colors)
    for age_group in age_groups:
        curr_color = next(color_cycle)
        line_cycle = cycle(lines)
        for label, disease_states in disease_state_groups:
            if len(age_groups) == 1:  # age_groups = [None] means do not age-stratify
                curr_color = next(color_cycle)
            if age_group is not None:
                new_label = label + "_%s_%s" % age_group
            else:
                new_label = label
            props = {'label': new_label, 'color': curr_color, 'linestyle': next(line_cycle)}
            props.update(additional_props)
            ret.append(
                AgeGroupDiseaseStatePlot(
                    age_group, disease_states,
                    props, is_integral=is_integral
                )
            )
    return ret

def make_infections_age_datas_to_plot(
    age_groups=((0, 19), (20, 39), (40, 59), (60, 79), (80, 99)),
    additional_props=None
):
    if additional_props is None:
        additional_props = {}
    colors = ['hotpink', 'lightskyblue', 'mediumpurple', 'mediumturquoise', 'darkorange']
    ret = []
    color_cycle = cycle(colors)
    for age_group in age_groups:
        curr_color = next(color_cycle)
        label = "%s_%s" % age_group
        props = {'label': label, 'color': curr_color}
        props.update(additional_props)
        ret.append(
            InfectionPlot(
                "infector_age", age_group, True,
                props, is_integral=False
            )
        )

    return ret

def make_infections_infector_state_datas_to_plot(
    disease_state_groups=(
        ("incubating", (DiseaseState.INCUBATINGPOSTLATENT,)),
        ("symptomatic", (DiseaseState.SYMPTOMATICINFECTIOUS,)),
        ("asymptomatic", (DiseaseState.ASYMPTOMATICINFECTIOUS,)),
        ("critical", (DiseaseState.CRITICAL,))
    ), additional_props=None
):
    if additional_props is None:
        additional_props = {}
    colors = ['hotpink', 'lightskyblue', 'mediumpurple', 'mediumturquoise', 'darkorange']
    ret = []
    color_cycle = cycle(colors)
    for label, dis_group in disease_state_groups:
        curr_color = next(color_cycle)
        props = {'label': label, 'color': curr_color}
        props.update(additional_props)
        ret.append(
            InfectionPlot(
                "infector_disease_state", dis_group, False,
                props, is_integral=False
            )
        )

    return ret

def make_infections_environment_datas_to_plot(
    environments_groups=tuple(
        (env, (env,)) for env in (
            'workplace',
            'neighborhood_community',
            'city_community',
            'household',
            'school',
            'initial_group'
        )
    ), additional_props=None
):
    """
    A default constructor to plot the infection enviroment by time
    """
    if additional_props is None:
        additional_props = {}
    colors = ['hotpink', 'lightskyblue', 'mediumpurple', 'mediumturquoise', 'darkorange']
    ret = []
    color_cycle = cycle(colors)
    for label, env_group in environments_groups:
        curr_color = next(color_cycle)
        props = {'label': label, 'color': curr_color}
        props.update(additional_props)
        ret.append(
            InfectionPlot(
                "infection_env_short", env_group, False,
                props, is_integral=False
            )
        )

    return ret


def fill_in_dates(list_of_data_and_daterange, edge_values, dates=None):
    """
    takes multiple dates range and fill the gaps to the maximal range

    list_of_data_and_daterange = the datas
    edge_values = bool of whether fill with the first/last value or with NaN
    dates = optional custom final date range (if None the range will be the maximal range)
    """
    if dates is None:
        min_date = min(s[1][0] for s in list_of_data_and_daterange)
        max_date = max(s[1][1] for s in list_of_data_and_daterange)
    else:
        min_date, max_date = dates[0], dates[-1]
    aligned_filled_data = []
    for s, date_range in list_of_data_and_daterange:
        initial_delay = (date_range[0] - min_date).days
        post_delay = (max_date - date_range[1]).days
        new_data = [s[0] if edge_values else NaN] * initial_delay + \
                   s + \
                   [s[-1] if edge_values else NaN] * post_delay
        aligned_filled_data.append(new_data)
    return aligned_filled_data


def max_date_range(stats, max_num_days=None) -> List[datetime.datetime]:
    """
    Returns a list of the days between stats.min_date and stats.max_date
    If max_num_days is provided, the list is cut to be at most that long
    (with the same start date)
    :param stats: A Statistics object
    :param max_num_days: The maximum number of days we wish to use
    (non-inclusive and the output is inclusive, so there's a minus one here)
    :return: A list of days, from s.min_date until s.max_date and no more than
    max_num_days
    """
    min_date = min(s.min_date for s in stats)
    max_date = max(s.max_date for s in stats)
    num_days = (max_date - min_date).days
    if max_num_days is not None:
        num_days = min(max_num_days, num_days)
    return [min_date + timedelta(i) for i in range(num_days + 1)]


def get_mean_and_confidence_from_statistics(stats_files, datas_to_plot, name, outdir):
    """
    takes dumps of Statistics and compute the mean and confidence for each data in datas_to_plot
    the typical input is files from  multiple runs of the same simulation
    """
    all_stats = [
        Statistics.load(file_name) for file_name in
        stats_files
    ]
    list_of_samples_with_props = []
    longest_date_range = max_date_range(all_stats)
    for data_to_plot in datas_to_plot:
        data_with_ranges = []
        for stat in all_stats:
            days_data = stat.sum_days_data(data_to_plot.property_to_count, data_to_plot.is_integral, data_to_plot.infection_data)
            data_with_ranges.append((days_data, (stat.min_date, stat.max_date)))
        aligned_data = fill_in_dates(data_with_ranges, True)
        list_of_samples_with_props.append({
            'samples': aligned_data,
            'props': data_to_plot.props
        })
    compute_and_plot_mean_stddev_confidence(longest_date_range, [list_of_samples_with_props], outdir, name)


def get_r_mean_and_confidence_from_statistics(stats_files, name, outdir):
    """
    takes dumps of Statistics and compute the mean and confidence for r as function of time
    the typical input is files from multiple runs of the same simulation
    """
    colors = ['hotpink', 'lightskyblue', 'mediumpurple', 'mediumturquoise', 'darkorange']
    color_cycle = cycle(colors)

    all_stats = [
        Statistics.load(file_name) for file_name in
        stats_files
    ]
    longest_date_range = max_date_range(all_stats)

    to_plot_array = []

    for key in ["smoothed_avg_r0", "avg_r0", "instantaneous_r"]:
        smoothed_r0_avg_with_date_range = [
            (s.get_r0_data()[key], (s.get_r0_data()['dates'][0], s.get_r0_data()['dates'][-1]))
            for s in all_stats if s.get_r0_data() is not None
        ]
        aligned_data = fill_in_dates(smoothed_r0_avg_with_date_range, False, longest_date_range)
        to_plot = [
            {
                'samples': aligned_data,
                'props': {'label': key, 'color': next(color_cycle)}
             }
        ]
        to_plot_array.append(to_plot)
    compute_and_plot_mean_stddev_confidence(longest_date_range, to_plot_array, outdir, name + "_r_data_" + key,
                                            all_stats[0].make_background_stripes())


def get_multiple_stats_summary_file(stats_files, name, outdir, shortened=False):
    """
    write a summary of multiple runs of the same simulation whose simulation dumps are stored in stats_files
    """
    all_stats = [
        Statistics.load(file_name) for file_name in
        stats_files
    ]
    Statistics.write_multiple_stats_summary_file(all_stats, outdir, name, shortened=shortened)


def apply_unless_all_nans(func, arr):
    if all(isnan(arr)):
        return NaN
    else:
        return func(arr)


def compute_and_plot_mean_stddev_confidence(dates, list_of_all_samples_with_props, outdir, name, background_stripes_data = None):
    """
    process the samples in list_of_all_samples_with_props to compute the mean, the stddev and the confidence and plot them
    """
    exp_data = []
    std_data = []
    confidence_data = []
    first_graph = True
    for r_graphs in list_of_all_samples_with_props:
        for samples_with_props in r_graphs:
            samples = samples_with_props['samples']
            props = samples_with_props['props']
            std_conf_props = props.copy()
            std_conf_props.pop('label')
            expectation = [apply_unless_all_nans(nanmean, nums) for nums in zip(*samples)]
            std = [apply_unless_all_nans(nanstd, nums) for nums in zip(*samples)]
            confidence = [
                apply_unless_all_nans(nanstd, nums) / sqrt(apply_unless_all_nans(len, nums))
                for nums in zip(*samples)
            ]

            exp_data.append({
                'data': expectation,
                'props': props
            })
            std_data.append({
                'data': std,
                'props': std_conf_props
            })
            confidence_data.append({
                'data': confidence,
                'props': std_conf_props
            })
        #Statistics.plot_with_err(os.path.join(outdir, name + '_exp_std'), dates, exp_data, std_data,
        #                         background_stripes_data, first_graph)
        first_graph = False
        Statistics.plot_with_err(os.path.join(outdir, name + '_exp_confidence'), dates, exp_data,
                                 confidence_data, background_stripes_data, True)


def compute_r_from_statistics(param_and_stats_files, max_num_days, name, outdir):
    """
    plot r value as a function of params
    """
    data = []
    params = []
    for param, stats_files in param_and_stats_files:
        all_stats = [
            Statistics.load(file_name) for file_name in
            stats_files
        ]

        smoothed_r0_avg_with_date_range = [
            (s.get_r0_data()['smoothed r'], (s.get_r0_data()['dates'][0], s.get_r0_data()['dates'][-1]))
            for s in all_stats if s is not None
        ]
        longest_date_range = max_date_range(all_stats, max_num_days=max_num_days)
        aligned_data = fill_in_dates(smoothed_r0_avg_with_date_range, False, longest_date_range)
        expectation = [apply_unless_all_nans(nanmean, nums) for nums in zip(*aligned_data)]
        std = [apply_unless_all_nans(nanstd, nums) for nums in zip(*aligned_data)]
        confidence = [
            apply_unless_all_nans(nanstd, nums) / sqrt(apply_unless_all_nans(len, nums))
            for nums in zip(*aligned_data)
        ]
        data.append((expectation, std, confidence))
        params.append(param)
    exp_data = []
    std_data = []
    confidence_data = []
    for day in range(max_num_days + 1):
        props = {'label': "smoothed r"}
        exp_data.append({
            'data': [elem[0][day] for elem in data],
            'props': props
        })
        std_data.append({
            'data': [elem[1][day] for elem in data],
            'props': props
        })
        confidence_data.append({
            'data': [elem[2][day] for elem in data],
            'props': props
        })

    #TODO NOAM: why std_data and confidence_data are all 0?
    Statistics.plot_with_err(os.path.join(outdir, name + '_exp_std'), params, exp_data, std_data, is_dates=True)
    Statistics.plot_with_err(os.path.join(outdir, name + '_exp_confidence'), params, exp_data, confidence_data, is_dates=True)
