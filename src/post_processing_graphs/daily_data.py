import math
import os
import statistics
from collections import namedtuple
from typing import List
import pandas

import numpy as np
from functional import seq, pipeline


def path_to_city(path):
    if "Holon" in path:
        return "Holon"
    else:
        return "Benei Brak"


def get_sample_line(root_path, sample, file_name, line_name, as_is=False, factor_to_100_000=True):
    city = path_to_city(root_path)
    per_100_000_factor = per_100_000(city)
    filename = f"{root_path}/sample_{sample}/{file_name}"
    if os.path.isfile(filename):
        data = pandas.read_csv(filename)
        line = data[data[data.columns[0]] == line_name]
        if as_is:
            result = line.keys().to_list()[1:]
        else:
            line = line.to_numpy().flatten()[1:].astype('float')
            result = np.clip(line, a_min=0.0, a_max=None)
            if factor_to_100_000:
                result = [x * per_100_000_factor for x in result]
        return result
    return None


def get_daily_column(root_path, sample, column_name, factor_to_100_000=True):
    city = path_to_city(root_path)
    per_100_000_factor = per_100_000(city)
    filename = f"{root_path}/sample_{sample}/daily_delta.csv"
    if os.path.isfile(filename):
        data = pandas.read_csv(filename)
        column = data[column_name].to_numpy().astype('float')
        column = np.clip(column, a_min=0.0, a_max=None)
        if factor_to_100_000:
            column = [x * per_100_000_factor for x in column]
        return column
    return None


def add_missing_dates(partial_data, all_dates, partial_dates, default_data=0):
    all_data_len = len(all_dates)
    result = [0.0] * all_data_len

    for i in range(all_data_len):
        if all_dates[i] in partial_dates:
            partial_index = partial_dates.index(all_dates[i])
            result[i] = partial_data[partial_index]
        else:
            # print(f"add_missing_dates() date {all_dates[i]} is missing")
            result[i] = default_data
    return result


def calculate_r(root_path: str, max_iterations=None):
    r_instantaneous = None
    r_case_reproduction_cases = None
    if max_iterations is None:
        max_iterations = 1000
    # since the r0 csv files do not cover all the dates, we have to complete them with zeros in the missing dates...
    all_dates = get_sample_line(root_path, 0, "amit_graph_daily.csv", line_name="Dates:", as_is=True)
    for repetition in range(max_iterations):
        r0_dates = get_sample_line(root_path, repetition, "r0_data_amit_graph_integral.csv", line_name="Dates:",
                                   as_is=True)
        r_case_reproduction_cases_today = get_sample_line(root_path, repetition, "r0_data_amit_graph_integral.csv",
                                                          line_name="smoothed base reproduction number r",
                                                          factor_to_100_000=False)
        r_instantaneous_today = get_sample_line(root_path, repetition, "r0_data_amit_graph_integral.csv",
                                                line_name="instantaneous r",
                                                factor_to_100_000=False)

        if r_case_reproduction_cases_today is None or r_instantaneous_today is None:
            break

        r_case_reproduction_cases_today = add_missing_dates(r_case_reproduction_cases_today, all_dates, r0_dates)
        r_instantaneous_today = add_missing_dates(r_instantaneous_today, all_dates, r0_dates)

        if r_instantaneous is None:
            r_instantaneous = np.array([r_instantaneous_today])
            r_case_reproduction_cases = np.array([r_case_reproduction_cases_today])
        else:
            r_instantaneous = np.append(r_instantaneous, [r_instantaneous_today], axis=0)
            r_case_reproduction_cases = np.append(r_case_reproduction_cases, [r_case_reproduction_cases_today], axis=0)
    if r_case_reproduction_cases is None or r_instantaneous is None:
        return None, None
    number_of_repetitions = r_instantaneous.shape[0]
    r_instantaneous_average = r_instantaneous.mean(axis=0)
    r_instantaneous_stderr = (r_instantaneous.std(axis=0) / math.sqrt(number_of_repetitions)).tolist()
    r_case_reproduction_cases_average = r_case_reproduction_cases.mean(axis=0)
    r_case_reproduction_cases_stderr = (
            r_case_reproduction_cases.std(axis=0) / math.sqrt(number_of_repetitions)).tolist()
    return r_case_reproduction_cases_average, \
           r_case_reproduction_cases_stderr, \
           r_instantaneous_average, \
           r_instantaneous_stderr


def remove_outliers(data: List[float], method="stdev"):
    if method == "stdev":
        data_mean, data_std = statistics.mean(data), statistics.stdev(data)
        # identify outliers
        cut_off = data_std * 3
        lower, upper = data_mean - cut_off, data_mean + cut_off
    else:  # use percentile
        from numpy import percentile
        q25, q75 = percentile(data, 25), percentile(data, 75)
        iqr = q75 - q25
        cut_off = iqr * 1.5
        lower, upper = q25 - cut_off, q75 + cut_off

    outliers_removed = [x for x in data if lower < x < upper]
    outliers = [x for x in data if x > upper or x < lower]
    return outliers_removed, outliers


DAILY_INFO = namedtuple("DAILY_INFO", ("number_of_samples",
                                       "daily_infection",
                                       "daily_critical_cases",
                                       "total_hospitalized_mean",
                                       "total_hospitalized_stderr",
                                       "total_hospitalized_samples",
                                       "total_infected_in_the_community", "total_infected_in_the_community_stderr",
                                       "total_critical_in_the_community", "total_critical_in_the_community_stderr",
                                       "infected_cumulative_mean", "infected_cumulative_stdev",
                                       "total_infected_samples",
                                       "critical_cumulative_mean", "critical_cumulative_stdev",
                                       "infected_max_mean", "infected_max_stdev", "infected_max",
                                       "critical_max_mean", "critical_max_stdev", "critical_max",
                                       "daily_infection_stderr", "daily_critical_stderr",
                                       "r_case_reproduction_number", "r_instantaneous",
                                       "r_case_reproduction_number_stderr", "r_instantaneous_stderr",
                                       "num_repetitions"))


def stderr(data):
    return statistics.stdev(data) / math.sqrt(len(data))


def cumulative_sum(data):
    return np.cumsum(data)


def get_hospitalization_given_symptomatic(age):
    hospitalization_given_symptomatic_per_age = [0.0, 0.008, 0.008, 0.01, 0.019, 0.054, 0.151, 0.333, 0.618]
    return hospitalization_given_symptomatic_per_age[int(min(age // 10, 8))]


def per_100_000(city):
    """
    return the factor to multiply each sample to normalize the data per 100K
    """
    city_population = {"Holon": 189836, "Benei Brak": 185882}
    both_cities_population = city_population["Holon"] + city_population["Benei Brak"]
    if city == "Holon":
        city_size = city_population["Holon"]
    else:
        city_size = city_population["Benei Brak"]
    return 100_000.0 / city_size


def get_daily_info(root_path, max_days=None, max_iterations=None) -> DAILY_INFO:
    infected_cumulative_full_data = []
    max_infectious_in_community = []

    number_of_repetitions = 0
    daily_infection = None
    daily_infection_full_data = None

    if max_days is None:
        max_days = 1000

    if max_iterations is None:
        max_iterations = 1000

    for i in range(max_iterations):
        infected_today = get_daily_column(root_path, sample=i, column_name="infected_today")
        total_infected_today = get_sample_line(root_path, i, "amit_graph_integral.csv", line_name="infected_0_99")

        if infected_today is not None and total_infected_today is not None:
            infected_today = infected_today[0:max_days]
            total_infected_today = total_infected_today[0:max_days]

            if daily_infection is None:
                daily_infection = seq(infected_today)
                daily_infection_full_data = np.array([infected_today])
                infected_cumulative_full_data = np.array([cumulative_sum(infected_today)])
            else:
                daily_infection = daily_infection.zip(seq(infected_today)).map(lambda x: x[0] + x[1])
                daily_infection_full_data = np.append(daily_infection_full_data, [infected_today], axis=0)
                infected_cumulative_full_data = np.append(infected_cumulative_full_data,
                                                          [cumulative_sum([infected_today])], axis=0)
            max_infectious_in_community.append(max(total_infected_today))
            number_of_repetitions += 1
        else:
            break

    daily_infection = daily_infection.map(lambda x: x / number_of_repetitions)
    total_infected_in_community = infected_cumulative_full_data[:, -1]
    daily_infection = daily_infection[0:max_days]
    total_infected_in_community = total_infected_in_community[0:max_days]

    critical_cumulative = None
    critical_max = []
    daily_critical_cases = None
    daily_critical_full_data = None
    infected_cumulative_0_19_full_data = None
    infected_cumulative_20_59_full_data = None
    infected_cumulative_60_99_full_data = None

    for i in range(max_iterations):
        critical_today = get_daily_column(root_path, sample=i, column_name="CRITICAL")
        total_critical_today = get_sample_line(root_path, sample=i, file_name="amit_graph_integral.csv",
                                               line_name="critical_0_99")

        infected_0_19 = get_sample_line(root_path, sample=i, file_name="amit_graph.csv", line_name="infected_0_19")
        infected_20_59 = get_sample_line(root_path, sample=i, file_name="amit_graph.csv", line_name="infected_20_59")
        infected_60_99 = get_sample_line(root_path, sample=i, file_name="amit_graph.csv", line_name="infected_60_99")

        if critical_today is not None and total_critical_today is not None:
            if critical_cumulative is None:
                critical_cumulative = np.array([cumulative_sum(critical_today)])
            else:
                critical_cumulative = np.append(critical_cumulative, [cumulative_sum([critical_today])], axis=0)
            critical_max.append(max(total_critical_today))
        else:
            break

        if daily_critical_cases is None:
            daily_critical_cases = seq(critical_today)
            daily_critical_full_data = np.array([critical_today])
            infected_cumulative_0_19_full_data = np.array([cumulative_sum(infected_0_19[1:])])
            infected_cumulative_20_59_full_data = np.array([cumulative_sum(infected_20_59[1:])])
            infected_cumulative_60_99_full_data = np.array([cumulative_sum(infected_60_99[1:])])
        else:
            daily_critical_cases = daily_critical_cases.zip(seq(critical_today)).map(lambda x: x[0] + x[1])
            daily_critical_full_data = np.append(daily_critical_full_data, [critical_today], axis=0)
            infected_cumulative_0_19_full_data = np.append(infected_cumulative_0_19_full_data,
                                                           [cumulative_sum(infected_0_19[1:])], axis=0)
            infected_cumulative_20_59_full_data = np.append(infected_cumulative_20_59_full_data,
                                                            [cumulative_sum(infected_20_59[1:])], axis=0)
            infected_cumulative_60_99_full_data = np.append(infected_cumulative_60_99_full_data,
                                                            [cumulative_sum(infected_60_99[1:])], axis=0)

    # calculate the amount of hospitalized based on probability for an infected person to be hospitalized
    hospitalized_cumulative_full_data = \
        infected_cumulative_0_19_full_data * get_hospitalization_given_symptomatic((0 + 19) / 2) + \
        infected_cumulative_20_59_full_data * get_hospitalization_given_symptomatic((20 + 59) / 2) + \
        infected_cumulative_60_99_full_data * get_hospitalization_given_symptomatic((60 + 99) / 2)

    daily_critical_cases = daily_critical_cases.map(lambda x: x / number_of_repetitions)
    total_critical_in_community = critical_cumulative[:, -1]

    hospitalized_cumulative_full_data = hospitalized_cumulative_full_data[:, 0:max_days]
    daily_critical_cases = daily_critical_cases[0:max_days]
    total_critical_in_community = total_critical_in_community[0:max_days]
    critical_cumulative = critical_cumulative[:, 0:max_days]
    daily_critical_full_data = daily_critical_full_data[:, 0:max_days]
    critical_max = critical_max[0:max_days]

    r_case_reproduction_cases_mean, r_case_reproduction_cases_stderr, r_instantaneous_mean, r_instantaneous_stderr = \
        calculate_r(root_path, max_iterations=max_iterations)
    r_case_reproduction_cases_mean = r_case_reproduction_cases_mean[0:max_days]
    r_case_reproduction_cases_stderr = r_case_reproduction_cases_stderr[0:max_days]
    r_instantaneous_mean = r_instantaneous_mean[0:max_days]
    r_instantaneous_stderr = r_instantaneous_stderr[0:max_days]

    # infected_sum_no_outliers, infected_sum_outliers = remove_outliers(infected_cumulative_full_data, method="percentile")
    # critical_sum_no_outliers, critical_sum_outliers = remove_outliers(critical_cumulative, method="percentile")
    # infected_max_no_outliers, infected_max_outliers = remove_outliers(max_infectious_in_community,method="percentile")
    # critical_max_no_outliers, critical_max_outliers = remove_outliers(critical_max, method="percentile")
    # critical_max_no_outliers, critical_max_outliers = remove_outliers(critical_max, method="percentile")

    return DAILY_INFO(
        number_of_samples=number_of_repetitions,
        daily_infection=daily_infection.to_list(),

        total_hospitalized_mean=hospitalized_cumulative_full_data.mean(axis=0).tolist(),
        total_hospitalized_stderr=(hospitalized_cumulative_full_data.astype('float').std(axis=0) / math.sqrt(
            number_of_repetitions)).tolist(),
        # list of total-hospitalized numbers as we got in the different execution of the same scenario
        total_hospitalized_samples=hospitalized_cumulative_full_data[:, -1].tolist(),

        daily_critical_cases=daily_critical_cases.to_list(),

        total_infected_in_the_community=total_infected_in_community.tolist(),
        total_infected_in_the_community_stderr=(
                total_infected_in_community.std(axis=0) / math.sqrt(number_of_repetitions)).tolist(),

        total_critical_in_the_community=total_critical_in_community.tolist(),
        total_critical_in_the_community_stderr=(
                total_critical_in_community.std(axis=0) / math.sqrt(number_of_repetitions)).tolist(),

        infected_cumulative_mean=infected_cumulative_full_data.mean(axis=0).tolist(),
        infected_cumulative_stdev=(
                infected_cumulative_full_data.std(axis=0) / math.sqrt(number_of_repetitions)).tolist(),
        # list of total-infected numbers as we got in the different execution of the same scenario
        total_infected_samples=infected_cumulative_full_data[:, -1].tolist(),

        critical_cumulative_mean=critical_cumulative.mean(axis=0).tolist(),
        critical_cumulative_stdev=(critical_cumulative.std(axis=0) / math.sqrt(number_of_repetitions)).tolist(),

        infected_max=max_infectious_in_community,
        infected_max_mean=statistics.mean(max_infectious_in_community),
        infected_max_stdev=stderr(max_infectious_in_community),
        critical_max=critical_max,
        critical_max_mean=statistics.mean(critical_max),
        critical_max_stdev=stderr(critical_max),
        daily_infection_stderr=np.apply_along_axis(stderr, 0, daily_infection_full_data).tolist(),
        daily_critical_stderr=np.apply_along_axis(stderr, 0, daily_critical_full_data).tolist(),
        r_case_reproduction_number=r_case_reproduction_cases_mean,
        r_case_reproduction_number_stderr=r_case_reproduction_cases_stderr,
        r_instantaneous=r_instantaneous_mean,
        r_instantaneous_stderr=r_instantaneous_stderr,
        num_repetitions=hospitalized_cumulative_full_data.shape[0]
    )
