from tabulate import tabulate
import pandas as pd
import scipy.stats
import numpy as np
import sys
import math


def ci_to_p_value(low, high, mean_difference, SE):
    # SE = (high - low) / (2 * 1.96)
    z = mean_difference / SE
    return math.exp(-0.717 * z - 0.416 * (z ** 2))


def plot_mixed_strategy_confidence_interval(c, root_path) -> None:
    """
    calculate the effect of applying different strategies for different cities
    :param c: a list of tuple [(strategy, DataFrame)]
    :return: None
    """
    index = 0
    bootstrap_results = pd.DataFrame(columns=['Strategy 1', 'Strategy 2', 'Confidence Low', 'Confidence High', 'Mean', 'P Value', 'T Test - P Value'])
    d = c.head(200)

    # pick specific combinations to compare
    combinations = [
        [
            {"city": "Holon", "vaccination_strategy": "NEIGHBORHOOD", "order": "DESCENDING",
             "intervention": "hh_isolation"},
            {"city": "Benei Brak", "vaccination_strategy": "GENERAL", "order": "DESCENDING",
             "intervention": "hh_isolation"},

            {"city": "Holon", "vaccination_strategy": "GENERAL", "order": "DESCENDING", "intervention": "hh_isolation"},
            {"city": "Benei Brak", "vaccination_strategy": "GENERAL", "order": "DESCENDING",
             "intervention": "hh_isolation"},
        ],

        [
            {"city": "Holon", "vaccination_strategy": "NEIGHBORHOOD", "order": "ASCENDING",
             "intervention": "hh_isolation"},
            {"city": "Benei Brak", "vaccination_strategy": "NEIGHBORHOOD", "order": "DESCENDING",
             "intervention": "hh_isolation"},

            {"city": "Holon", "vaccination_strategy": "NEIGHBORHOOD", "order": "ASCENDING",
             "intervention": "hh_isolation"},
            {"city": "Benei Brak", "vaccination_strategy": "NEIGHBORHOOD", "order": "ASCENDING",
             "intervention": "hh_isolation"},
        ],

        [
            {"city": "Holon", "vaccination_strategy": "NEIGHBORHOOD", "order": "ASCENDING",
             "intervention": "asymptomatic_detection"},
            {"city": "Benei Brak", "vaccination_strategy": "NEIGHBORHOOD", "order": "DESCENDING",
             "intervention": "asymptomatic_detection"},

            {"city": "Holon", "vaccination_strategy": "NEIGHBORHOOD", "order": "DESCENDING",
             "intervention": "asymptomatic_detection"},
            {"city": "Benei Brak", "vaccination_strategy": "NEIGHBORHOOD", "order": "DESCENDING",
             "intervention": "asymptomatic_detection"},
        ],

        [
            {"city": "Holon", "vaccination_strategy": "NEIGHBORHOOD", "order": "ASCENDING",
             "intervention": "asymptomatic_detection"},
            {"city": "Benei Brak", "vaccination_strategy": "NEIGHBORHOOD", "order": "DESCENDING",
             "intervention": "asymptomatic_detection"},

            {"city": "Holon", "vaccination_strategy": "GENERAL", "order": "DESCENDING",
             "intervention": "asymptomatic_detection"},
            {"city": "Benei Brak", "vaccination_strategy": "GENERAL", "order": "DESCENDING",
             "intervention": "asymptomatic_detection"},
        ],

    ]

    for combination in combinations:
        pair_1_city_1 = d[(d.city == combination[0]["city"]) &
                          (d.vaccination_strategy == combination[0]["vaccination_strategy"]) &
                          (d.order == combination[0]["order"]) &
                          (d.intervention == combination[0]["intervention"])]
        pair_1_city_2 = d[(d.city == combination[1]["city"]) &
                          (d.vaccination_strategy == combination[1]["vaccination_strategy"]) &
                          (d.order == combination[1]["order"]) &
                          (d.intervention == combination[1]["intervention"])]
        pair_2_city_1 = d[(d.city == combination[2]["city"]) &
                          (d.vaccination_strategy == combination[2]["vaccination_strategy"]) &
                          (d.order == combination[2]["order"]) &
                          (d.intervention == combination[2]["intervention"])]
        pair_2_city_2 = d[(d.city == combination[3]["city"]) &
                          (d.vaccination_strategy == combination[3]["vaccination_strategy"]) &
                          (d.order == combination[3]["order"]) &
                          (d.intervention == combination[3]["intervention"])]

        # calculate for hospitalisation
        pair_1 = (np.array(pair_1_city_1.total_hospitalized_samples.to_list()) +
                  np.array(pair_1_city_2.total_hospitalized_samples.to_list())).flatten()
        pair_2 = (np.array(pair_2_city_1.total_hospitalized_samples.to_list()) + \
                  np.array(pair_2_city_2.total_hospitalized_samples.to_list())).flatten()
        try:
            res = scipy.stats.bootstrap(data=[(pair_1 - pair_2)], statistic=np.mean)
        except ValueError as e:
            print(f"plot_mixed_strategy_confidence_interval() Exception! e={e}")
            return
        pvalue_res = scipy.stats.ttest_ind(pair_1, pair_2, equal_var=True)
        pvalue = ci_to_p_value(
            low=res.confidence_interval.low,
            high=res.confidence_interval.high,
            mean_difference=(pair_1 - pair_2).mean(),
            SE=res.standard_error)
        bootstrap_results = pd.concat([bootstrap_results,
           pd.DataFrame.from_records([{
               'Strategy 1': f'HOSPITALISATION: {combination[0]["city"]}: {combination[0]["intervention"]} - {combination[0]["vaccination_strategy"]} - {combination[0]["order"]}' + "\n" +
                             f'{combination[1]["city"]}: {combination[1]["intervention"]} - {combination[1]["vaccination_strategy"]} - {combination[1]["order"]}',
               'Strategy 2': f'HOSPITALISATION: {combination[2]["city"]}: {combination[2]["intervention"]} - {combination[2]["vaccination_strategy"]} - {combination[2]["order"]}' + "\n" +
                             f'{combination[3]["city"]}: {combination[3]["intervention"]} - {combination[3]["vaccination_strategy"]} - {combination[3]["order"]}',
               'Confidence Low': res.confidence_interval.low,
               'Confidence High': res.confidence_interval.high,
               'Mean': (pair_1 - pair_2).mean(),
               'P Value': pvalue,
               'T Test - P Value': pvalue_res.pvalue,
           }], index=[index])])
        index += 1

        # do the same for Infections
        pair_1 = (np.array(pair_1_city_1.total_infected_samples.to_list()) +
                  np.array(pair_1_city_2.total_infected_samples.to_list())).flatten()
        pair_2 = (np.array(pair_2_city_1.total_infected_samples.to_list()) +
                  np.array(pair_2_city_2.total_infected_samples.to_list())).flatten()
        res = scipy.stats.bootstrap(data=[(pair_1 - pair_2)], statistic=np.mean)
        pvalue_res = scipy.stats.ttest_ind(pair_1, pair_2, equal_var=True)
        pvalue = ci_to_p_value(
            low=res.confidence_interval.low,
            high=res.confidence_interval.high,
            mean_difference=(pair_1 - pair_2).mean(),
            SE=res.standard_error)
        bootstrap_results = pd.concat([bootstrap_results, pd.DataFrame.from_records([{
            'Strategy 1': f'INFECTIONS: {combination[0]["city"]}: {combination[0]["intervention"]} - {combination[0]["vaccination_strategy"]} - {combination[0]["order"]}' + "\n" +
                          f'{combination[1]["city"]}: {combination[1]["intervention"]} - {combination[1]["vaccination_strategy"]} - {combination[1]["order"]}',
            'Strategy 2': f'INFECTIONS: {combination[2]["city"]}: {combination[2]["intervention"]} - {combination[2]["vaccination_strategy"]} - {combination[2]["order"]}' + "\n" +
                          f'{combination[3]["city"]}: {combination[3]["intervention"]} - {combination[3]["vaccination_strategy"]} - {combination[3]["order"]}',
            'Confidence Low': res.confidence_interval.low,
            'Confidence High': res.confidence_interval.high,
            'Mean': (pair_1 - pair_2).mean(),
            'P Value': pvalue,
            'T Test - P Value': pvalue_res.pvalue,
        }], index=[index])], ignore_index=True)
        index += 1

    # print(f"\n{combinations}")
    # print(tabulate(bootstrap_results, headers='keys', tablefmt='fancy_grid'))
    bootstrap_results.to_csv(f'{root_path}/outputs/{sys.argv[1]}/bootstrap_combinations.csv')
    bootstrap_results.to_html(f'{root_path}/outputs/{sys.argv[1]}/bootstrap_combinations.html')
