from typing import List

import pandas
import seaborn
from functional import seq
import numpy as np
import matplotlib
from enum import Enum


def select_daily_graph_colors(vaccination_strategy, vaccination_order):
    #colors = ['darkviolet', 'darkviolet', 'darkviolet', 'darkviolet']
    colors = ['hotpink', 'lightskyblue', 'mediumpurple', 'mediumturquoise', 'darkorange']
    lines = [(0, (5, 0)), (0, (1, 1)), (0, (8, 8)), (0, (5, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (10, 0))]

    if vaccination_order == "DESCENDING":
        line_color = colors[0]
    else:
        line_color = colors[1]

    if vaccination_strategy == "GENERAL":
        line_pattern = lines[0]
    elif vaccination_strategy == "BY_NEIGHBORHOOD":
        line_pattern = lines[1]
    else:
        line_pattern = lines[2]

    line_label = f"{vaccination_strategy} - {vaccination_order}"

    return line_color, line_pattern, line_label


class DailyGraphType(Enum):
    INFECTED = 1
    CRITICAL = 2
    HOSPITALISED = 3


def draw_daily_graphs(df, ax, graph_type: DailyGraphType):
    def draw_graph(data_mean_key: str, data_stderr_key: str):
        max_y = max([max(x) for x in df[data_mean_key]])
        for key, daily_results in enumerate(df[data_mean_key]):
            std_err = df[data_stderr_key].tolist()[key]
            color, line_style, label = select_daily_graph_colors(df["vaccination_strategy"].to_list()[key],
                                                                 df["order"].to_list()[key])
            ax.plot(range(len(daily_results)), daily_results, label=label,
                    color=color,
                    linestyle=line_style)
            # draw the confidence interval
            lower_range = (np.array(daily_results) - np.array(std_err)).tolist()
            upper_range = (np.array(daily_results) + np.array(std_err)).tolist()
            ax.fill_between(range(len(daily_results)), lower_range, upper_range, color=color, alpha=.1)

            ax.set_xlim(0, len(daily_results))
            ax.set_ylim(0, max_y)
            ax.legend(prop={"size": 8})
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(16)

    if graph_type == DailyGraphType.INFECTED:
        draw_graph("infected_cumulative_mean", "infected_cumulative_stdev")
    elif graph_type == DailyGraphType.CRITICAL:  # plot daily critical cases
        draw_graph("critical_cumulative_mean", "critical_cumulative_stdev")
    elif graph_type == DailyGraphType.HOSPITALISED:
        draw_graph("total_hospitalized_mean", "total_hospitalized_stderr")
    else:
        raise Exception("UNKNOWN DAILY GRAPH SELECTED: " + graph_type)


def draw_heatmap(ax, categories: pandas.DataFrame, for_total_infections: bool = True):
    number_of_categories = categories.__len__()
    number_of_strategies = categories.head(2)["vaccination_strategy"].count()
    d = np.ndarray(shape=(number_of_categories, number_of_strategies), dtype=float)
    for category in categories:
        df = category[1]
        data = np.array(df["total_infected_in_the_community"])[:][-1]
        for key, r_case_reproduction_number in enumerate(df["total_infected_in_the_community"]):
            pass
        seaborn.set_theme()
        x = range(len(df["vaccination_strategy"].to_list()))
        seaborn.heatmap()
        ax.legend(prop={"size": 8})


def draw_daily_r_graph_2(ax, df, use_r_instantaneous):
    for key, r_case_reproduction_number in enumerate(df["r_case_reproduction_number"]):
        r_instantaneous = df["r_instantaneous"].to_list()[key]
        r_instantaneous_stderr = df["r_instantaneous_stderr"].to_list()[key]
        r_case_reproduction_number_stderr = df["r_case_reproduction_number_stderr"].to_list()[key]
        if r_instantaneous is None:
            return
        color, line_style, label = select_daily_graph_colors(df["vaccination_strategy"].to_list()[key],
                                                             df["order"].to_list()[key])

        starting_day = 14
        if use_r_instantaneous:
            x = range(starting_day, len(r_instantaneous))
            ax.plot(x, r_instantaneous[starting_day:], label=label,
                    color=color,
                    linestyle=line_style)
            lower_range = (np.array(r_instantaneous[starting_day:]) - np.array(r_instantaneous_stderr[starting_day:])).tolist()
            upper_range = (np.array(r_instantaneous[starting_day:]) + np.array(r_instantaneous_stderr[starting_day:])).tolist()
            ax.fill_between(x, lower_range, upper_range, color=color, alpha=.1)
            ax.set_xlim(min(x), max(x))

        else:
            x = range(starting_day, len(r_case_reproduction_number))
            ax.plot(x, r_case_reproduction_number[starting_day:], label=label,
                    color=color,
                    linestyle=line_style)
            lower_range = (
                        np.array(r_case_reproduction_number[starting_day:]) -
                        np.array(r_case_reproduction_number_stderr[starting_day:])).tolist()
            upper_range = (
                        np.array(r_case_reproduction_number[starting_day:]) +
                        np.array(r_case_reproduction_number_stderr[starting_day:])).tolist()
            ax.fill_between(x, lower_range, upper_range, color=color, alpha=.1)
        ax.plot(x, [1] * len(x), color="lightgreen")
        ax.set_xlim(min(x), max(x))
        ax.set_ylim(bottom=0, top=1.5)
        ax.legend(prop={"size": 8})
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(16)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def calculate_r0_instantaneous(infected_per_day: List[int], w) -> List[float]:
    """calculates the instantaneous r based on a look back of look_back_days days"""

    r = []
    look_back_days = 0
    for i in range(len(infected_per_day)):
        look_back_days = w.w_len()
        if i <= look_back_days:
            r.append(0)
        else:
            r_today = infected_per_day[i] \
                      / max(seq([w.get_w(look_back) *
                                 infected_per_day[i - look_back]
                                 for look_back in range(1, look_back_days)]).sum(), 1)
            r.append(r_today)

    return r
