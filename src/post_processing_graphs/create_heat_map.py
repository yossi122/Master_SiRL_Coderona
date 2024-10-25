import sys

from matplotlib import pyplot
import seaborn as sns
import pandas as pd
import numpy as np
import os


def draw_heatmap(ax, df: pd.DataFrame):
    sns.set_theme()
    sns.set(font_scale=1.4)
    sns.heatmap(data=df, ax=ax, cmap="Spectral_r", annot=True, fmt=",.0f")

    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)
    # ax.legend()


def get_total(df: pd.DataFrame,
              is_infection: bool,
              city: str,
              intervention: str,
              order: str,
              vaccination_strategy: str) -> float:
    d = df[(df.city == city) & (df.intervention == intervention) & (df.order == order) & (
            df.vaccination_strategy == vaccination_strategy)]
    if is_infection:
        d = d.infected_cumulative_mean
    else:
        d = d.total_hospitalized_mean
    # its a cumulative graph. so the last item in the array, is the total over the whole period
    res = np.array(d.to_list()[0].replace('[', '').replace(']', '').split(',')).astype(float)[-1]
    return res


if __name__ == "__main__":
    plots = 8
    fig, ax = pyplot.subplots(plots, 1)
    fig.set_figwidth(22)
    fig.set_figheight(plots * 10)

    root_path = os.path.join(os.path.dirname(__file__), "../../")
    df = pd.read_csv(f"{root_path}/outputs/{sys.argv[1]}/results_including_city_True.csv")

    city_population = {"Holon": 189836, "Benei Brak": 185882}
    both_cities_population = city_population["Holon"] + city_population["Benei Brak"]

    total_infections_hh_isolation = [
        (get_total(df, True, "Benei Brak", "hh_isolation", "ASCENDING", "GENERAL"),
         get_total(df, True, "Benei Brak", "hh_isolation", "DESCENDING", "GENERAL"),
         get_total(df, True, "Benei Brak", "hh_isolation", "ASCENDING", "NEIGHBORHOOD"),
         get_total(df, True, "Benei Brak", "hh_isolation", "DESCENDING", "NEIGHBORHOOD")),  # Bnei Brak

        (get_total(df, True, "Holon", "hh_isolation", "ASCENDING", "GENERAL"),
         get_total(df, True, "Holon", "hh_isolation", "DESCENDING", "GENERAL"),
         get_total(df, True, "Holon", "hh_isolation", "ASCENDING", "NEIGHBORHOOD"),
         get_total(df, True, "Holon", "hh_isolation", "DESCENDING", "NEIGHBORHOOD"))]  # Holon

    total_critical_hh_isolation = [
        (get_total(df, False, "Benei Brak", "hh_isolation", "ASCENDING", "GENERAL"),
         get_total(df, False, "Benei Brak", "hh_isolation", "DESCENDING", "GENERAL"),
         get_total(df, False, "Benei Brak", "hh_isolation", "ASCENDING", "NEIGHBORHOOD"),
         get_total(df, False, "Benei Brak", "hh_isolation", "DESCENDING", "NEIGHBORHOOD")),  # Bnei Brak

        (get_total(df, False, "Holon", "hh_isolation", "ASCENDING", "GENERAL"),
         get_total(df, False, "Holon", "hh_isolation", "DESCENDING", "GENERAL"),
         get_total(df, False, "Holon", "hh_isolation", "ASCENDING", "NEIGHBORHOOD"),
         get_total(df, False, "Holon", "hh_isolation", "DESCENDING", "NEIGHBORHOOD"))]  # Holon

    # List of tuples
    total_infections_asymptomatic_detection = [
        (get_total(df, True, "Benei Brak", "asymptomatic_detection", "ASCENDING", "GENERAL"),
         get_total(df, True, "Benei Brak", "asymptomatic_detection", "DESCENDING", "GENERAL"),
         get_total(df, True, "Benei Brak", "asymptomatic_detection", "ASCENDING", "NEIGHBORHOOD"),
         get_total(df, True, "Benei Brak", "asymptomatic_detection", "DESCENDING", "NEIGHBORHOOD")),  # Bnei Brak

        (get_total(df, True, "Holon", "asymptomatic_detection", "ASCENDING", "GENERAL"),
         get_total(df, True, "Holon", "asymptomatic_detection", "DESCENDING", "GENERAL"),
         get_total(df, True, "Holon", "asymptomatic_detection", "ASCENDING", "NEIGHBORHOOD"),
         get_total(df, True, "Holon", "asymptomatic_detection", "DESCENDING", "NEIGHBORHOOD"))]  # Holon

    total_critical_asymptomatic_detection = [
        (get_total(df, False, "Benei Brak", "asymptomatic_detection", "ASCENDING", "GENERAL"),
         get_total(df, False, "Benei Brak", "asymptomatic_detection", "DESCENDING", "GENERAL"),
         get_total(df, False, "Benei Brak", "asymptomatic_detection", "ASCENDING", "NEIGHBORHOOD"),
         get_total(df, False, "Benei Brak", "asymptomatic_detection", "DESCENDING", "NEIGHBORHOOD")),  # Bnei Brak

        (get_total(df, False, "Holon", "asymptomatic_detection", "ASCENDING", "GENERAL"),
         get_total(df, False, "Holon", "asymptomatic_detection", "DESCENDING", "GENERAL"),
         get_total(df, False, "Holon", "asymptomatic_detection", "ASCENDING", "NEIGHBORHOOD"),
         get_total(df, False, "Holon", "asymptomatic_detection", "DESCENDING", "NEIGHBORHOOD"))]  # Holon

    # some up the cities
    # hh combinations:
    #   neighborhood_descending_bnei_brak + neighborhood_ascending_holon
    #   general_descending_bnei_brak + neighborhood_descending_holon
    # asymptomatic combinations:
    #   general_descending_bnei_brak + neighborhood_descending_holon
    total_infection_hh_both_cities_detection = [
        total_infections_hh_isolation[0][i] + total_infections_hh_isolation[1][i] for i in range(4)] + \
           [
               get_total(df, True, "Benei Brak", "hh_isolation", "DESCENDING", "NEIGHBORHOOD") +
               get_total(df, True, "Holon", "hh_isolation", "ASCENDING", "NEIGHBORHOOD"),

               get_total(df, True, "Benei Brak", "hh_isolation", "DESCENDING", "GENERAL") +
               get_total(df, True, "Holon", "hh_isolation", "DESCENDING", "NEIGHBORHOOD"),
           ]
    # normalize the numbers to be out of 100K
    total_infections_asymptomatic_both_cities_detection = [
        total_infections_asymptomatic_detection[0][i] + total_infections_asymptomatic_detection[1][i] for i in range(4)] + \
            [
                get_total(df, True, "Benei Brak", "asymptomatic_detection", "DESCENDING", "GENERAL") +
                get_total(df, True, "Holon", "asymptomatic_detection", "DESCENDING", "NEIGHBORHOOD"),
            ]
    total_infections_asymptomatic_both_cities_detection

    # hh combinations:
    #   neighborhood_descending_bnei_brak + neighborhood_ascending_holon
    #   general_descending_bnei_brak + neighborhood_descending_holon
    # asymptomatic combinations:
    #   general_descending_bnei_brak + neighborhood_descending_holon
    total_critical_hh_both_cities_detection = [
        total_critical_hh_isolation[0][i] + total_critical_hh_isolation[1][i] for i in range(4)] + \
                  [
                      get_total(df, False, "Benei Brak", "hh_isolation", "DESCENDING", "NEIGHBORHOOD") +
                      get_total(df, False, "Holon", "hh_isolation", "ASCENDING", "NEIGHBORHOOD"),

                      get_total(df, False, "Benei Brak", "hh_isolation", "DESCENDING", "GENERAL") +
                      get_total(df, False, "Holon", "hh_isolation", "DESCENDING", "NEIGHBORHOOD"),
                  ]
    total_critical_asymptomatic_both_cities_detection = [
        total_critical_asymptomatic_detection[0][i] + total_critical_asymptomatic_detection[1][i] for i in range(4)] + \
                [
                    get_total(df, False, "Benei Brak", "asymptomatic_detection", "DESCENDING", "GENERAL") +
                    get_total(df, False, "Holon", "asymptomatic_detection", "DESCENDING", "NEIGHBORHOOD"),
                ]

    # ----------- END ------------

    df = pd.DataFrame(total_infections_hh_isolation,
                      columns=[
                          'General\nAscending',
                          'General\nDescending',
                          'Neighborhood\nAscending',
                          'Neighborhood\nDescending',
                      ],
                      index=['Bnei Brak', 'Holon'])
    draw_heatmap(ax[0], df)
    ax[0].set_title(f"Total Infection for - Household Isolation")

    df = pd.DataFrame(total_critical_hh_isolation,
                      columns=[
                          'General\nAscending',
                          'General\nDescending',
                          'Neighborhood\nAscending',
                          'Neighborhood\nDescending',
                      ],
                      index=['Bnei Brak', 'Holon'])
    draw_heatmap(ax[1], df)
    ax[1].set_title(f"Total Critical for - Household Isolation")

    df = pd.DataFrame(total_infections_asymptomatic_detection,
                      columns=[
                          'General\nAscending',
                          'General\nDescending',
                          'Neighborhood\nAscending',
                          'Neighborhood\nDescending',
                      ],
                      index=['Bnei Brak', 'Holon'])
    draw_heatmap(ax[2], df)
    ax[2].set_title(f"Total Infection for - Asymptomatic Detection")

    df = pd.DataFrame(total_critical_asymptomatic_detection,
                      columns=[
                          'General\nAscending',
                          'General\nDescending',
                          'Neighborhood\nAscending',
                          'Neighborhood\nDescending',
                      ],
                      index=['Bnei Brak', 'Holon'])
    draw_heatmap(ax[3], df)
    ax[3].set_title(f"Total Critical for - Asymptomatic Detection")

    # ------------------------------------------------------------------------------------------------
    total_infections = [0, 0]
    total_infections[0] = total_infections_hh_isolation[0] + total_infections_asymptomatic_detection[0]
    total_infections[1] = total_infections_hh_isolation[1] + total_infections_asymptomatic_detection[1]

    df = pd.DataFrame(total_infections,
                      columns=[
                          'hh_isolation\nGeneral\nAscending',
                          'hh_isolation\nGeneral\nDescending',
                          'hh_isolation\nNeighborhood\nAscending',
                          'hh_isolation\nNeighborhood\nDescending',
                          'asymptomatic isolation\nGeneral\nAscending',
                          'asymptomatic isolation\n\nGeneral\nDescending',
                          'asymptomatic isolation\n\nNeighborhood\nAscending',
                          'asymptomatic isolation\n\nNeighborhood\nDescending',
                      ],
                      index=['Bnei Brak', 'Holon'])
    draw_heatmap(ax[4], df)
    ax[4].set_title(f"Total Infections")

    total_critical = [0, 0]
    total_critical[0] = total_critical_hh_isolation[0] + total_critical_asymptomatic_detection[0]
    total_critical[1] = total_critical_hh_isolation[1] + total_critical_asymptomatic_detection[1]

    df = pd.DataFrame(total_critical,
                      columns=[
                          'hh_isolation\nGeneral\nAscending',
                          'hh_isolation\nGeneral\nDescending',
                          'hh_isolation\nNeighborhood\nAscending',
                          'hh_isolation\nNeighborhood\nDescending',
                          'asymptomatic isolation\nGeneral\nAscending',
                          'asymptomatic isolation\nGeneral\nDescending',
                          'asymptomatic isolation\nNeighborhood\nAscending',
                          'asymptomatic isolation\nNeighborhood\nDescending',
                      ],
                      index=['Bnei Brak', 'Holon'])
    draw_heatmap(ax[5], df)
    ax[5].set_title(f"Total Hospitalization")

    total_critical_combinations = [0]
    total_critical_combinations[0] = total_critical_hh_both_cities_detection + \
                                     total_critical_asymptomatic_both_cities_detection

    df_critical_combinations = pd.DataFrame(total_critical_combinations,
                      columns=[
                          'hh_isolation\nGeneral\nAscending',
                          'hh_isolation\nGeneral\nDescending',
                          'hh_isolation\nNeighborhood\nAscending',
                          'hh_isolation\nNeighborhood\nDescending',
                          'hh_isolation\nBnei Brak - ND\nHolon - NA',
                          'hh_isolation\nBnei Brak - GD\nHolon - ND',
                          'asymptomatic isolation\nGeneral\nAscending',
                          'asymptomatic isolation\nGeneral\nDescending',
                          'asymptomatic isolation\nNeighborhood\nAscending',
                          'asymptomatic isolation\nNeighborhood\nDescending',
                          'asymptomatic isolation\nBnei Brak - GD\nHolon - ND'
                      ],
                      index=['Bnei Brak+Holon'])
    df_critical_combinations.sort_values(inplace=True, by=['Bnei Brak+Holon'], axis=1, ascending=False)
    draw_heatmap(ax[6], df_critical_combinations)
    ax[6].set_title(f"Hospitalizations per 100K - Both Cities")

    total_infected_combinations = [0]
    total_infected_combinations[0] = total_infection_hh_both_cities_detection + \
                                     total_infections_asymptomatic_both_cities_detection

    df_infected_combinations = pd.DataFrame(total_infected_combinations,
                      columns=[
                          'hh_isolation\nGeneral\nAscending',
                          'hh_isolation\nGeneral\nDescending',
                          'hh_isolation\nNeighborhood\nAscending',
                          'hh_isolation\nNeighborhood\nDescending',
                          'hh_isolation\nBnei Brak - ND\nHolon - NA',
                          'hh_isolation\nBnei Brak - GD\nHolon - ND',
                          'asymptomatic isolation\nGeneral\nAscending',
                          'asymptomatic isolation\nGeneral\nDescending',
                          'asymptomatic isolation\nNeighborhood\nAscending',
                          'asymptomatic isolation\nNeighborhood\nDescending',
                          'asymptomatic isolation\nBnei Brak - GD\nHolon - ND'
                      ],
                      index=['Bnei Brak+Holon'])
    df_infected_combinations = df_infected_combinations[df_critical_combinations.columns]
    draw_heatmap(ax[7], df_infected_combinations)
    ax[7].set_title(f"Infections per 100K - Both Cities")

    # ------------------------------------------------------------------------------------------------
    fig.tight_layout(pad=12.0)
    fig.savefig(f"../../outputs/{sys.argv[1]}/heatmap.svg")

    # fig.set_figwidth(16)
    # fig.set_figheight(plots * 30)
