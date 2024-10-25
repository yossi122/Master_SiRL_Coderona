from statistics import mean, stdev
from enum import Enum
import numpy as np 

class TableFormat(Enum):
    TEXTUAL = 1
    CSV = 2

    def format(self, table):
        if self == TableFormat.TEXTUAL:
            for col in range(len(table[0])):
                max_strlen = max(len(table[i][col]) for i in range(len(table)))
                for i in range(len(table)):
                    if col == 0:
                        table[i][col] = table[i][col].ljust(max_strlen, " ")
                    else:
                        table[i][col] = table[i][col].rjust(max_strlen, " ")
            return "\n".join("  ".join(row) for row in table)
        assert self == TableFormat.CSV
        return "\n".join(",".join(row) for row in table)

    def get_file_extension(self):
        if self == TableFormat.TEXTUAL:
            return 'txt'
        else:
            return 'csv'

def summary_datas_to_absolute_and_relative_means_and_stdevs(summary_datas):
    """
    Accepts multiple summary datas (one from each run of the same strategy),
    computes and returns strings describing the mean and standard variation of different properties,
    both absolute (i.e. 100 +- 20 infected) and relative (i.e. 30% +- 5% infected).
    In the case where there is only one summary data, does not
    compute standard deviation.
    :param summary_datas: A list of 'summary_data' dicts, one from each run
    :return: Two dicts from property names to "value"
    (string containing mean and standard variation).
    One for absolute amounts and another for relative amounts.
    """
    keys_to_data_arrays = {}
    for i, data in enumerate(summary_datas):
        assert len(data) == len(summary_datas[0])
        for key, val in data.items():
            if i == 0:
                keys_to_data_arrays[key] = []
            else:
                assert key in keys_to_data_arrays
            keys_to_data_arrays[key].append(val)
    absolute_data = {}
    relative_data = {}
    for key, absolute_array in keys_to_data_arrays.items():
        #relative_array = [x / y for (x, y) in zip(absolute_array, keys_to_data_arrays["Total people"])]
        relative_array = []
        for (x,y) in zip(absolute_array, keys_to_data_arrays["Total people"]):
            if y != 0:
                relative_array.append(x/y)
            else:
                relative_array.append(np.nan)
        
        if len(relative_array) == 1:
            absolute_data[key] = "{}".format(absolute_array[0])
            relative_data[key] = "{:3.3f}%".format(100 * relative_array[0])
        else:
            absolute_data[key] = "{m} ±{s:6.1f}".format(m=mean(absolute_array), s=stdev(absolute_array))
            relative_data[key] = "{m}% ±{s:3.3f}%".format(m=100 * mean(relative_array), s=100 * stdev(relative_array))
    return absolute_data, relative_data

def make_summary_by_age_table(age_group_to_summary_tables, table_format, is_relative=True):
    """
    :param age_group_to_summary_tables: A table from age_group to
    a list of 'summary_data's on that age group
    :return: A short readable text containing a table of
    age-stratified results (total deceased, total infected, ...)
    """
    assert isinstance(table_format, TableFormat)
    age_group_to_absolute_data = {}
    age_group_to_relative_data = {}
    for age_group, summary_datas in age_group_to_summary_tables.items():
        age_group_to_absolute_data[age_group], age_group_to_relative_data[age_group] = \
            summary_datas_to_absolute_and_relative_means_and_stdevs(summary_datas)
    age_groups = list(age_group_to_summary_tables.keys())
    table = []
    age_groups_text = ['Age group']
    for age_group in age_groups:
        if age_group is None:
            age_groups_text.append("Total")
        else:
            age_groups_text.append("{}-{}".format(*age_group))
    table.append(age_groups_text)
    keys = ["Total people"] + [key for key in age_group_to_relative_data[age_groups[0]] if key != "Total people"]
    for key in keys:
        curr = [key]
        for age_group in age_groups:
            if (key == "Total people") or not is_relative:
                curr.append(age_group_to_absolute_data[age_group][key])
            else:
                curr.append(age_group_to_relative_data[age_group][key])
        table.append(curr)
    return table_format.format(table)
