from src.logs.r0_data import save_population_to_file, load_population_to_file, calculate_r0_data, debug_save_population_to_file


def test_calculate_r0_data():
    population = load_population_to_file()
    result = calculate_r0_data(population)
    print(result)

