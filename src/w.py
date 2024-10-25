from src.seir.seir_times import RealDataSeirTimesGeneration
from datetime import timedelta
from functional import seq
from src.simulation.params import Params
import os
import json

class W:
    def __init__(self):
        p_hulon = 0.793
        p_bnei_brak = 0.754
        p_city_avg = (p_hulon+p_bnei_brak)/2
        number_of_samples = 1000
        s = RealDataSeirTimesGeneration()
        infectious_before_symptomatic_distribution = int(seq([s._infectious_before_symptomatic_distribution.sample() for i in range(number_of_samples)]).average())
        infectious_before_immune_distribution = seq([s._infectious_before_immune_distribution.sample() for i in range(number_of_samples)]).average()
        # print(f"infectious_before_symptomatic_distribution={infectious_before_symptomatic_distribution}, infectious_before_immune_distribution={infectious_before_immune_distribution}")
        factors = Params.loader()["disease_parameters"]["infectiousness_per_stage"]
        p_incubating_post_latent = factors["incubating_post_latent"]
        p_symptomatic = p_city_avg # factors["symptomatic"]
        p_per_day = [p_incubating_post_latent] * int(infectious_before_symptomatic_distribution) + [p_symptomatic] * int(infectious_before_immune_distribution)
        sum_p = seq(p_per_day).sum()
        self.p_per_day_rate = seq(p_per_day).map(lambda x: x/sum_p)

    def get_w(self, prev_days: int) -> float:
        try:
            return self.p_per_day_rate[-prev_days]
        except:
            return 0.0

    def w_len(self):
        return self.p_per_day_rate.len()


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path) as json_data_file:
        ConfigData = json.load(json_data_file)
        paramsDataPath = ConfigData['ParamsFilePath']

    Params.load_from(os.path.join(os.path.dirname(__file__), paramsDataPath), override=True)
    w = W()
    print(f"p per day rate = {[w.get_w(i) for i in range(1,w.w_len())]}")