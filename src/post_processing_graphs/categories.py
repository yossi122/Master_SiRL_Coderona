
class Categories:
    def __init__(self, one_run, including_city=False):
        self.including_city = including_city
        parameters = one_run.split(',')
        # patch the name of the city to make it correct. the original "wrong" name comes from our raw csv files
        self.city = parameters[0].lower().replace("holon", "Holon").replace("bene beraq", "Benei Brak")
        self.intervention = parameters[1]
        if "ASCENDING" in one_run:
            self.order = "ASCENDING"
        elif "DESCENDING" in one_run:
            self.order = "DESCENDING"
        else:
            self.order = "NONE"
        if "HOUSEHOLDS_ALL_AT_ONCE" in one_run:
            self.vaccination_strategy = "HH_ALL_AT_ONCE"
        elif "HOUSEHOLD" in one_run:
            self.vaccination_strategy = "HOUSEHOLD"
        elif "BY_NEIGHBORHOOD" in one_run:
            self.vaccination_strategy = "NEIGHBORHOOD"
        elif "GENERAL" in one_run:
            self.vaccination_strategy = "GENERAL"
        else:
            print(f"ERROR! unknown order! in {one_run}")
            exit(-1)
        self.immune_per_day = 0
        self.initial_infected = 0
        for i in range(len(parameters)):
            if 'imm_per_day' in parameters[i]:
                self.immune_per_day = parameters[i].split('=')[1]
            if 'inf=' in parameters[i]:
                self.initial_infected = parameters[i].split('=')[1]
            if 'comp=' in parameters[i]:
                self.compliance = parameters[i].split('=')[1]

    def __str__(self):
        # if the category includes City, then we do not want to show it in here
        if self.including_city:
            return f"{self.vaccination_strategy}\n{self.order}"
        else:
            return f"{self.city}\n{self.vaccination_strategy}\n{self.order}"
        # return f"{self.city}\nINF={self.initial_infected}\nIMMUNE={self.immune_per_day}\n" \
        #        f"{self.vaccination_strategy}\n{self.order}\ncompliance={self.compliance}"

