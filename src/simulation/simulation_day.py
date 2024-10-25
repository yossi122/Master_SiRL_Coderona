
class SimulationDate:
    _current_date = None

    @classmethod
    def get_current_date(cls):
        return cls._current_date

    @classmethod
    def set_current_date(cls, current_date):
        cls._current_date = current_date

