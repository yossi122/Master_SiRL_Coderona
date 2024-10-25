import json
import os


class Params(object):
    """
    A singleton class, that gives access to the params of the simulation.
    It supports loading a single param file in JSON format, update specific params, and saving them.
    """
    __slots__ = ('_data', '_changes')

    # indicated if the params are already loaded in the current process
    singleton = None

    def __init__(self, path):
        with open(path, 'r') as f:
            self._data = json.load(f)
        # keeps track of the temporary changes made after loading the params, not yet saved to file
        self._changes = {}

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        """
        Update the given key in the params data to have the new value.
        It will be saved in changed dict, and not on the disk.
        :param key: can be a sequence of keys, for deep update or single key
        :param value: new value
        """
        if isinstance(key, list) or isinstance(key, tuple):
            self.deep_update(key, value)
            return
        self._data[key] = value
        self._changes[(key,)] = value

    def __eq__(self, other):
        return isinstance(other, Params) and self._data == other._data

    def deep_update(self, key, value):
        """
        Make a deep update of data[key1][key2][...] = val in order to update keys[-1] to value
        :param key: list of the series of keys
        :param value: the new value of keys[-1]
        """
        key = tuple(key)
        self._changes[key] = value
        if isinstance(key, str):
            key = [key]
        d = self._data
        for k in key[:-1]:
            d = d[k]
        d[key[-1]] = value

    def description(self):
        """
        :return: str description of the changed of this params, to distinguish different params objects
        """
        return "".join("_" + "_".join(key) + "_" + str(val) for key, val in self._changes.items())

    @classmethod
    def loader(cls):
        return cls.singleton

    @classmethod
    def load_from(cls, path, override=False):
        """
        Loads data from the given path
        :param path: full file path to JSON file
        :param override: if False, won't update the params data if it's loaded already.
        """
        if cls.singleton is None or override:
            cls.singleton = Params(path)

    def dump(self, path):
        """
        Dumps the contents of the Params to a json file.
        Used in order to link the outputs of a run to its input params.
        :param path: The path to which the json file should be written.
        :return: None
        """
        assert not os.path.exists(path), "File {} already exists!".format(path)
        with open(path, 'w') as f:
            json.dump(self._data, f)
    
    @classmethod
    def clean(cls):
        """
        Clean the value of the singleton in case we want to update it's values
        """
        cls.singleton = None