import random as _random
from bisect import bisect as _bisect

class DiscreteDistribution(object):
    """
    Some distribution on objects,
    the probability to land in object[i] is probs[i].
    """
    def __init__(self, objects, probs, check_sum=True):
        self.objects = objects
        self.probs = probs
        for p in self.probs:
            assert p >= 0
        sum_probs = sum(self.probs)
        if check_sum:
            assert 0.998 < sum_probs < 1.002, "Sum of probabilities is %s" % sum_probs
        self.cumprob = [0]
        self.sum_probs = sum_probs
        for p in self.probs:
            self.cumprob.append(self.cumprob[-1] + p)

    def sample(self):
        """
        :return: A random element with this distribution.
        """
        ret = _bisect(self.cumprob, _random.random() * self.sum_probs) - 1
        assert ret >= 0
        assert ret < len(self.cumprob)
        return self.objects[ret]


class Distribution(object):
    """
    Some distribution on numbers.
    The probability to land in segments[i] is probs[i],
    and inside each segment the distribution is integer-uniform if is_integer is True,
    and real-uniform otherwise
    """
    __slots__ = ('segments', 'segment_distribution', 'is_integer', 'is_singleton', 'probs')

    def __init__(self, segments, probs, is_integer=True, check_sum=True, is_singleton=False):
        self.segments = segments
        self.segment_distribution = DiscreteDistribution(segments, probs, check_sum)
        self.is_integer = is_integer
        self.is_singleton = is_singleton

        if not self.is_singleton:
            for seg in self.segments:
                assert len(seg) == 2, str(seg) + " is not a segment!"
                if is_integer:
                    assert seg[0] <= seg[1], str(seg) + " is not a valid segment!"
                else:
                    assert seg[0] < seg[1], str(seg) + " is not a valid segment!"
            for i in range(len(self.segments) - 1):
                if is_integer:
                    assert self.segments[i][1] + 1 == self.segments[i + 1][0]
                else:
                    assert self.segments[i][1] == self.segments[i + 1][0]

    def sample(self):
        """
        :return: A random integer with this distribution.
        """
        segment = self.segment_distribution.sample()
        if self.is_singleton:
            return segment
        if self.is_integer:
            return _random.randint(*segment)
        return _random.uniform(*segment)

    def mean(self): # never used!
        """
        :return: The mean value of this distribution
        """
        return sum(
            [self.segment_distribution.probs[i] * (self.segments[i][0] + self.segments[i][1]) / 2.0 for i in range(len(self.segments))])
