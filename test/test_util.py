from random import randint
from src.util.divide_array import divide_array, divide_weighted_array
from src.util.distribution import DiscreteDistribution, Distribution


def test_divide_array():
    for _ in range(50):
        lst = [i for i in range((randint(50, 100)))]
        new_lst = divide_array(lst, randint(4, 15))
        assert len(new_lst) != 0
        len_lst = [len(x) for x in new_lst]
        assert max(len_lst) - min(len_lst) < 2    



def test_divide_weighted_array():
    for _ in range(50):
        dummy_lst = [Dummy() for _ in range(randint(50, 100))]
        weighted_lst = [(dummy, dummy.weight) for dummy in dummy_lst]
        new_lst = divide_weighted_array(weighted_lst, randint(10, 20))
        assert len(new_lst) != 0
        weight_lst = [find_total_weight(lst) for lst in new_lst]
        assert max(weight_lst) - min(weight_lst) < 15    


def find_total_weight(dummies):
    total = 0
    for dummy in dummies:
        total += dummy.weight
    return total

class Dummy(object):
    def __init__(self):
        self.weight = randint(1, 10)
    
def test_DiscreteDistribution():
    number_distribution = DiscreteDistribution(objects=[1,2,3],probs=[1/3,1/3,1/3])
    for _ in range(10^6):
        assert number_distribution.sample() in [1,2,3]

    acc_samples_dic ={1:0,2:0,3:0}
    total_rounds = 10^3
    samples_in_rounds = 10^3
    for _ in range(total_rounds):
        samples_dic = {1:0,2:0,3:0}
        for _ in range(samples_in_rounds):
            samples_dic[number_distribution.sample()]+=1
        for i in samples_dic.keys():
            acc_samples_dic[i] += samples_dic[i]
    for i in acc_samples_dic.keys():
            acc_samples_dic[i] /=total_rounds

    assert abs(acc_samples_dic[2] - acc_samples_dic[3]) <= 10
    assert abs(acc_samples_dic[1] - acc_samples_dic[3]) <= 10
    assert abs(acc_samples_dic[2] - acc_samples_dic[1]) <= 10

def test_DiscreteDistribution2():
    number_distribution = DiscreteDistribution(objects=[1,2,3],probs=[1/2,1/3,1/6])
    for _ in range(10^6):
        assert number_distribution.sample() in [1,2,3]
    
    acc_samples_dic ={1:0,2:0,3:0}
    total_rounds = 10^3
    samples_in_rounds = 10^3
    for _ in range(total_rounds):
        samples_dic = {1:0,2:0,3:0}
        for _ in range(samples_in_rounds):
            samples_dic[number_distribution.sample()]+=1
        for i in samples_dic.keys():
            acc_samples_dic[i] += samples_dic[i]
    for i in acc_samples_dic.keys():
            acc_samples_dic[i] /=total_rounds

    assert abs(acc_samples_dic[2] - 2*acc_samples_dic[3]) <= 10
    assert abs(acc_samples_dic[1] - 3*acc_samples_dic[3]) <= 10
    assert abs(acc_samples_dic[1] - 1.5*acc_samples_dic[2]) <= 10

def test_Distribution():
    number_distribution = Distribution(segments=[[0,1],[2,3],[4,5]],probs=[1/3,1/3,1/3])
    for _ in range(10^6):
        assert number_distribution.sample() in [0,1,2,3,4,5]
    
    acc_samples_dic ={0:0,1:0,2:0}
    total_rounds = 10^3
    samples_in_rounds = 10^3
    for _ in range(total_rounds):
        samples_dic = {0:0,1:0,2:0}
        for _ in range(samples_in_rounds):
            samples_dic[number_distribution.sample() //2 ]+=1
        for i in samples_dic.keys():
            acc_samples_dic[i] += samples_dic[i]
    for i in acc_samples_dic.keys():
            acc_samples_dic[i] /=total_rounds

    assert abs(acc_samples_dic[1] - acc_samples_dic[2]) <= 10
    assert abs(acc_samples_dic[0] - acc_samples_dic[2]) <= 10
    assert abs(acc_samples_dic[1] - acc_samples_dic[0]) <= 10

def test_Distribution2():
    number_distribution = Distribution(segments=[[0,1],[2,3],[4,5]],probs=[1/2,1/3,1/6])
    for _ in range(10^3):
        assert number_distribution.sample() in [0,1,2,3,4,5]

    acc_samples_dic ={0:0,1:0,2:0}
    total_rounds = 10^3
    samples_in_rounds = 10^3
    for round in range(total_rounds):
        samples_dic = {0:0,1:0,2:0}
        for _ in range(samples_in_rounds):
            samples_dic[number_distribution.sample() //2 ]+=1
        for i in samples_dic.keys():
            acc_samples_dic[i] += samples_dic[i]
    for i in acc_samples_dic.keys():
            acc_samples_dic[i] /=total_rounds

    assert abs(acc_samples_dic[1] - 2 * acc_samples_dic[2]) <= 10
    assert abs(acc_samples_dic[0] - 3 * acc_samples_dic[2]) <= 10
    assert abs(acc_samples_dic[0] - 1.5 * acc_samples_dic[1]) <= 10