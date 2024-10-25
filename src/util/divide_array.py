from math import ceil
import heapq

def divide_array(arr, max_segment_size):
    """
    Divides an array evenly to parts of size up to max_segment_size, with the following properties:
    - The number of segments is the smallest possible
    - The size of no two segments differ by more then 1
    :param arr: The array to divide
    :param max_segment_size: The maximum possible size for a segment
    :return: A list of sub-arrays of the original array
    """
    assert len(arr) != 0, "Trying to divide an empty array"
    num_segments = ceil(len(arr) / max_segment_size)
    # All segments will either be of size base_segment_size or base_segment_size+1
    base_segment_size = len(arr) // num_segments
    num_segments_of_larger_size = len(arr) % num_segments
    num_segments_of_smaller_size = num_segments - num_segments_of_larger_size
    t = 0
    ret = []
    for _ in range(num_segments_of_smaller_size):
        ret.append(arr[t:t+base_segment_size])
        t += base_segment_size
    for _ in range(num_segments_of_larger_size):
        ret.append(arr[t:t+base_segment_size+1])
        t += base_segment_size + 1
    assert t == len(arr)
    return ret

def divide_weighted_array(items_with_weights, max_segment_weight):
    """
    Divides an array of elements with weights into as few segments as possible
    with some maximum allowed total segment weight,
    in a way that all total segment weights are approximately equal
    :param items_with_weights: An array of (item, weight) to divide
    :param max_segment_weight: The maximum possible weight for each segment
    :return: A list of segments (only the items, not the weights)
    """
    total_weight = sum([weight for (item, weight) in items_with_weights])
    num_segments = ceil(total_weight / max_segment_weight)
    while True:
        # Try to divide to num_segments segments
        ret_segments = [[] for i in range(num_segments)]
        ret_total_weights_heap = [(0, i) for i in range(num_segments)]
        heapq.heapify(ret_total_weights_heap)
        failed = False
        for item, weight in items_with_weights:
            segment_weight, segment_idx = heapq.heappop(ret_total_weights_heap)
            if weight + segment_weight > max_segment_weight:
                failed = True
                break
            ret_segments[segment_idx].append(item)
            heapq.heappush(ret_total_weights_heap, (segment_weight + weight, segment_idx))
        if failed:
            num_segments += 1
            continue
        return ret_segments
