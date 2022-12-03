import numpy as np
from itertools import groupby


class AssociativeMemoryNetwork:
    layers_amount = 2

    def __init__(self, maximum_iterations):
        self.__weights = None
        self.__maximum_iterations = maximum_iterations
        self.__layer_buffer = None
        self.__stop_iteration = None

    def set_to_default(self, input_layer):
        self.__layer_buffer = [None, input_layer]
        self.__stop_iteration = [False, False]

    def set_weights(self, data):
        shapes = self.memory_data_checker(data)
        weights = np.zeros((shapes[0], shapes[1]), dtype=int)
        for association_layer, result_layer in data:
            weight_component = np.dot(np.transpose(association_layer), result_layer)
            weights += weight_component
        self.__weights = weights

    def process_step(self, is_association):
        multiply_on = self.__weights if is_association else self.__weights.transpose()
        result_layer = self.activation_function(np.dot(self.__layer_buffer[-1], multiply_on))
        compare_result = np.array_equal(result_layer, self.__layer_buffer[0])
        self.__stop_iteration[int(is_association)] = compare_result
        self.__layer_buffer.append(result_layer)
        self.__layer_buffer.pop(0)

    def process(self, input_layer, is_association):
        self.memory_vector_checker(input_layer)
        self.set_to_default(input_layer)
        iterations = 0
        while not all(self.__stop_iteration) and iterations != self.__maximum_iterations:
            self.process_step(is_association)
            is_association = not is_association
            iterations += 1
        return self.__layer_buffer[::-1] if iterations % 2 == 0 else self.__layer_buffer[::]

    @staticmethod
    def activation_function(array):
        return np.array([[1 if item > 0 else -1 for item in array[0]]])

    @staticmethod
    def memory_vector_checker(array):
        if not isinstance(array, np.ndarray):
            raise ValueError("Data containers should be iterable")
        if any(item not in (-1, 1) for item in array[0]):
            raise ValueError("Data container elements should be -1 or 1 ")

    @classmethod
    def memory_data_checker(cls, data):
        if not isinstance(data, (list, tuple)):
            raise ValueError("Data containers should be iterable")
        if any(not isinstance(item, (list, tuple)) for item in data):
            raise ValueError("Data sub-containers should be iterable")
        if any(len(item) != cls.layers_amount for item in data):
            raise ValueError(f"Data sub-containers should be sized with {cls.layers_amount}")
        for association_layer, result_layer in data:
            if not isinstance(association_layer, np.ndarray) and isinstance(result_layer, np.ndarray):
                raise ValueError("Data sub-sub-containers should be numpy arrays")
        if not all_equal((item[0].shape for item in data)) or not all_equal((item[1].shape for item in data)):
            raise ValueError("Data sub-sub-containers shapes should be identical")
        return data[0][0].shape[1], data[0][1].shape[1]


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


if __name__ == "__main__":
    pass
