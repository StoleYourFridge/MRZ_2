import cv2
import numpy as np
from AssociativeMemoryNetwork import AssociativeMemoryNetwork
import os


class ImageManager:
    ASSOCIATIONS_DIR = "Images/Associations"
    RESULTS_DIR = "Images/Results"
    BROKEN_DIR = "Images/BrokenAssociations"
    HEIGHT = 10
    WIDTH = 10

    @staticmethod
    def from_image_to_array(*image_filenames):
        result = []
        for image_file in image_filenames:
            image_data = cv2.imread(image_file, 0)
            image_data = image_data.reshape(1, image_data.shape[0] * image_data.shape[1])
            result.append(AssociativeMemoryNetwork.activation_function(image_data))
        return tuple(result)

    @staticmethod
    def deactivation_function(array):
        return np.array([[0 if item == -1 else 255 for item in array[0]]])

    @classmethod
    def from_array_to_image(cls, *arrays):
        result = []
        for array in arrays:
            array = cls.deactivation_function(array)
            array = array.reshape(cls.HEIGHT, cls.WIDTH)
            result.append(array)
        return tuple(result)

    @classmethod
    def get_data_container(cls):
        associations = os.listdir(cls.ASSOCIATIONS_DIR)
        results = os.listdir(cls.RESULTS_DIR)
        print(associations)
        print(results)
        data = (cls.from_image_to_array(f"{cls.ASSOCIATIONS_DIR}/{association}",
                                        f"{cls.RESULTS_DIR}/{result}") for association,
                                                                           result in zip(associations, results))
        return tuple(data)

    @staticmethod
    def run():
        neural_network = AssociativeMemoryNetwork(1_000)
        data = ImageManager.get_data_container()
        neural_network.set_weights(data)
        filename = input("From following broken image files:\n" +
                         f"{os.listdir(ImageManager.BROKEN_DIR)}\n"
                         "Enter filename to be repaired: ")
        to_repair = ImageManager.from_image_to_array(f"{ImageManager.BROKEN_DIR}/{filename}")[0]
        after_repair, association = neural_network.process(to_repair, True)
        after_repair, association = ImageManager.from_array_to_image(after_repair, association)
        before_repair = cv2.imread(f"{ImageManager.BROKEN_DIR}/{filename}", 0)
        scaled = np.concatenate((before_repair, after_repair, association), axis=1)
        cv2.imshow("Work, please!", scaled.astype(np.uint8))
        cv2.waitKey(0)


if __name__ == "__main__":
    pass
