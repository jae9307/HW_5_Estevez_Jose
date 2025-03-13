import os
import re
import numpy as np
import random
import sys
import math
import matplotlib.pyplot as plt
import seaborn as sns

# Defines a Threshold_Information object which stores the best threshold for a particular attribute, as well as the threshold's
# badness and whether aggressive drivers are the majority class in the left group of data
class Threshold_Information:
    def __init__(self, badness, left_is_aggressive_flag, threshold):
        self.badness = badness
        self.left_is_aggressive_flag = left_is_aggressive_flag
        self.threshold = threshold

def read_file(fileName):
    drivers = []
    attributes = []

    if os.path.isfile(fileName):
        file = open(fileName, "r")
        lines = file.readlines()

        for line in lines:
            entries = line.split(",")

            # If the line does not consist of numbers, create a list of attributes from the line
            float_regex = r"^-?\d+\.\d+$"
            if not re.match(float_regex, entries[0].replace(" ", "")):
                for word in entries:
                    attributes.append(word.replace("\n", ""))
                continue

            # Create a dictionary for each driver, where the key is the attribute name and
            # the value is the attribute value
            driver_dictionary = {}
            for index in range(len(attributes)):
                driver_dictionary.update({attributes[index]: int(np.round(float(entries[index])))})
            drivers.append(driver_dictionary)

    return drivers, attributes

# Reads all files in the project one file at a time. read_file() is called on the current file and returns the array of drivers
# from that file (if the file is one of the traffic station data files, otherwise returns an empty array). The contents of this
# array are then added to an array of drivers from all files
def read_all_files_in_project(start_path='.'):
    drivers = []
    attributes = []

    for root, dirs, files in os.walk(start_path):
        for file in files:
            filename = os.path.join(root, file)
            if filename.__contains__('Data_'):
                file_drivers, attributes = read_file(filename)

                if file_drivers.__len__() > 0:
                    drivers.extend(file_drivers)

    return drivers, attributes

def find_threshold_index(index, drivers, threshold, current_best_index, attribute):
    if drivers[index][attribute] > threshold:
        if index == 0:
            return current_best_index
        return find_threshold_index(math.ceil(drivers[:index].__len__() / 2) - 1, drivers[:index], threshold,
                                    current_best_index, attribute)
    else:
        if drivers.__len__()-1 == index:
            return index + 1 + current_best_index
        return find_threshold_index(math.ceil(drivers[index + 1:].__len__() / 2) - 1, drivers[index + 1:], threshold,
                                    index if index > current_best_index else current_best_index + index + 1, attribute)

def find_best_threshold(drivers, attribute):
    lowest_value = drivers[0][attribute]
    highest_value = drivers[-1][attribute]

    least_badness = sys.maxsize
    left_is_aggressive_flag = False  # tracks if the dominant class in the left group of data is aggressive drivers
    least_badness_threshold = -1

    for threshold in range(lowest_value, highest_value):
        threshold_index = find_threshold_index(math.ceil(drivers.__len__() / 2)-1, drivers, threshold, -1, attribute)

        # create the groups left and right of the threshold using the threshold_index
        left_data = drivers[:threshold_index + 1]
        right_data = drivers[threshold_index + 1:]

        # calculate if aggressive drivers are the majority group in the left data
        aggressive_drivers_in_left_data = sum(driver['INTENT'] == 2 for driver in left_data)
        left_is_aggressive = aggressive_drivers_in_left_data > len(left_data) - aggressive_drivers_in_left_data

        # track number of false alarms and misses at this threshold
        false_alarms = 0
        misses = 0

        if left_is_aggressive:  # aggressive drivers are majority group in left data
            # calculate number of false alarms in left group and misses in right group at this threshold
            false_alarms = sum(driver['INTENT'] < 2 for driver in left_data)
            misses = sum(driver['INTENT'] == 2 for driver in right_data)
        else:  # aggressive drivers are majority group in right data
            # calculate number of false alarms in right group and misses in left group at this threshold
            false_alarms = sum(driver['INTENT'] < 2 for driver in right_data)
            misses = sum(driver['INTENT'] == 2 for driver in left_data)

        # If this threshold is the best so far, track its badness, the threshold value, and whether aggressive drivers are the
        # majority class in the left group
        badness = false_alarms + misses
        if badness < least_badness:
            least_badness = badness
            left_is_aggressive_flag = left_is_aggressive
            least_badness_threshold = threshold

    return Threshold_Information(least_badness, left_is_aggressive_flag, least_badness_threshold)

def make_classifier(threshold_information, best_attribute):
    classifer_program = """
import os
import re
import numpy as np
import argparse
    
def classifier():
    parser = argparse.ArgumentParser(prog='classifier', description='Classifies drivers as safe or aggressive')
    parser.add_argument('filename')
    args = parser.parse_args()
    
    drivers = read_file(args.filename)
    sorted_drivers = drivers
    
    sorted_drivers = sorted(drivers, key=lambda driver: driver.{attribute})
        
    print("Aggressive drivers: %d" % sum(driver.{attribute} {aggressive_comparator} {threshold} for driver in sorted_drivers))
    print("Safe drivers: %d" % sum(driver.{attribute} {safe_comparator} {threshold} for driver in sorted_drivers))
    
classifier()
    """.format(attribute = best_attribute, threshold = threshold_information.threshold,
               aggressive_comparator = "<=" if threshold_information.left_is_aggressive_flag else ">",
               safe_comparator = ">" if threshold_information.left_is_aggressive_flag else "<=")

    file = open("HW_05_Classifier_Estevez_Jose.py", "w")
    file.write(classifer_program)
    file.close()

def scatter_all_attributes(drivers, attributes):
    colors = {0: 'blue', 1: 'blue', 2: 'red'}
    for i in range(len(attributes) - 1):
        for j in range(i + 1, len(attributes)-1):
            attribute_name = attributes[i]
            second_attribute_name = attributes[j]
            first_attributes_values = [driver[attributes[i]] + np.random.uniform(-0.5, 0.5) for driver in drivers]
            second_attributes_values = [driver[attributes[j]] + np.random.uniform(-0.5, 0.5) for driver in drivers]
            color_values = [colors[int(driver['INTENT'])] for driver in drivers]
            sns.scatterplot(x=first_attributes_values, y=second_attributes_values, hue=color_values, palette={'blue': 'blue', 'red': 'red'}, alpha=0.6, edgecolor='w')
            plt.title(f"{attribute_name} vs {second_attribute_name}")
            plt.savefig(f"{attribute_name}_vs_{second_attribute_name}.png")
            plt.clf()

def create_tree(drivers, attributes, depth):
    best_threshold = None
    best_attribute = ''
    best_left_data = []
    best_right_data = []

    for attribute in attributes:
        sorted_drivers = sorted(drivers, key=lambda single_driver: single_driver[attribute])

        multi_value_attributes = ['Speed', 'NLaneChanges', 'Brightness', 'NumDoors']

        if multi_value_attributes.__contains__(attribute):
            threshold_information = find_best_threshold(sorted_drivers, attribute)

            if best_threshold is None or threshold_information.badness < best_threshold.badness:
                best_threshold = threshold_information
                best_attribute = attribute
                best_left_data = list(filter(lambda driver: driver[attribute] <= threshold_information.threshold,
                                        drivers))
                best_right_data = list(filter(lambda driver: driver[attribute] > threshold_information.threshold,
                                        drivers))
        else:   # most attributes only have values of 1 or 0
            left_data = list(filter(lambda driver: driver[attribute] == 0, drivers))
            right_data = list(filter(lambda driver: driver[attribute] == 1, drivers))

            # calculate if aggressive drivers are the majority group in the left data
            aggressive_drivers_in_left_data = sum(driver['INTENT'] == 2 for driver in left_data)
            left_is_aggressive = aggressive_drivers_in_left_data > len(left_data) - aggressive_drivers_in_left_data

            # track number of false alarms and misses at this threshold
            false_alarms = 0
            misses = 0

            if left_is_aggressive:  # aggressive drivers are majority group in left data
                # calculate number of false alarms in left group and misses in right group at this threshold
                false_alarms = sum(driver['INTENT'] < 2 for driver in left_data)
                misses = sum(driver['INTENT'] == 2 for driver in right_data)
            else:  # aggressive drivers are majority group in right data
                # calculate number of false alarms in right group and misses in left group at this threshold
                false_alarms = sum(driver['INTENT'] < 2 for driver in right_data)
                misses = sum(driver['INTENT'] == 2 for driver in left_data)

            badness = false_alarms + misses
            threshold_information = Threshold_Information(badness, left_is_aggressive, 0)

            if best_threshold is None or badness < best_threshold.badness:
                best_threshold = threshold_information
                best_attribute = attribute
                best_left_data = left_data
                best_right_data = right_data

    return best_threshold, best_attribute

def main():
    drivers, attributes = read_all_files_in_project()
    best_attributes = ['Speed', 'BumperDamage', 'HasGlasses', 'RoofRack',
                       'SideDents', 'Wears_Hat', 'Brightness', 'NLaneChanges']  # decided using scatter plots

    safe_drivers = []
    aggressive_drivers = []

    for driver in drivers:
        if driver["INTENT"] == 2:
            aggressive_drivers.append(driver)
        else:
            safe_drivers.append(driver)

    # TODO: we should probably remove outliers before balancing the data?

    # Remove safe drivers until there is an equal number of safe and aggressive drivers (balance the data)
    for index in range(len(safe_drivers) - len(aggressive_drivers)):
        safe_drivers.pop(random.randrange(len(safe_drivers)))

    drivers = safe_drivers + aggressive_drivers

    best_threshold, best_attribute = create_tree(drivers, attributes[:-1], 0)
    print(f"Best attribute: {best_attribute}")
    print(f"Best threshold: {best_threshold.threshold}")

    # make_classifier(best_threshold, best_attribute)

    # scatter_all_attributes(drivers, attributes)

if __name__ == '__main__':
    main()