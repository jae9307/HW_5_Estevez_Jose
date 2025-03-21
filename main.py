import copy
import os
import re
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns

# Defines a Threshold_Information object which stores the best threshold for a particular attribute, as well as the threshold's
# badness and whether aggressive drivers are the majority class in the left group of data
class Threshold_Information:
    def __init__(self, gain_ratio, left_is_aggressive_flag, threshold, attribute):
        self.gain_ratio = gain_ratio
        self.left_is_aggressive_flag = left_is_aggressive_flag
        self.threshold = threshold
        self.attribute = attribute

    def to_string(self, depth=0):
        indent = "    " * depth
        return (f"{indent}Threshold Information:\n"
                f"  {indent}Left is Aggressive Flag: {self.left_is_aggressive_flag}\n"
                f"  {indent}Threshold: {self.threshold}\n"
                f"  {indent}Attribute: {self.attribute}\n")

class Node:
    def __init__(self, threshold_information, left, right, drivers):
        self.threshold_information = threshold_information
        self.left = left
        self.right = right
        self.drivers = drivers
    def to_string(self, depth=0):
        indent = "    " * depth
        left_str = self.left.to_string(depth + 1) if self.left else f"    {indent}None\n"
        right_str = self.right.to_string(depth + 1) if self.right else f"    {indent}None\n"
        return (f"{indent}Node:\n"
                f"  {indent}{self.threshold_information.to_string(depth)}"
                f"  {indent}Drivers: {len(self.drivers)}\n"
                f"  {indent}Left: \n{left_str}"
                f"  {indent}Right: \n{right_str}")

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
            driver_dictionary = dict()
            for index in range(len(attributes)):
                attribute_value = int(np.round(float(entries[index])))
                driver_dictionary.update(dict([(attributes[index], attribute_value)]))
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


# Recursive function which finds the last index in the array of drivers which has a value <= the threshold value for a specific
# attribute. Starts by checking the value at the middle index of the array. If the value is > threshold, the function is called
# again, this time checking at the index in the middle of the group of values to the left of the current index. If the value
# is <= threshold, the function is called again checking at the index in middle of the group of values to the right of the current
# index. The function recursively calls itself in this way until the last instance (index wise) of a value <= the threshold is
# found.
def find_threshold_index(index, drivers, threshold, current_best_index, attribute):
    if drivers[index][attribute] > threshold:
        if index == 0:  # if first element after the previous midpoint is greater than threshold, then the previous midpoint must be the best
            return current_best_index
        return find_threshold_index(math.ceil(drivers[:index].__len__() / 2) - 1, drivers[:index], threshold,
                                    current_best_index, attribute)
    else:
        if drivers.__len__() - 1 == index:  # only happens when there is only driver left (length of drivers is 1)
            return index + 1 + current_best_index  # since the current driver has value <= threshold, it is the
            # one we're looking for, and its index is 1 more than the previous midpoint (index is currently 0)
            # (i guess the condition should be changed to 'if drivers.len == 1' and the return should be changed
            # to 'return current_best_index + 1' but im paranoid to change it since it works as is)
        return find_threshold_index(math.ceil(drivers[index + 1:].__len__() / 2) - 1, drivers[index + 1:], threshold,
                                    index if index > current_best_index else current_best_index + index + 1, attribute)

# Find the threshold with the most gain ratio for a given attribute. Return a Threshold_Information object
def find_best_threshold(drivers, attribute, min_split_size=5):
    lowest_value = drivers[0][attribute]
    highest_value = drivers[-1][attribute]

    best_gain_ratio = -1
    left_is_aggressive_flag = False  # tracks if the dominant class in the left group of data is aggressive drivers
    best_threshold = -1

    combined_node_labels = [driver['INTENT'] for driver in drivers]

    for threshold in range(lowest_value, highest_value):
        threshold_index = find_threshold_index(math.ceil(drivers.__len__() / 2) - 1, drivers, threshold, -1, attribute)

        # create the groups left and right of the threshold using the threshold_index
        left_data = drivers[:threshold_index + 1]
        right_data = drivers[threshold_index + 1:]

        # Check if the number of records is at least 5 in either left or right node if not ignore the current threshold
        if len(left_data) < min_split_size or len(right_data) < min_split_size:
            continue

        left_labels = [driver['INTENT'] for driver in left_data]
        right_labels = [driver['INTENT'] for driver in right_data]

        # calculate if aggressive drivers are the majority group in the left data
        aggressive_drivers_in_left_data = sum(driver['INTENT'] == 2 for driver in left_data)
        left_is_aggressive = aggressive_drivers_in_left_data > len(left_data) - aggressive_drivers_in_left_data

        current_gain_ratio = gain_ratio(combined_node_labels, left_labels, right_labels)

        if current_gain_ratio > best_gain_ratio:
            best_gain_ratio = current_gain_ratio
            left_is_aggressive_flag = left_is_aggressive
            best_threshold = threshold

    return Threshold_Information(best_gain_ratio, left_is_aggressive_flag, best_threshold, attribute)

def gain_ratio(combined_labels, left_labels, right_labels):
    combined_entropy = calculate_entropy(combined_labels)
    left_entropy = calculate_entropy(left_labels)
    right_entropy = calculate_entropy(right_labels)

    total = len(left_labels) + len(right_labels)
    avg_left = len(left_labels) / total
    avg_right = len(right_labels) / total
    avg_entropy_of_two_nodes = left_entropy * avg_left + right_entropy * avg_right

    gain_split = combined_entropy - avg_entropy_of_two_nodes
    split_info_left = avg_left * np.log2(avg_left)
    split_info_right = avg_right * np.log2(avg_right)
    split_info = -(split_info_left + split_info_right)

    return gain_split / split_info

def calculate_entropy(labels):
    i, count_of_each_label = np.unique(labels, return_counts=True)
    probabilities = count_of_each_label / count_of_each_label.sum()
    return -np.sum(probabilities * np.log2(probabilities))

def make_classifier(tree):
    classifer_program = f"""
import os
import re
import numpy as np
import argparse

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
                    attributes.append(word.replace("\\n", ""))
                continue

            # Create a dictionary for each driver, where the key is the attribute name and
            # the value is the attribute value
            driver_dictionary = dict()
            for index in range(len(attributes) - 1):
                attribute_value = int(np.round(float(entries[index])))
                driver_dictionary.update(dict([(attributes[index], attribute_value)]))
            drivers.append(driver_dictionary)

    return drivers

def classifier():
    parser = argparse.ArgumentParser(prog='classifier', description='Classifies drivers as safe or aggressive')
    parser.add_argument('filename')
    args = parser.parse_args()

    drivers = read_file(args.filename)

    classification_file = open("HW_05_Estevez_Rigoglioso_MyClassifications.csv", "w")
    classification_file.write("")
    classification_file.close()
    for driver in drivers:
    """

    main_string = """
def main():
    classifier()

if __name__ == '__main__':
    main()
    """

    classifer_program += "\n" + recursive_classifier_program(tree, 0)

    file = open("HW_05_Classifier_Estevez_Jose_and_Rigoglioso_Dan.py", "w")
    file.write(classifer_program)
    file.write("\n" + main_string)
    file.close()

def leaf_node(node, indent):
    num_aggressive = sum(driver['INTENT'] == 2 for driver in node.drivers)
    num_safe = sum(driver['INTENT'] < 2 for driver in node.drivers)

    classify_str = f"""
            {indent}intent = f"{2 if num_aggressive > num_safe else 1}"
            {indent}print(intent)
            {indent}classification_file = open("HW_05_Estevez_Rigoglioso_MyClassifications.csv", "a")
            {indent}classification_file.write(str(intent) + "\\n")
            {indent}classification_file.close()
            """

    return f"{classify_str}\n"

# creates the nested if else statements for the attributes in the decision tree
def recursive_classifier_program(node, depth=0):
    indent = "    " * depth
    if node is None:
        return ""

    if node.left is None and node.right is None:
        return leaf_node(node, indent)

    conditional_left_aggressive = (f"driver['{node.threshold_information.attribute}'] <= {node.threshold_information.threshold}")
    conditional_right_aggressive = (f"driver['{node.threshold_information.attribute}'] > {node.threshold_information.threshold}")

    left_tree = recursive_classifier_program(node.left, depth + 1)
    right_tree = recursive_classifier_program(node.right, depth + 1) # Do we need to increase the depth both times?

    right_else_string = f"\n        {indent}else:\n{right_tree}"
    left_else_string = f"\n        {indent}else:\n{left_tree}"
    left_agg = f"left_is_agg: {node.threshold_information.left_is_aggressive_flag}"
    left_child: Node = node.left
    right_child: Node = node.right
    left_drivers = f"left_amt: {sum(driver[node.threshold_information.attribute] <= node.threshold_information.threshold for driver in node.drivers)}"
    right_drivers = f"right_amt: {sum(driver[node.threshold_information.attribute] > node.threshold_information.threshold for driver in node.drivers)}"
    recursive_string = ""
    if node.threshold_information.left_is_aggressive_flag:
        recursive_string = f"        {indent}if {conditional_left_aggressive}:\n{left_tree if left_tree else leaf_node(node, indent)}{right_else_string if right_tree else ''}"
    else:
        recursive_string = f"        {indent}if {conditional_right_aggressive}:\n{right_tree if right_tree else leaf_node(node, indent)}{left_else_string if left_tree else ''}"

    return recursive_string

def scatter_all_attributes(drivers, attributes):
    colors = {0: 'blue', 1: 'blue', 2: 'red'}
    for i in range(len(attributes) - 1):
        for j in range(i + 1, len(attributes) - 1):
            attribute_name = attributes[i]
            second_attribute_name = attributes[j]
            first_attributes_values = [driver[attributes[i]] + np.random.uniform(-0.5, 0.5) for driver in drivers]
            second_attributes_values = [driver[attributes[j]] + np.random.uniform(-0.5, 0.5) for driver in drivers]
            color_values = [colors[int(driver['INTENT'])] for driver in drivers]
            sns.scatterplot(x=first_attributes_values, y=second_attributes_values, hue=color_values,
                            palette={'blue': 'blue', 'red': 'red'}, alpha=0.6, edgecolor='w')
            plt.title(f"{attribute_name} vs {second_attribute_name}")
            plt.savefig(f"{attribute_name}_vs_{second_attribute_name}.png")
            plt.clf()

def create_tree(drivers, attributes, depth):
    # TODO: implement stopping condition
    if depth > 8 or len(drivers) == 0:
        return None
    fraction_aggressive = sum(driver['INTENT'] == 2 for driver in drivers) / len(drivers)
    if fraction_aggressive >= 0.9 or fraction_aggressive <= 0.1:
        return None

    best_threshold = None
    best_left_data = []
    best_right_data = []
    min_split_size = 5

    for attribute in attributes:
        sorted_drivers = sorted(drivers, key=lambda single_driver: single_driver[attribute])

        multi_value_attributes = ['Speed', 'NLaneChanges', 'Brightness', 'NumDoors']

        if multi_value_attributes.__contains__(attribute):
            threshold_information = find_best_threshold(sorted_drivers, attribute)

            if best_threshold is None or threshold_information.gain_ratio > best_threshold.gain_ratio:
                best_threshold = threshold_information
                best_left_data = list(filter(lambda driver: driver[attribute] <= threshold_information.threshold,
                                             drivers))
                best_right_data = list(filter(lambda driver: driver[attribute] > threshold_information.threshold,
                                              drivers))
        else:  # most attributes only have values of 1 or 0
            left_data = list(filter(lambda driver: driver[attribute] == 0, drivers))
            right_data = list(filter(lambda driver: driver[attribute] == 1, drivers))

            # calculate if aggressive drivers are the majority group in the left data
            aggressive_drivers_in_left_data = sum(driver['INTENT'] == 2 for driver in left_data)
            left_is_aggressive = aggressive_drivers_in_left_data > len(left_data) - aggressive_drivers_in_left_data

            if len(left_data) < min_split_size or len(right_data) < min_split_size:
                return None

            left_labels = [driver['INTENT'] for driver in left_data]
            right_labels = [driver['INTENT'] for driver in right_data]

            combined_labels = [driver['INTENT'] for driver in drivers] #This might have to be moved outside of the for loop
            current_gain_ratio = gain_ratio(combined_labels, left_labels, right_labels)

            threshold_information = Threshold_Information(current_gain_ratio, left_is_aggressive, 0, attribute)

            if best_threshold is None or current_gain_ratio > best_threshold.gain_ratio:
                best_threshold = threshold_information
                best_left_data = left_data
                best_right_data = right_data
    # If the top level finds speed is the best attribute for example then the inner levels should not use speed as an attribute

    attributes.remove(best_threshold.attribute)
    copied_attributes = copy.deepcopy(attributes)

    left_node = create_tree(best_left_data, copied_attributes, depth + 1)
    right_node = create_tree(best_right_data, copied_attributes, depth + 1)
    parent_node = Node(best_threshold, left_node, right_node, drivers)

    return parent_node

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

    decision_tree = create_tree(drivers, attributes[:-1], 0)
    print(decision_tree.to_string(0))
    test = 1

    make_classifier(decision_tree)

    # scatter_all_attributes(drivers, attributes)

if __name__ == '__main__':
    main()