import numpy as np
from Data import Data

"""
CS383: Hw6
Instructor: Ian Gemp
TAs: Scott Jordan, Yash Chandak
University of Massachusetts, Amherst

README:

Feel free to make use of the function/libraries imported
You are NOT allowed to import anything else.

Following is a skeleton code which follows a Scikit style API.
Make necessary changes, where required, to get it correctly running.

Note: Running this empty template code might throw some error because 
currently some return values are not as per the required API. You need to
change them.

Good Luck!
"""


class Node(object):
    def __init__(self, att_idx=None, att_values=None, answer=None):
        self.att_idx = att_idx
        self.att_values = att_values
        self.branches = {}
        self.answer = answer

    def route(self, sample):
        if len(self.branches) < 1:
            return self.answer
        att = sample[self.att_idx]
        return self.branches[att].route(sample)


class DecisionTree:
    def __init__(self):
        self.root = None

    def fit(self, examples, attribute_names, attribute_values):
        """
        :param examples: input data, a list (len N) of lists (len num attributes)
        :param attribute_names: name of each attribute, list (len num attributes)
        :param attribute_values: possible values for each attribute, a list (len num attributes) of lists (len num values for each attribute)
        """
        self.attribute_names = attribute_names
        self.attribute_values = attribute_values
        self.attribute_idxs = dict(zip(attribute_names, range(len(attribute_names))))
        self.attribute_map = dict(zip(attribute_names, attribute_values))
        self.P = sum([example[-1] for example in examples])
        self.N = len(examples) - self.P
        self.H_Data = self.H(self.P / (self.P + self.N))  # this is B on p.704 - to be used in InfoGain
        self.root = self.DTL(examples, attribute_names)
        # self.ExpectedH('Pat', examples)
        # print(self.attribute_idxs)

    def DTL(self, examples, attribute_names, default=True):
        """
        Learn the decision tree.
        :param examples: input data, a list (len N) of lists (len num attributes)
        :param attribute_names: name of each attribute, list (len num attributes)
        :return: the root node of the decision tree
        """
        # WRITE the required CODE HERE and return the computed values
        if len(examples) == 0:
            return None
        elif self.is_same(examples):
            leaf = Node()
            leaf.answer = examples[0][-1]
            return leaf
        elif len(attribute_names) == 0:
            return None
        else:
            best_attribute = self.chooseAttribute(attribute_names, examples)
            tree = Node(self.attribute_idxs[best_attribute], self.attribute_map[best_attribute])
            partition_dict = self.partition_given_attribute(examples, best_attribute)

            # name is the name of the attribute value of the given attribute
            # value is dictionary of pos and neg, each is a list of list of training data
            for name, values in partition_dict.items():
                new_examples = values['pos'] + values['neg']
                subtree = self.DTL(new_examples, attribute_names)
                if subtree is not None:
                    # print(name)
                    tree.branches[name] = subtree

            return tree


    def mode(self, answers):
        """
        Compute the mode of a list of True/False values.
        :param answers: a list of boolean values
        :return: the mode, i.e., True or False
        """
        # WRITE the required CODE HERE and return the computed values
        return None

    def H(self, p):
        """
        Compute the entropy of a binary distribution.
        :param p: p, the probability of a positive sample
        :return: the entropy (float)
        """
        # WRITE the required CODE HERE and return the computed values
        pos_entropy = p * np.log2(p) if p > 0 else 0
        neg_entropy = (1 - p) * np.log2(1 - p) if p < 1 else 0
        return -(pos_entropy + neg_entropy)

    def ExpectedH(self, attribute_name, examples):
        """
        Compute the expected entropy of an attribute over its values (branches).
        :param attribute_name: name of the attribute, a string
        :param examples: input data, a list of lists (len num attributes)
        :return: the expected entropy (float)
        """
        # WRITE the required CODE HERE and return the computed values
        POS = 'pos'
        NEG = 'neg'
        partitions = self.partition_given_attribute(examples, attribute_name)
        total_entropy = 0.0
        for values in partitions.values():
            branch_total = len(values[POS]) + len(values[NEG])
            p = len(values[POS]) / branch_total if branch_total != 0 else 0
            total_entropy += (branch_total / len(examples)) * self.H(p)
        # print(total_entropy)
        return total_entropy

    def InfoGain(self, attribute_name, examples):
        """
        Compute the information gained by selecting the attribute.
        :param attribute_name: name of the attribute, a string
        :param examples: input data, a list of lists (len num attributes)
        :return: the information gain (float)
        """
        return self.H_Data - self.ExpectedH(attribute_name, examples)

    def chooseAttribute(self, attribute_names, examples):
        """
        Choose to split on the attribute with the highest expected information gain.
        :param attribute_names: name of each attribute, list (len num attributes)
        :param examples: input data, a list of lists (len num attributes)
        :return: the name of the selected attribute, string
        """
        InfoGains = []
        for att in attribute_names:
            InfoGains += [self.InfoGain(att, examples)]
        return attribute_names[np.argmax(InfoGains)]

    def predict(self, X):
        """
        Return your predictions
        :param X: inputs, shape:(N,num_attributes)
        :return: predictions, shape:(N,)
        """
        # WRITE the required CODE HERE and return the computed values
        # prediction should be a simple matter of recursively routing
        # a sample starting at the root node
        myroot  = self.root
        predict_result = []
        for x in X:
            while True:
                attribute = x[myroot.att_idx]
                if myroot.branches[attribute].answer is not None:
                    predict_result.append(myroot.branches[attribute].answer)
                    break
                else:
                    myroot = myroot.branches[attribute]
        return predict_result

    def print(self):
        """
        Print the decision tree in a readable format.
        """
        self.print_tree(self.root)

    def print_tree(self, node):
        """
        Print the subtree given by node in a readable format.
        :param node: the root of the subtree to be printed
        """
        if len(node.branches) < 1:
            print('\t\tanswer', node.answer)
        else:
            att_name = self.attribute_names[node.att_idx]
            for value, branch in node.branches.items():
                print('att_name', att_name, '\tbranch_value', value)
                self.print_tree(branch)

    def partition_given_attribute(self, examples, attribute_name):
        POS = 'pos'
        NEG = 'neg'
        partitions = {x: {POS: [], NEG: []} for x in self.attribute_map[attribute_name]}
        for x in examples:
            attribute = x[self.attribute_idxs[attribute_name]]
            if x[-1]:
                partitions[attribute][POS].append(x)
            else:
                partitions[attribute][NEG].append(x)

        return partitions

    def is_same(self, examples):
        initial = examples[0]
        for x in examples:
            if initial[-1] != x[-1]:
                return False

        return True

if __name__ == '__main__':
    # Get data
    data = Data()
    examples, attribute_names, attribute_values = data.get_decision_tree_data()

    # Decision tree trained with max info gain for choosing attributes
    model = DecisionTree()
    model.fit(examples, attribute_names, attribute_values)
    print(model.ExpectedH('Est',examples))
    # print(model.partition_given_attribute(examples,'Pat'))
    for key, value in model.partition_given_attribute(examples,'Est').items():
        print(key,len(value['pos']),len(value['neg']))
    y = model.predict(examples)
    print(y)
    # model.print()
