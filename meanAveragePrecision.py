import numpy as np

#######################################################################################################################
# Function computeMeanAveragePrecision(labels,softmaxEstimations)
# Compute the Mean Average Precision of a trained Deep Neural Network on a testing set.
# 
# Input arguments:
#    - [list] labels: list of integers of length nbExamples containing the ground truth testing labels
#    - [array] softmaxEstimations: 2D numpy array of size nbExamples x nbClasses containing the softmax scores
#        obtained by the trained Deep Neural Network on the testing set
#
# Output arguments:
#    - [float] output1: global Mean Average Precision
#    - [array] output2: 1D array of size nbClasses containing the class Mean Average Precisions
#######################################################################################################################

def computeMeanAveragePrecision(labels,softmaxEstimations):

    nbExamples , nbClasses = softmaxEstimations.shape

    averagePrecisions = np.zeros((nbClasses),dtype=np.float32)

    # For all classes
    for classIdx in range(nbClasses):

        # Sort the softmaxEstimations by decreasing order, and keep the order consistent with the labels
        permutation = list(reversed(np.argsort(softmaxEstimations[:,classIdx])))
        labelArray = np.asarray(labels)
        labelsTmp = list(labelArray[permutation])

        # Convert the labels to binary (1-vs-all)
        for idx in range(len(labelsTmp)):
            if labelsTmp[idx] == classIdx:
                labelsTmp[idx] = 1
            else:
                labelsTmp[idx] = 0

        # Compute the averaged sum of precisions by descending order
        nbPrecisionComputations = 0
        averagePrecisionSum = 0

        for idx in range(len(labelsTmp)):
            if labelsTmp[idx] == 1:
                averagePrecisionSum += np.sum(labelsTmp[:idx+1])/float(idx+1)
                nbPrecisionComputations += 1

        if nbPrecisionComputations == 0:
             averagePrecisions[classIdx] = 0
        else:
            averagePrecisions[classIdx] = averagePrecisionSum/float(nbPrecisionComputations)

    # Return global MAP and class MAPs
    return np.mean(averagePrecisions), averagePrecisions

