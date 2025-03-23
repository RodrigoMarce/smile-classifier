import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

def fPC (y, yhat):
    return np.mean(y == yhat)

def measureAccuracyOfPredictors (predictors, X, y):
    # Ensembles of predictors
    # X is the images
    # y is the truth values
    # Predictors is a list of functions that take in an image and return a prediction
    # For each image, we will apply each predictor and measure the accuracy of the predictor
    predictions = np.zeros((len(y)))
    voteForEach = np.zeros((len(y)))
    for predictor in predictors:
        r1, c1, r2, c2 = predictor #Coordinates
        Smile = ((X[:, r1, c1] - X[:, r2, c2]) > 0).astype(int) #Smile is 1 if the difference is greater than 0
        voteForEach += Smile  # Cumulative sum of votes for each image
    predictions = (voteForEach / len(predictors)) > 0.5 #Check each if majority vote
    return fPC(y, predictions)

def stepwiseRegression (trainingFaces, trainingLabels):
    # Stepwise regression
    predictors = [] #Array of predictors
    bestAccuracy = 0
    numParams = 6
    for m in range(numParams):
        bestAccuracy = 0
        for r1 in range(24):
            for c1 in range(24):
                for r2 in range(24):
                    for c2 in range(24):
                        if r1 == r2 and c1 == c2:
                            continue
                        if (r1, c1, r2, c2) in predictors:
                            continue
                        predictors.append((r1, c1, r2, c2))
                        accuracy = measureAccuracyOfPredictors(predictors, trainingFaces, trainingLabels)
                        predictors.pop()
                        if accuracy > bestAccuracy:
                            bestAccuracy = accuracy
                            bestPredictor = (r1, c1, r2, c2)
        predictors.append(bestPredictor)
    print("Best Predictors" + str(predictors))
    # (20, 7, 17, 7), (12, 5, 10, 13), (20, 17, 16, 17), (11, 19, 12, 12), (19, 11, 14, 7), (14, 5, 16, 6)]
    show = True
    if show:
        # Show an arbitrary test image in grayscale
        im = testingFaces[0,:,:]
        fig,ax = plt.subplots(1)
        ax.imshow(im, cmap='gray')
        for i, predictor in enumerate(predictors):
            r1, c1, r2, c2 = predictor
            # Show r1,c1
            rect1 = patches.Rectangle((c1 - 0.5, r1 - 0.5), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect1)
            ax.text(c1 - 1, r1, f'F{i+1}', color='red', fontsize=8, ha='center', va='center')
            # Show r2,c2
            rect2 = patches.Rectangle((c2 - 0.5, r2 - 0.5), 1, 1, linewidth=2, edgecolor='b', facecolor='none')
            ax.add_patch(rect2)
            ax.text(c2, r2 - 1, f'F{i+1}', color='blue', fontsize=8, ha='center', va='center')
        # Display the merged result
        plt.show()
    return predictors

def analyzePredictors(trainingFaces, trainingLabels, testingFaces, testingLabels):
    maxN = 2000
    minN = 200
    step = 200
    for N in range(minN, maxN+1, step):
        predictors = stepwiseRegression(trainingFaces[:N], trainingLabels[:N])
        trainAccuracy = measureAccuracyOfPredictors(predictors, trainingFaces[:N], trainingLabels[:N])
        testAccuracy = measureAccuracyOfPredictors(predictors, testingFaces, testingLabels)
        print("N = {}, Training Accuracy = {}, Testing Accuracy = {}".format(N, trainAccuracy, testAccuracy))
    

def loadData (which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels

if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")
    # below for image
    predictors = stepwiseRegression(trainingFaces[:2000], trainingLabels[:2000])
    # below for Ns
    analyzePredictors(trainingFaces, trainingLabels, testingFaces, testingLabels)