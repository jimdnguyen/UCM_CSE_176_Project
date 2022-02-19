from rgf.sklearn import RGFClassifier
from sklearn.model_selection import train_test_split
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(0)


def main():
    mat = scipy.io.loadmat("MNISTmini.mat")

    data_train_fea = np.array(mat["train_fea1"]).astype(np.float32)/255
    data_train_gnd = np.array(mat["train_gnd1"])

    data_test_fea = np.array(mat["test_fea1"]).astype(np.float32)/255
    data_test_fea = np.array(mat["test_gnd1"])

    data_train_x, data_valid_x, data_train_y, data_valid_y, = train_test_split(
        data_train_fea, data_train_gnd, test_size=0.10, random_state=0, shuffle=True)

    listtrainscore = []
    listvalidscore = []
    """
    Depth values [1,2,3,4,5,6,7,8,9]
    Validation Score: [0.9201666666666667, 0.8588333333333333, 0.842, 0.8306666666666667, 0.8266666666666667, 0.822, 0.817, 0.8116666666666666, 0.8076666666666666]
    Train Score: [0.9322037037037038, 0.8688703703703704, 0.8496296296296296, 0.8400185185185185, 0.8331481481481482, 0.8273518518518519, 0.8227222222222222, 0.8189814814814815, 0.8145740740740741]
    Best depth value is 1
    """
    for i in [10**-4, 10**-3, 10**-2, 10**-1]:
        rgf = RGFClassifier(
            max_leaf=400, algorithm="RGF_Opt", reg_depth=1, l2=i)
        rgf.fit(data_train_x, data_train_y.ravel())
        currentscore_val = rgf.score(
            data_valid_x, data_valid_y)
        currentscore_train = rgf.score(
            data_train_x, data_train_y)
        # currentscore_test = rgf.score(
        #     data_fea[test_idx, :], data_gnd[test_idx, :])
        listtrainscore.append(currentscore_train)
        listvalidscore.append(currentscore_val)
        # print(f"Validation Score: {currentscore_val}")
        # print(f"Train Score: {currentscore_train}")
        # print(f"Test Score: {currentscore_test}")
    print(f"Validation Score: {listvalidscore}")
    print(f"Train Score: {listtrainscore}")


if __name__ == "__main__":
    main()
