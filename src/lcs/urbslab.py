import numpy as np
import pandas as pd
from skeLCS import eLCS
from sklearn.model_selection import cross_val_score


def main():
    # Load Data Using Pandas
    data = pd.read_csv("./6Multiplexer_Data_Complete.csv")
    classLabel = "Class"

    # DEFINE classLabel variable as the Str at the top of your dataset's class column
    dataFeatures = data.drop(classLabel, axis=1).values
    dataPhenotypes = data[classLabel].values

    # Shuffle Data Before CV
    formatted = np.insert(dataFeatures, dataFeatures.shape[1], dataPhenotypes, 1)
    np.random.shuffle(formatted)
    dataFeatures = np.delete(formatted, -1, axis=1)
    dataPhenotypes = formatted[:, -1]

    # Initialize eLCS Model
    model = eLCS(learning_iterations=5000)

    # 3-fold CV
    print(np.mean(cross_val_score(model, dataFeatures, dataPhenotypes, cv=3)))


if __name__ == "__main__":
    main()
