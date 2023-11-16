import matplotlib.pyplot as plt

@staticmethod
def plot_features(X_train_minmax):
    plt.title("Features MinMaxScaler")
    plt.xlabel("X MinMaxScaler")
    plt.ylabel("Features")
    plt.plot(X_train_minmax, color="green")
    plt.show()