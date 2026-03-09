import matplotlib.pyplot as plt


def plot_soc(true_soc, pred_soc):

    plt.figure()

    plt.plot(true_soc, label="True SOC")
    plt.plot(pred_soc, label="Predicted SOC")

    plt.legend()

    plt.title("SOC Tracking")

    plt.show()