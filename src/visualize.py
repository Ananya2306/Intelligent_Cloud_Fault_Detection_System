import matplotlib.pyplot as plt

def plot(data):
    data_sample = data.head(2000)

    plt.figure(figsize=(10,5))
    plt.plot(data_sample['cpu'], label='CPU')

    plt.scatter(range(len(data_sample)),
                data_sample['cpu'],
                c=data_sample['rf_pred'],
                cmap='coolwarm',
                label='Fault')

    plt.title("Fault Detection")
    plt.legend()

    plt.savefig("outputs/graphs/result.png")
    plt.show()

    print("Graph saved")