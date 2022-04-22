import csv
import matplotlib.pyplot as plt

class TrainingMetadata(object):

    epochs = []
    file_name = ''

    def __init__(self, file_name):
        super(TrainingMetadata, self).__init__()
        self.file_name = file_name + '.csv'

    def record(self, loss, validation_loss):
        line = ','.join([str(loss), str(validation_loss)])
        self.epochs.append(line)

    def write(self):
        with open(self.file_name, "w+") as f:
            f.write('\n'.join(self.epochs))
            f.close()

    def read(self):
        self.epochs = []
        with open(self.file_name) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                self.epochs.append(row)

    def plot(self):
        losses = [float(i[0]) for i in self.epochs]
        validation_losses = [float(i[1]) for i in self.epochs]
        epochs = range(len(self.epochs))
        plt.plot(epochs, losses, epochs, validation_losses)
        plt.show()

    #def plot(self):
if __name__ == "__main__":
    t = TrainingMetadata('EpochData/epoch_data_test')
    t.read()
    t.plot()
