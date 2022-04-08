class EpochData(object):
    """Output from each epoch"""

    epoch_number = 0
    loss = 0
    #validation loss
    time_to_complete = 0

    def __init__(self, epoch_number):
        super(EpochData, self).__init__()
        self.epoch_number = epoch_number

    def __str__(self):
        return ','.join([str(self.epoch_number), str(self.loss), str(self.time_to_complete)])


class EpochRecorder(object):

    epochs = []
    file_name = ''

    def __init__(self, file_name='EpochData.csv'):
        super(EpochRecorder, self).__init__()
        self.file_name = file_name

    def record(self, epoch_data, write=True):
        self.epochs.append(str(epoch_data))
        if write:
            self.write()

    def write(self):
        f = open(self.file_name, "a")
        f.write('\n'.join(self.epochs) + '\n')
        f.close()
