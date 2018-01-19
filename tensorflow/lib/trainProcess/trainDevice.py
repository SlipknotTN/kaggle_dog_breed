import os


def selectTrainDevice(useGpu):
    """
    Select cpu/gpu as train device
    :param useGpu:
    :return:
    """
    if useGpu is not None:
        trainDevice = '/gpu:' + useGpu
        print('Train device: ' + useGpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = useGpu
    else:
        trainDevice = '/cpu:0'
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
    return trainDevice
