import os


def selectTrainDevice(useGpu):
    """
    Select cpu/gpu as train device (single GPU support)
    :param useGpu:
    :return:
    """
    if len(useGpu.split(",")) > 1:
        raise Exception("We support only 1 GPU")
    if useGpu is not None:
        trainDevice = '/gpu:0'
        print('Train device: ' + useGpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = useGpu
    else:
        trainDevice = '/cpu:0'
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
    return trainDevice
