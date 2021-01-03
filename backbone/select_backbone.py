from .s3dg import S3D
from .resnet_2d3d import r2d3d50

def select_backbone(network, first_channel=3):
    param = {'feature_size': 1024}
    if network == 's3d':
        model = S3D(input_channel=first_channel)
    elif network == 's3dg':
        model = S3D(input_channel=first_channel, gating=True)
    elif network == 'r50':
        param['feature_size'] = 2048
        model = r2d3d50(input_channel=first_channel)
    else: 
        raise NotImplementedError

    return model, param
