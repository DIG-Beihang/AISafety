attack = dict(
    type='RFGSM',
    params=[
        dict(name='rfgsm_01', epsilon=0.1, alpha=0.5),
        dict(name='rfgsm_02', epsilon=0.15, alpha=0.5),
        dict(name='rfgsm_03', epsilon=0.2, alpha=0.5),
        dict(name='rfgsm_04', epsilon=0.25, alpha=0.5),
        dict(name='rfgsm_05', epsilon=0.3, alpha=0.5),
        dict(name='rfgsm_06', epsilon=0.35, alpha=0.5),
        dict(name='rfgsm_07', epsilon=0.4, alpha=0.5),
    ],
    IS_WHITE=True,
    IS_PYTORCH_WHITE=False,
    IS_DOCKER_BLACK=False,
    IS_TARGETED=False,
    IS_COMPARE_MODEL=False,
)
defense = dict(
    model='Models.UserModel.FP_resnet',
    path='Models/weights/FP_ResNet20.th'
)
evaluation = dict(
    type="ACAC"
)
model = dict(
    name="ResNet20",
    path='Models.UserModel.ResNet2',
    weights='Models/weights/resnet20_cifar.pt'
)
datasets = dict(
    type='cifar10',
    dict_path='test/dict_lists/cifar10_dict.txt',
    test_path=[
        "Datasets/CIFAR_cln_data/cifar10_30_origin_inputs.npy",
        "Datasets/CIFAR_cln_data/cifar10_30_origin_labels.npy",
        "Datasets/CIFAR_cln_data/cifar10_30_origin_inputs.npy",
        "Datasets/CIFAR_cln_data/cifar10_30_origin_labels.npy",
    ],
    augment=dict(
        Crop_ImageSize=(32, 32),
        Scale_ImageSize=(32, 32)
    ),
    batch_size=4,
    CAM_layer=28,
)
result = dict(
    IS_SAVE=False,
    save_path='test/Attack_generation/',
    save_method='.npy',
    save_visualization_base_path='test/temp/',
    black_Result_dir='..',
)
gpu = dict(
    nums=2,
    index='0,1'
)
