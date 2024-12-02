import OCR.local_config as local_config
import torch
from ovotools import AttrDict

settings = AttrDict(
    max_epochs=100,
    tensorboard_port=6006,
    # device='cuda:3',
    # device='cpu',
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    findLR=False,
    # can_overwrite=False,
    can_overwrite=True,
)

params = AttrDict(
    data_root = local_config.data_path,
    # model_name = 'NN_results/dsbi_lay{model_params.num_fpn_layers}',
    model_name = 'weights/model.t7',
    data = AttrDict(
        get_points = False,
        class_as_6pt=False,    # классификация присутствия каждой точки в рамке отдельно
        batch_size = 2,    # 12 -> 2
        net_hw = (416, 416),
        rect_margin = 0.3, #  every of 4 margins to char width
        max_std = 0.1,
        train_list_file_names = [
            #r'DSBI/data/val_li2.txt',
            # r'DSBI/data/train_li2.txt',
            r'data_train/train_image_list.txt',
        ],
        val_list_file_names = {
            # 'val' :  [r'DSBI/data/val_li2.txt',],
            # 'test' :  [r'DSBI/data/test_li2.txt',]
            'val': [r'data_train/train_image_list.txt',],
        }
    ),
    augmentation = AttrDict(
        img_width_range=( 1024, 1376, ),  # 810, 890 -> 1024, 1376
        stretch_limit = 0.1,
        rotate_limit = 5,
    ),
    model = 'retina',
    model_params = AttrDict(
        # num_fpn_layers=5,
        encoder_params = AttrDict(
            anchor_areas=[ 128.0, 288.0, 512.0, ], # [34*55/4,], -> [ 128.0, 288.0, 512.0, ],
            aspect_ratios=[0.62,],  # [0.62,], -> [ 0.5, ],
            # scale_ratios=[1., 2 ** (1 / 3.), 2 ** (2 / 3.)],
            # scale_ratios=[1.],
            iuo_fit_thr = 0, # if iou > iuo_fit_thr => rect fits anchor
            iuo_nofit_thr = 0,
        ),
        loss_params=AttrDict(
            class_loss_scale = 100, # 1 -> 100
        ),
    ),
    #load_model_from = 'NN_results/dsbi_tst1_lay5_083746/models/clr.003.t7',  # retina_chars_d58e5f # retina_chars_7e1d4e
    load_model_from = 'weights/model.t7',
    optim = 'torch.optim.Adam',
    # optim = 'torch.optim.SGD',
    optim_params = AttrDict(
        # Adam
        lr=0.0001,
        
        #momentum=0.9,
        #weight_decay = 0, #0.001,
        #nesterov = False,
        
        # SGD
        # lr = 1e-4,
        # momentum = 0.9,
        
        # weight_decay = 1e-4,
    ),
    lr_finder=AttrDict(
        iters_num=200,
        log_lr_start=-4, # -5 -> -4
        log_lr_end=-0.3,   # -1 -> -0.3
    ),
    lr_scheduler=AttrDict(
        type='clr',
        #params=AttrDict(
        #    milestones=[5000, 10000,],
        #    gamma=0.1,
        #),
    ),
    clr=AttrDict(
        warmup_epochs=10,
        min_lr=1e-5,        # 1e-5 -> 0.0001
        max_lr=0.0001,      # 0.0002 -> 0.0001
        period_epochs=500,
        scale_max_lr=0.95,
        scale_min_lr=0.95,
    ),
)