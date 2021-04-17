import os
import argparse


class BaseOption(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--debug', action='store_true', default=False, help='for checking code')
        self.parser.add_argument('--gpu_ids', type=int, default=0, help='gpu number. If -1, use cpu')
        self.parser.add_argument('--HD', action='store_true', default=True, help='if True, use pix2pixHD')
        self.parser.add_argument('--data_format_input', type=str, default='png',
                                 help="Input data extension. This will be used for loading and saving. [npy, png]")
        self.parser.add_argument('--data_format_target', type=str, default='npy',
                                 help="Target data extension.")

        # data option
        # dynamic range options are applied to fits, fts, and npy extension data. If an extension is png, jpeg, or jpg,
        # dynamic range options are not used.
        # When data are normlized, its normalized by 2 * dynamic range to ensure all the values are between [-1, 1]
        # after normalized.
        self.parser.add_argument('--dynamic_range_input', type=int, default=100, help="Dynamic range of input")
        self.parser.add_argument('--dynamic_range_target', type=int, default=1400, help="Dynamic range of target")

        # data augmentation
        self.parser.add_argument('--padding_size', type=int, default=0, help='padding size')
        self.parser.add_argument('--max_rotation_angle', type=int, default=0, help='rotation angle in degrees')

        self.parser.add_argument('--additional_name', type=str, default="test123", help="additional mark for checkpoint dir")
        self.parser.add_argument('--batch_size', type=int, default=1, help='the number of batch_size')
        self.parser.add_argument('--dataset_name', type=str, default='Over_0_std_0107', help='[dataset directory name')
        self.parser.add_argument('--data_type', type=int, default=32, help='float dtype')
        self.parser.add_argument('--image_height', type=int, default=1024, help='[512, 1024]')
        self.parser.add_argument('--image_mode', type=str, default='png', help='extension for saving image')
        self.parser.add_argument('--max_ch', type=int, default=1024, help='maximum number of channel for pix2pix')
        self.parser.add_argument('--n_downsample', type=int, default=4,
                                 help='how many times you want to downsample input data in G')
        self.parser.add_argument('--n_residual', type=int, default=9, help='the number of residual blocks in G')
        self.parser.add_argument('--n_workers', type=int, default=2, help='how many threads you want to use')
        self.parser.add_argument('--norm_type', type=str, default='InstanceNorm2d',
                                 help='[BatchNorm2d, InstanceNorm2d]')
        self.parser.add_argument('--padding_type', type=str, default='reflection',
                                 help='[reflection, replication, zero]')
        self.parser.add_argument('--val_during_train', action='store_true', default=False)

    def parse(self):
        opt = self.parser.parse_args()
        opt.input_ch = 3
        opt.n_df = 64
        opt.n_gf = 32  # 32 if opt.HD and (opt.image_height == 1024) else 64
        opt.output_ch = 3

        if opt.is_train:
            opt.format = 'png'

            opt.flip = False

            if opt.data_type == 16:
                opt.eps = 1e-4
            elif opt.data_type == 32:
                opt.eps = 1e-8

            dataset_name = opt.dataset_name
            model_name = "pix2pixHD" if opt.HD else 'pix2pix'
            model_name += "_padding" if opt.padding_size > 0 else ''
            model_name += "_rotation{}".format(str(opt.max_rotation_angle)) if opt.max_rotation_angle > 0 else ''
            model_name += "_{}".format(opt.data_format_target)
            model_name += opt.additional_name

            os.makedirs(os.path.join(dataset_name, 'Image', 'Training', model_name), exist_ok=True)
            os.makedirs(os.path.join(dataset_name, 'Image', 'Test', model_name), exist_ok=True)
            os.makedirs(os.path.join(dataset_name, 'Model', model_name), exist_ok=True)
            opt.image_dir = os.path.join(dataset_name, 'Image/Training', model_name)

            opt.model_dir = os.path.join(dataset_name, 'Model', model_name)
            log_path = os.path.join(dataset_name, 'Model', model_name, 'opt.txt')

            if opt.debug:
                opt.display_freq = 1
                opt.epoch_decay = 2
                opt.n_epochs = 4
                opt.report_freq = 1
                opt.save_freq = 1

                return opt

            if os.path.isfile(log_path) and opt.is_train:
                permission = 'Y' 
                # input(
                    # "{} log already exists. Do you really want to overwrite this log? Y/N. : ".format(model_name + '/opt'))
                if permission in ['y', 'Y', 'yes']:
                    pass
                else:
                    permission = 'Y'
                    # input("Do you want to resume training {}? Y/N. : ".format(model_name))
                    if permission in ['y', 'Y', 'yes']:
                        return opt

                    else:
                        raise NotImplementedError("Please check {}".format(log_path))

            with open(os.path.join(opt.model_dir, 'Analysis.txt'), 'wt') as analysis:
                analysis.write('Iteration, CorrCoef_TUMF, CorrCoef_1x1, CorrCoef_2x2, CorrCoef_4x4, CorrCoef_8x8, '
                               'R1_mean, R1_std, R2_mean, R2_std\n')

                analysis.close()

            args = vars(opt)
            with open(log_path, 'wt') as log:
                log.write('-' * 50 + 'Options' + '-' * 50 + '\n')
                print('-' * 50 + 'Options' + '-' * 50)
                for k, v in sorted(args.items()):
                    log.write('{}: {}\n'.format(str(k), str(v)))
                    print("{}: {}".format(str(k), str(v)))
                log.write('-' * 50 + 'End' + '-' * 50)
                print('-' * 50 + 'End' + '-' * 50)
                log.close()

        else:
            # opt.image_dir = os.path.join('./checkpoints', opt.dataset_name, 'Image/Test', opt.model_name)

            dataset_name = opt.dataset_name
            model_name = opt.model_name
            iteration = str(opt.iteration)

            if "HD" in model_name:
                opt.HD = True

            dir_model = '{}/Model/{}'.format(dataset_name, model_name)

            if os.path.isfile(os.path.join(dir_model, iteration + '_dict.pt')):
                opt.path_model = '{}/Model/{}/{}_dict.pt'.format(dataset_name, model_name, iteration)

            elif os.path.isfile(os.path.join(dir_model, iteration + '_G.pt')):
                opt.path_model = '{}/Model/{}/{}_G.pt'.format(dataset_name, model_name, iteration)

            else:
                raise FileNotFoundError

            opt.dir_image_save = '{}/Image/Test/{}/{}'.format(dataset_name, model_name, iteration)
            os.makedirs(opt.dir_image_save, exist_ok=True)

            if not opt.no_save_npy:
                opt.dir_npy_save = '{}/npy/Test/{}/{}'.format(dataset_name, model_name, iteration)
                os.makedirs(opt.dir_npy_save, exist_ok=True)

        return opt


class TrainOption(BaseOption):
    def __init__(self):
        super(TrainOption, self).__init__()

        self.parser.add_argument('--is_train', type=bool, default=True, help='train flag')
        self.parser.add_argument('--latest', type=int, default=0, help='Resume epoch')

        self.parser.add_argument('--beta1', type=float, default=0.5)
        self.parser.add_argument('--beta2', type=float, default=0.999)
        self.parser.add_argument('--display_freq', type=int, default=100)
        self.parser.add_argument('--epoch_decay', type=int, default=100, help='when to start decay the lr')
        self.parser.add_argument('--GAN_type', type=str, default='LSGAN', help='[GAN, LSGAN, WGAN_GP]')
        self.parser.add_argument('--lambda_FM', type=int, default=10, help='weight for FM loss')
        self.parser.add_argument('--lr', type=float, default=0.0002)
        self.parser.add_argument('--report_freq', type=int, default=5)
        self.parser.add_argument('--save_freq', type=int, default=5000)
        self.parser.add_argument('--no_shuffle', action='store_true', default=False, help='if you want to shuffle the order')
        self.parser.add_argument('--n_D', type=int, default=2,
                                 help='how many discriminators in differet scales you want to use')
        self.parser.add_argument('--n_epochs', type=int, default=200, help='how many epochs you want to train')
        self.parser.add_argument('--VGG_loss', action='store_true', default=False,
                                 help='if you want to use VGGNet for additional feature matching loss')
        self.parser.add_argument('--val_freq', type=int, default=5000)


class TestOption(BaseOption):
    def __init__(self):
        super(TestOption, self).__init__()

        self.parser.add_argument('--is_train', type=bool, default=False, help='test flag')
        self.parser.add_argument('--no_shuffle', type=bool, default=True, help='if you want to shuffle the order')
        self.parser.add_argument('--iteration', type=int, default=490000, help='the iteration of the model')
        self.parser.add_argument('--model_name', type=str, default="pix2pixHD_npy_corrected", help="the name of the model")
        self.parser.add_argument('--no_save_npy', action="store_true", default=False)