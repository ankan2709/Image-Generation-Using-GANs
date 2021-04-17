if __name__ == '__main__':
    import os
    import torch
    import numpy as np
    from options import TestOption
    from pipeline import CustomDataset
    from networks import Generator
    from utils import Manager, binning_and_cal_pixel_cc
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    ITERATION = 470000
    STD = 0
    #abcsafdasdf

    MODEL_NAME = 'pix2pix'
    torch.backends.cudnn.benchmark = True

    dir_input = './datasets/Over_{}_std/Test/Input'.format(str(STD))
    dir_target = './datasets/Over_{}_std/Test/Target'.format(str(STD))
    dir_model = './checkpoints/Over_{}_std/Model/{}'.format(str(STD), MODEL_NAME)
    path_model = './checkpoints/Over_{}_std/Model/{}/{}_G.pt'.format(str(STD), MODEL_NAME, str(ITERATION))

    dir_image_save = './checkpoints/Over_{}_std/Image/Test/{}/{}'.format(str(STD), MODEL_NAME, str(ITERATION))
    os.makedirs(dir_image_save, exist_ok=True)

    opt = TestOption().parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_ids)
    device = torch.device('cuda:0')

    dataset = CustomDataset(opt)
    test_data_loader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=False)

    G = Generator(opt).to(device)
    G.load_state_dict(torch.load(path_model))

    manager = Manager(opt)

    list_TUMF_fake = list()
    list_TUMF_real = list()

    list_cc_1x1_fake = list()
    list_cc_1x1_real = list()
    list_cc_1x1 = list()
    list_cc_bin_2x2 = list()
    list_cc_bin_4x4 = list()
    list_cc_bin_8x8 = list()
    list_R1 = list()
    list_R2 = list()

    circle_index = list()
    k = 0
    for i in range(1024):
        for j in range(1024):
            if (i - 511) ** 2 + (j - 511) ** 2 <= 392 ** 2:
                circle_index.append(k)
            k += 1
    with torch.no_grad():
        G.eval()
        for input, target, _, name in tqdm(test_data_loader):
            input = input.to(device)
            fake = G(input)
            manager.save_image(fake, path=os.path.join(dir_image_save, name[0] + '_fake.png'))
            manager.save_image(target, path=os.path.join(dir_image_save, name[0] + '_real.png'))

    #         # Model measurements
    #         np_fake = fake.cpu().numpy().squeeze() * 100.
    #         np_real = target.cpu().numpy().squeeze() * 100.
    #         np_fake_flatten, np_real_flatten = np_fake.flatten(), np_real.flatten()
    #         # rearrange [-100, 100]
    #         carrier_fake, carrier_real = list(), list()
    #
    #         for i in circle_index:
    #             list_cc_1x1_fake.append(np_fake_flatten[i])
    #             list_cc_1x1_real.append(np_real_flatten[i])
    #             if abs(np_fake_flatten[i]) >= 10.:
    #                 carrier_fake.append(abs(np_fake_flatten[i]))
    #             if abs(np_real_flatten[i]) >= 10.:
    #                 carrier_real.append(abs(np_real_flatten[i]))
    #
    #         TUMF_fake, TUMF_real = np.array(carrier_fake).sum(), np.array(carrier_real).sum()
    #         list_TUMF_fake.append(TUMF_fake)
    #         list_TUMF_real.append(TUMF_real)
    #         list_R1.append((TUMF_fake - TUMF_real) / TUMF_real)
    #
    #         list_cc_1x1.append(np.corrcoef(list_cc_1x1_fake, list_cc_1x1_real)[0][1])
    #         list_R2.append(((np.array(list_cc_1x1_fake) - np.array(list_cc_1x1_real)) ** 2).sum() / (
    #                          np.array(list_cc_1x1_real) ** 2).sum())
    #
    #         # list_cc_bin_2x2.append(binning_and_cal_pixel_cc(np_fake, np_real, 2))
    #         # list_cc_bin_4x4.append(binning_and_cal_pixel_cc(np_fake, np_real, 4))
    #         list_cc_bin_8x8.append(binning_and_cal_pixel_cc(np_fake, np_real, 8))
    #
    #         del input, target, fake, np_fake, np_real, np_fake_flatten, np_real_flatten, carrier_fake, carrier_real
    #         del TUMF_fake, TUMF_real, _, name
    #
    # cc_TUMF = np.corrcoef(list_TUMF_fake, list_TUMF_real)[0][1]
    # cc_1x1 = np.mean(list_cc_1x1)
    # # cc_bin_2x2 = np.mean(list_cc_bin_2x2)
    # # cc_bin_4x4 = np.mean(list_cc_bin_4x4)
    # cc_bin_8x8 = np.mean(list_cc_bin_8x8)
    #
    # R1_mean = np.mean(list_R1)
    # R1_std = np.std(list_R1)
    #
    # R2_mean = np.mean(list_R2)
    # R2_std = np.std(list_R2)
    #
    # with open(os.path.join(dir_image_save, 'Analysis.txt'), 'wt') as analysis:
    #     analysis.write(str(ITERATION) + ', ' + str(cc_TUMF) + ', ' + str(cc_1x1) + ', ' +
    #                    # str(cc_bin_2x2) + ', ' + str(cc_bin_4x4) + ', ' +
    #                    str(cc_bin_8x8) + ', ' +
    #                    str(R1_mean) + ', ' + str(R1_std) + ', ' + str(R2_mean) + ', ' + str(R2_std) + '\n')
    #     analysis.close()
