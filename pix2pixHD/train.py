if __name__ == '__main__':
    import os
    import torch
    from torch.utils.data import DataLoader
    from networks import Discriminator, Generator, Loss
    from options import TrainOption
    from pipeline import CustomDataset
    from utils import binning_and_cal_pixel_cc, Manager, update_lr, weights_init
    import numpy as np
    from tqdm import tqdm
    import datetime

    torch.backends.cudnn.benchmark = True

    opt = TrainOption().parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_ids)

    device = torch.device('cuda:0' if opt.gpu_ids != -1 else 'cpu:0')
    dtype = torch.float16 if opt.data_type == 16 else torch.float32
    image_height = opt.image_height
    radius = 392 if image_height == 1024 else 196

    if opt.val_during_train:
        from options import TestOption
        test_opt = TestOption().parse()
        val_freq = opt.val_freq

    init_lr = opt.lr
    lr = opt.lr

    dataset = CustomDataset(opt)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=opt.batch_size,
                             num_workers=opt.n_workers,
                             shuffle=not opt.no_shuffle)

    G = Generator(opt).apply(weights_init).to(device=device, dtype=dtype)
    D = Discriminator(opt).apply(weights_init).to(device=device, dtype=dtype)

    criterion = Loss(opt)

    G_optim = torch.optim.Adam(G.parameters(), lr=lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)
    D_optim = torch.optim.Adam(D.parameters(), lr=lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)


    if opt.latest: # and os.path.isfile(opt.model_dir + '/' + str(opt.latest) + '_dict.pt'):  # _dict.pt
        pt_file = torch.load(opt.model_dir + '/' + str(opt.latest) + '_dict.pt')   # _dict.pt
        init_epoch = pt_file['Epoch']
        print("Resume at epoch: ", init_epoch)
        G.load_state_dict(pt_file['G_state_dict'])
        D.load_state_dict(pt_file['D_state_dict'])
        G_optim.load_state_dict(pt_file['G_optim_state_dict'])
        D_optim.load_state_dict(pt_file['D_optim_state_dict'])
        current_step = init_epoch * len(dataset)

        for param_group in G_optim.param_groups:
            lr = param_group['lr']

    else:
        init_epoch = 1
        current_step = 0

    manager = Manager(opt)

    total_step = opt.n_epochs * len(data_loader)
    start_time = datetime.datetime.now()
    for epoch in range(init_epoch, opt.n_epochs + 1):
        for input, target, _, _ in tqdm(data_loader):
            G.train()

            current_step += 1
            input, target = input.to(device=device, dtype=dtype), target.to(device, dtype=dtype)

            D_loss, G_loss, target_tensor, generated_tensor = criterion(D, G, input, target)

            G_optim.zero_grad()
            G_loss.backward()
            G_optim.step()

            D_optim.zero_grad()
            D_loss.backward()
            D_optim.step()

            package = {'Epoch': epoch,
                       'current_step': current_step,
                       'total_step': total_step,
                       'D_loss': D_loss.detach().item(),
                       'G_loss': G_loss.detach().item(),
                       'D_state_dict': D.state_dict(),
                       'G_state_dict': G.state_dict(),
                       'D_optim_state_dict': D_optim.state_dict(),
                       'G_optim_state_dict': G_optim.state_dict(),
                       'target_tensor': target_tensor,
                       'generated_tensor': generated_tensor.detach()}

            manager(package)
            if opt.val_during_train and (current_step % val_freq == 0):
                G.eval()
                test_image_dir = os.path.join(test_opt.image_dir, str(current_step))
                os.makedirs(test_image_dir, exist_ok=True)
                test_model_dir = test_opt.model_dir

                test_dataset = CustomDataset(test_opt)
                test_data_loader = DataLoader(dataset=test_dataset,
                                              batch_size=test_opt.batch_size,
                                              num_workers=test_opt.n_workers,
                                              shuffle=not test_opt.no_shuffle)

                for p in G.parameters():
                    p.requires_grad_(False)

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
                for input, target, _, name in tqdm(test_data_loader):
                    input, target = input.to(device=device, dtype=dtype), target.to(device, dtype=dtype)
                    fake = G(input)
                    manager.save_image(fake, path=os.path.join(test_image_dir, name[0] + '_fake.png'))
                    manager.save_image(target, path=os.path.join(test_image_dir, name[0] + '_real.png'))

                    # Model measurements
                    bin_size = 8
                    np_fake, np_real = fake.cpu().numpy().squeeze() * 100., target.cpu().numpy().squeeze() * 100.
                    # rearrange [-100, 100]
                    carrier_fake = list()
                    carrier_real = list()
                    for i in range(image_height):
                        for j in range(image_height):
                            if (i - 511) ** 2 + (j - 511) ** 2 <= 392 ** 2:
                                list_cc_1x1_fake.append(np_fake[i, j])
                                list_cc_1x1_real.append(np_real[i, j])
                                if abs(np_fake[i, j]) >= 10:
                                    carrier_fake.append(abs(np_fake[i, j]))
                                if abs(np_real[i, j]) >= 10:
                                    carrier_real.append(abs(np_real[i, j]))
                    TUMF_fake, TUMF_real = np.array(carrier_fake).sum(), np.array(carrier_real).sum()
                    list_TUMF_fake.append(TUMF_fake)
                    list_TUMF_real.append(TUMF_real)
                    list_R1.append((TUMF_fake - TUMF_real) / TUMF_real)

                    list_cc_1x1.append(np.corrcoef(list_cc_1x1_fake, list_cc_1x1_real)[0][1])
                    list_R2.append(((np.array(list_cc_1x1_fake) - np.array(list_cc_1x1_real)) ** 2).sum() / (np.array(list_cc_1x1_real) ** 2).sum())

                    list_cc_bin_2x2.append(binning_and_cal_pixel_cc(np_fake, np_real, 2))
                    list_cc_bin_4x4.append(binning_and_cal_pixel_cc(np_fake, np_real, 4))
                    list_cc_bin_8x8.append(binning_and_cal_pixel_cc(np_fake, np_real, 8))

                cc_TUMF = np.corrcoef(np.array(list_TUMF_fake), np.array(list_TUMF_real))
                cc_1x1 = np.mean(list_cc_1x1)
                cc_bin_2x2 = np.mean(list_cc_bin_2x2)
                cc_bin_4x4 = np.mean(list_cc_bin_4x4)
                cc_bin_8x8 = np.mean(list_cc_bin_8x8)

                R1_mean = np.mean(list_R1)
                R1_std = np.std(list_R1)
                R2_mean = np.mean(list_R2)
                R2_std = np.std(list_R2)

                with open(os.path.join(test_model_dir, 'Analysis.txt'), 'a') as analysis:
                    analysis.write(str(current_step) + ', ' + str(cc_TUMF[0][1]) + ', ' + str(cc_1x1) + ', ' +
                                   str(cc_bin_2x2) + ', ' + str(cc_bin_4x4) + ', ' + str(cc_bin_8x8) + ', ' +
                                   str(R1_mean) + ', ' + str(R1_std) + ', ' + str(R2_mean) + ', ' + str(R2_std) + '\n')
                    analysis.close()

                for p in G.parameters():
                    p.requires_grad_(True)

            if opt.debug:
                break

        if epoch > opt.epoch_decay and opt.HD:
            lr = update_lr(lr, init_lr, opt.n_epochs - opt.epoch_decay, D_optim, G_optim)

    print("Total time taken: ", datetime.datetime.now() - start_time)
