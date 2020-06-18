import matplotlib.pyplot as plt
import os
from torch.autograd import Variable
import torch
import numpy as np
import itertools
from torchvision.utils import save_image
from torch import autograd


def train_cgan(generator, discriminator, data_loader, n_epochs, lr, b1, b2, latent_dim, n_classes, cuda,
               fist_train=False):
    path = "GANParameters/CGAN"
    os.makedirs(path, exist_ok=True)

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    loss = torch.nn.MSELoss()

    if not fist_train:
        generator.load_state_dict(torch.load(path + "/generator.pt"))
        discriminator.load_state_dict(torch.load(path + "/discriminator.pt"))

    if cuda:
        generator.cuda()
        discriminator.cuda()
        loss.cuda()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    for epoch in range(n_epochs):
        for i, (imgs, labels) in enumerate(data_loader):
            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
            gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))

            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            validity_real = discriminator(real_imgs, labels)
            d_real_loss = loss(validity_real, valid)

            # Loss for fake images
            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            if i == 1:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, n_epochs, i, len(data_loader), d_loss.item(), g_loss.item())
                )
                if epoch % 200 == 0:
                    if cuda:
                        gen_imgs = gen_imgs.cpu()
                        gen_labels = gen_labels.cpu()
                    plt.contourf(gen_imgs[0][0].detach().numpy())
                    plt.axis('off')
                    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
                    plt.savefig('caches/label' + str(gen_labels[0].detach().numpy()) + '_' + str(epoch) + '.jpg')
                    plt.close()
                    if epoch % 200 == 0:
                        torch.save(generator.state_dict(), path + "/generator.pt")
                        torch.save(discriminator.state_dict(), path + "/discriminator.pt")


def train_cdcgan(generator, discriminator, data_loader, n_epochs, lr, b1, b2, latent_dim, n_classes, cuda,
                 fist_train=False):
    path = "GANParameters/CDCGAN"
    os.makedirs(path, exist_ok=True)

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    loss = torch.nn.BCELoss()

    if not fist_train:
        generator.load_state_dict(torch.load(path + "/generator.pt"))
        discriminator.load_state_dict(torch.load(path + "/discriminator.pt"))

    if cuda:
        generator.cuda()
        discriminator.cuda()
        loss.cuda()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    for epoch in range(n_epochs):
        for i, (imgs, labels) in enumerate(data_loader):
            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
            gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))

            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            validity_real = discriminator(real_imgs, labels)
            d_real_loss = loss(validity_real, valid)

            # Loss for fake images
            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            if i == 1:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, n_epochs, i, len(data_loader), d_loss.item(), g_loss.item())
                )
                if epoch % 50 == 0:
                    if cuda:
                        gen_imgs = gen_imgs.cpu()
                        gen_labels = gen_labels.cpu()
                    plt.contourf(gen_imgs[0][0].detach().numpy())
                    plt.axis('off')
                    plt.savefig('caches/label' + str(gen_labels[0].detach().numpy()) + '_' + str(epoch) + '.jpg')
                    plt.close()
                    if epoch % 200 == 0:
                        torch.save(generator.state_dict(), path + "/generator.pt")
                        torch.save(discriminator.state_dict(), path + "/discriminator.pt")


def train_wcgan(generator, discriminator, data_loader, n_epochs, lr, b1, b2, latent_dim, n_classes, cuda,
                fist_train=False):
    path = "GANParameters/WCGAN"
    os.makedirs(path, exist_ok=True)

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    if not fist_train:
        generator.load_state_dict(torch.load(path + "/generator.pt"))
        discriminator.load_state_dict(torch.load(path + "/discriminator.pt"))

    if cuda:
        generator.cuda()
        discriminator.cuda()

    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lr)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=lr)

    for epoch in range(n_epochs):
        for i, (imgs, labels) in enumerate(data_loader):
            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))

            optimizer_D.zero_grad()

            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
            gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))

            fake_imgs = generator(z, labels).detach()

            loss_D = -torch.mean(discriminator(real_imgs, labels)) + torch.mean(discriminator(fake_imgs, gen_labels))

            loss_D.backward()
            optimizer_D.step()

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)

            if i % 5 == 0:
                optimizer_G.zero_grad()

                gen_imgs = generator(z, gen_labels)

                loss_G = -torch.mean(discriminator(gen_imgs, gen_labels))

                loss_G.backward()
                optimizer_G.step()

            if i == 1:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, n_epochs, i, len(data_loader), loss_D.item(), loss_G.item())
                )
                if epoch % 1000 == 0:
                    if cuda:
                        gen_imgs = gen_imgs.cpu()
                        gen_labels = gen_labels.cpu()
                    plt.contourf(gen_imgs[0][0].detach().numpy())
                    plt.axis('off')
                    plt.savefig('caches/label' + str(gen_labels[0].detach().numpy()) + '_' + str(epoch) + '.jpg')
                    plt.close()
                    if epoch % 1000 == 0:
                        torch.save(generator.state_dict(), path + "/generator.pt")
                        torch.save(discriminator.state_dict(), path + "/discriminator.pt")


def train_dcwcgan(generator, discriminator, data_loader, n_epochs, lr, b1, b2, latent_dim, n_classes, cuda,
                  fist_train=False):
    path = "GANParameters/DCWCGAN"
    os.makedirs(path, exist_ok=True)

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    if not fist_train:
        generator.load_state_dict(torch.load(path + "/generator.pt"))
        discriminator.load_state_dict(torch.load(path + "/discriminator.pt"))

    if cuda:
        generator.cuda()
        discriminator.cuda()

    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lr)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=lr)

    for epoch in range(n_epochs):
        for i, (imgs, labels) in enumerate(data_loader):
            batch_size = imgs.shape[0]

            # Adversarial ground truths
            # valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            # fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            imgs = imgs[:, np.newaxis, :, :]
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))

            optimizer_D.zero_grad()

            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
            gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))

            fake_imgs = generator(z, labels).detach()

            loss_D = -torch.mean(discriminator(real_imgs, labels)) + torch.mean(discriminator(fake_imgs, gen_labels))

            loss_D.backward()
            optimizer_D.step()

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)

            # gen_imgs = generator(z, gen_labels)
            # loss_G = -torch.mean(discriminator(gen_imgs, gen_labels))
            # loss_G.backward()
            if i % 1 == 0:
                gen_imgs = generator(z, gen_labels)

                loss_G = -torch.mean(discriminator(gen_imgs, gen_labels))

                loss_G.backward()
                optimizer_G.step()
                optimizer_G.zero_grad()

            if i == 1:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, n_epochs, i, len(data_loader), loss_D.item(), loss_G.item())
                )
                if epoch % 50 == 0:
                    if cuda:
                        gen_imgs = gen_imgs.cpu()
                        gen_labels = gen_labels.cpu()
                    plt.contourf(gen_imgs[0][0].detach().numpy())
                    plt.axis('off')
                    plt.savefig('caches/label' + str(gen_labels[0].detach().numpy()) + '_' + str(epoch) + '.jpg')
                    plt.close()
                    if epoch != 0:
                        torch.save(generator.state_dict(), path + "/generator.pt")
                        torch.save(discriminator.state_dict(), path + "/discriminator.pt")


def train_ponodcwcgan(generator, discriminator, data_loader, n_epochs, lr, b1, b2, latent_dim, n_classes, cuda,
                      fist_train=False):
    path = "GANParameters/PONODCWCGAN"
    os.makedirs(path, exist_ok=True)

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    if not fist_train:
        generator.load_state_dict(torch.load(path + "/generator.pt"))
        discriminator.load_state_dict(torch.load(path + "/discriminator.pt"))

    if cuda:
        generator.cuda()
        discriminator.cuda()

    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lr)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=lr)

    for epoch in range(n_epochs):
        for i, (imgs, labels) in enumerate(data_loader):
            batch_size = imgs.shape[0]

            # Adversarial ground truths
            # valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            # fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            imgs = imgs[:, np.newaxis, :, :]
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))

            optimizer_D.zero_grad()

            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
            gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))

            fake_imgs = generator(z, labels).detach()

            loss_D = -torch.mean(discriminator(real_imgs, labels)) + torch.mean(discriminator(fake_imgs, gen_labels))

            loss_D.backward()
            optimizer_D.step()

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)

            # gen_imgs = generator(z, gen_labels)
            # loss_G = -torch.mean(discriminator(gen_imgs, gen_labels))
            # loss_G.backward()
            if i % 5 == 0:
                gen_imgs = generator(z, gen_labels)

                loss_G = -torch.mean(discriminator(gen_imgs, gen_labels))

                loss_G.backward()
                optimizer_G.step()
                optimizer_G.zero_grad()

            if i == 1:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, n_epochs, i, len(data_loader), loss_D.item(), loss_G.item())
                )
                if epoch % 50 == 0:
                    if cuda:
                        gen_imgs = gen_imgs.cpu()
                        gen_labels = gen_labels.cpu()
                    plt.contourf(gen_imgs[0][0].detach().numpy())
                    plt.axis('off')
                    plt.savefig('caches/label' + str(gen_labels[0].detach().numpy()) + '_' + str(epoch) + '.jpg')
                    plt.close()
                    if epoch % 50 == 0:
                        torch.save(generator.state_dict(), path + "/generator.pt")
                        torch.save(discriminator.state_dict(), path + "/discriminator.pt")


def train_self_noise_gan(generator, discriminator, data_loader, n_epochs, lr, b1, b2, cuda, first_train=False):
    path = "GANParameters/SELFNOISEGAN"
    os.makedirs(path, exist_ok=True)

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    if not first_train:
        generator.load_state_dict(torch.load(path + "/generator.pt"))
        discriminator.load_state_dict(torch.load(path + "/discriminator.pt"))

    if cuda:
        generator.cuda()
        discriminator.cuda()

    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lr)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=lr)

    for epoch in range(n_epochs):
        for i, (imgs, labels) in enumerate(data_loader):
            batch_size = imgs.shape[0]

            # Configure input
            imgs = imgs[:, np.newaxis, :, :]
            real_imgs = Variable(imgs.type(FloatTensor))  # cuda()

            optimizer_D.zero_grad()

            fake_imgs = generator(real_imgs).detach()  # self noise

            loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

            loss_D.backward()
            optimizer_D.step()

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)

            # gen_imgs = generator(z, gen_labels)
            # loss_G = -torch.mean(discriminator(gen_imgs, gen_labels))
            # loss_G.backward()
            if i % 4 == 0:
                gen_imgs = generator(real_imgs)

                loss_G = -torch.mean(discriminator(gen_imgs))

                loss_G.backward()
                optimizer_G.step()
                optimizer_G.zero_grad()

            if i == 1:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, n_epochs, i, len(data_loader), loss_D.item(), loss_G.item())
                )
                if epoch % 100 == 0:
                    if cuda:
                        gen_imgs = gen_imgs.cpu()
                    plt.contourf(gen_imgs[0][0].detach().numpy())
                    plt.axis('off')
                    plt.savefig('caches/label' + str(labels[0].detach().numpy()) + '_' + str(epoch) + '.jpg')
                    plt.close()
                    if epoch != 0:
                        torch.save(generator.state_dict(), path + "/generator.pt")
                        torch.save(discriminator.state_dict(), path + "/discriminator.pt")


def train_info_gan(generator, discriminator, data_loader, n_epochs, lr, b1, b2, latent_dim, n_classes, code_dim, cuda,
                   first_train=False):
    path = "GANParameters/INFOGAN"
    os.makedirs(path, exist_ok=True)
    os.makedirs('caches/static', exist_ok=True)
    os.makedirs('caches/varying_c1', exist_ok=True)
    os.makedirs('caches/varying_c2', exist_ok=True)

    adversarial_loss = torch.nn.MSELoss()
    categorical_loss = torch.nn.CrossEntropyLoss()
    continuous_loss = torch.nn.MSELoss()

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    if not first_train:
        generator.load_state_dict(torch.load(path + "/generator.pt"))
        discriminator.load_state_dict(torch.load(path + "/discriminator.pt"))

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        categorical_loss.cuda()
        continuous_loss.cuda()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_info = torch.optim.Adam(
        itertools.chain(generator.parameters(), discriminator.parameters()), lr=lr, betas=(b1, b2)
    )

    static_z = Variable(FloatTensor(np.zeros((n_classes ** 2, latent_dim))))
    static_label = to_categorical(
        np.array([num for _ in range(n_classes) for num in range(n_classes)]), num_columns=n_classes
    )
    static_code = Variable(FloatTensor(np.zeros((n_classes ** 2, code_dim))))

    # Loss weights of info
    lambda_cat = 1
    lambda_con = 0.1

    for epoch in range(n_epochs):
        for i, (imgs, labels) in enumerate(data_loader):
            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            imgs = imgs[:, np.newaxis, :, :]
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = to_categorical(labels.numpy(), num_columns=n_classes)

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
            label_input = to_categorical(np.random.randint(0, n_classes, batch_size), num_columns=n_classes)
            code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, code_dim))))

            # Generate a batch of images
            gen_imgs = generator(z, label_input, code_input)

            # Loss measures generator's ability to fool the discriminator
            validity, _, _ = discriminator(gen_imgs)
            g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            real_pred, _, _ = discriminator(real_imgs)
            d_real_loss = adversarial_loss(real_pred, valid)

            # Loss for fake images
            fake_pred, _, _ = discriminator(gen_imgs.detach())
            d_fake_loss = adversarial_loss(fake_pred, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # ------------------
            # Information Loss
            # ------------------

            optimizer_info.zero_grad()

            # Sample labels
            sampled_labels = np.random.randint(0, n_classes, batch_size)

            # Ground truth labels
            gt_labels = Variable(LongTensor(sampled_labels), requires_grad=False)

            # Sample noise, labels and code as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
            label_input = to_categorical(sampled_labels, num_columns=n_classes)
            code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, code_dim))))

            gen_imgs = generator(z, label_input, code_input)
            _, pred_label, pred_code = discriminator(gen_imgs)

            info_loss = lambda_cat * categorical_loss(pred_label, gt_labels) + lambda_con * continuous_loss(
                pred_code, code_input
            )

            info_loss.backward()
            optimizer_info.step()

            if i == 0:
                print(
                    "[Epoch %d/%d] [D loss: %f] [G loss: %f] [info loss: %f]"
                    % (epoch, n_epochs, d_loss.item(), g_loss.item(), info_loss.item())
                )

        if epoch % 100 == 0:
            """Saves a grid of generated digits ranging from 0 to n_classes"""
            # Static sample
            z = Variable(FloatTensor(np.random.normal(0, 1, (n_classes ** 2, latent_dim))))
            static_sample = generator(z, static_label, static_code)
            save_image(static_sample.data, "caches/static/%d.jpg" % epoch, nrow=n_classes, normalize=True)

            # Get varied c1 and c2
            zeros = np.zeros((n_classes ** 2, 1))
            c_varied = np.repeat(np.linspace(-1, 1, n_classes)[:, np.newaxis], n_classes, 0)
            c1 = Variable(FloatTensor(np.concatenate((c_varied, zeros), -1)))
            c2 = Variable(FloatTensor(np.concatenate((zeros, c_varied), -1)))
            sample1 = generator(static_z, static_label, c1)
            sample2 = generator(static_z, static_label, c2)
            save_image(sample1.data, "caches/varying_c1/%d.jpg" % epoch, nrow=n_classes, normalize=True)
            save_image(sample2.data, "caches/varying_c2/%d.jpg" % epoch, nrow=n_classes, normalize=True)

            if epoch % 200 == 0 and epoch != 0:
                torch.save(generator.state_dict(), path + "/generator.pt")
                torch.save(discriminator.state_dict(), path + "/discriminator.pt")


def train_conv1d_gan(generator, discriminator, data_loader, n_epochs, lr, b1, b2, latent_dim, n_classes, cuda,
                     first_train=False):
    path = "GANParameters/CONV1DGAN"
    os.makedirs(path, exist_ok=True)

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    if not first_train:
        generator.load_state_dict(torch.load(path + "/generator.pt"))
        discriminator.load_state_dict(torch.load(path + "/discriminator.pt"))

    if cuda:
        generator.cuda()
        discriminator.cuda()

    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lr)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=lr)

    for epoch in range(n_epochs):
        for i, (datas, labels) in enumerate(data_loader):
            batch_size = datas.shape[0]

            # Adversarial ground truths
            # valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            # fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            datas = datas[:, np.newaxis, :]
            real_datas = Variable(datas.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))

            optimizer_D.zero_grad()

            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, 1, latent_dim))))
            gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))

            fake_datas = generator(z, labels).detach()

            loss_D = -torch.mean(discriminator(real_datas, labels)) + torch.mean(discriminator(fake_datas, gen_labels))

            loss_D.backward()
            optimizer_D.step()

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)

            # gen_imgs = generator(z, gen_labels)
            # loss_G = -torch.mean(discriminator(gen_imgs, gen_labels))
            # loss_G.backward()
            if i % 2 == 0:
                gen_datas = generator(z, gen_labels)

                loss_G = -torch.mean(discriminator(gen_datas, gen_labels))

                loss_G.backward()
                optimizer_G.step()
                optimizer_G.zero_grad()

            if i == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, n_epochs, i, len(data_loader), loss_D.item(), loss_G.item())
                )
                if epoch % 200 == 0:
                    if cuda:
                        gen_datas = gen_datas.cpu()
                        gen_labels = gen_labels.cpu()
                    plt.plot(gen_datas[0][0].detach().numpy())
                    # plt.axis('off')
                    plt.savefig('caches/label' + str(gen_labels[0].detach().numpy()) + '_' + str(epoch) + '.jpg')
                    plt.close()
                    if epoch != 0:
                        torch.save(generator.state_dict(), path + "/generator.pt")
                        torch.save(discriminator.state_dict(), path + "/discriminator.pt")


def train_linear_1d_gan(generator, discriminator, data_loader, n_epochs, lr, b1, b2, latent_dim, n_classes, cuda,
                        first_train=False):
    path = "GANParameters/LINEAR1DGAN"
    os.makedirs(path, exist_ok=True)

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    if not first_train:
        generator.load_state_dict(torch.load(path + "/generator.pt"))
        discriminator.load_state_dict(torch.load(path + "/discriminator.pt"))

    if cuda:
        generator.cuda()
        discriminator.cuda()

    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lr)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=lr)

    for epoch in range(n_epochs):
        for i, (datas, labels) in enumerate(data_loader):
            batch_size = datas.shape[0]

            # Adversarial ground truths
            # valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            # fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            # datas = datas[:, np.newaxis, :]
            real_datas = Variable(datas.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))

            optimizer_D.zero_grad()

            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
            gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))

            fake_datas = generator(z, labels).detach()

            loss_D = -torch.mean(discriminator(real_datas, labels)) + torch.mean(discriminator(fake_datas, gen_labels))

            loss_D.backward()
            optimizer_D.step()

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)

            # gen_imgs = generator(z, gen_labels)
            # loss_G = -torch.mean(discriminator(gen_imgs, gen_labels))
            # loss_G.backward()
            if i % 2 == 0:
                gen_datas = generator(z, gen_labels)

                loss_G = -torch.mean(discriminator(gen_datas, gen_labels))

                loss_G.backward()
                optimizer_G.step()
                optimizer_G.zero_grad()

            if i == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, n_epochs, i, len(data_loader), loss_D.item(), loss_G.item())
                )
                if epoch % 500 == 0:
                    if cuda:
                        gen_datas = gen_datas.cpu()
                        gen_labels = gen_labels.cpu()
                    plt.plot(gen_datas[0].detach().numpy())
                    # plt.axis('off')
                    plt.savefig('caches/label' + str(gen_labels[0].detach().numpy()) + '_' + str(epoch) + '.jpg')
                    plt.close()
                    if epoch != 0:
                        torch.save(generator.state_dict(), path + "/generator.pt")
                        torch.save(discriminator.state_dict(), path + "/discriminator.pt")


def train_selfnoise_1d_gan(generator, discriminator, data_loader, n_epochs, lr, b1, b2, latent_dim, n_classes, cuda,
                           first_train=False):
    path = "GANParameters/SELFNOISE1DGAN"
    os.makedirs(path, exist_ok=True)
    wgan = True
    print('wgan:', wgan)

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    if not first_train:
        generator.load_state_dict(torch.load(path + "/generator.pt"))
        discriminator.load_state_dict(torch.load(path + "/discriminator.pt"))

    loss = torch.nn.BCELoss()
    if cuda:
        generator.cuda()
        discriminator.cuda()
        loss.cuda()

    if not wgan:
        optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lr)
        optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=lr)
    else:
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
        lambda_gp = 10

    for epoch in range(n_epochs):
        for i, (datas, labels) in enumerate(data_loader):
            batch_size = datas.shape[0]

            # Adversarial ground truths
            if not wgan:
                valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
                fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            datas = datas[:, np.newaxis, :]
            real_datas = Variable(datas.type(FloatTensor))

            optimizer_D.zero_grad()

            fake_datas = generator(real_datas).detach()  # self noise

            if wgan:
                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(discriminator, real_datas.data, fake_datas.data, cuda)
                loss_D = -torch.mean(discriminator(real_datas)) + torch.mean(
                    discriminator(fake_datas)) + lambda_gp * gradient_penalty
            else:
                validity_real = discriminator(real_datas)
                d_real_loss = loss(validity_real, valid)

                # Loss for fake images
                validity_fake = discriminator(fake_datas.detach())
                d_fake_loss = loss(validity_fake, fake)

                # Total discriminator loss
                loss_D = (d_real_loss + d_fake_loss) / 2

            loss_D.backward()
            optimizer_D.step()

            if wgan:
                if i % 3 == 0:
                    gen_datas = generator(real_datas).detach()

                    loss_G = -torch.mean(discriminator(gen_datas))

                    loss_G.backward()
                    optimizer_G.step()
                    optimizer_G.zero_grad()
                    # Clip weights of discriminator
                    # for p in discriminator.parameters():
                    #     p.data.clamp_(-0.01, 0.01)
            else:
                gen_datas = generator(real_datas)

                # Loss measures generator's ability to fool the discriminator
                validity = discriminator(gen_datas)
                loss_G = loss(validity, valid)

            if i == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, n_epochs, i, len(data_loader), loss_D.item(), loss_G.item())
                )
                if epoch % 50 == 0:
                    if cuda:
                        gen_datas = gen_datas.cpu()
                    plt.plot(gen_datas[0][0].detach().numpy())
                    # plt.axis('off')
                    plt.savefig('caches/label' + str(labels[0].detach().numpy()) + '_' + str(epoch) + '.jpg')
                    plt.close()
                    if epoch != 0:
                        torch.save(generator.state_dict(), path + "/generator.pt")
                        torch.save(discriminator.state_dict(), path + "/discriminator.pt")


''' iner function
......................................................................................................................
'''


def to_categorical(y, num_columns, cuda=True):
    """Returns one-hot encoded Variable"""
    y_cat = np.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.0

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    return Variable(FloatTensor(y_cat))


def compute_gradient_penalty(D, real_samples, fake_samples, cuda):
    """Calculates the gradient penalty loss for WGAN GP"""
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
