import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from model import Generator, Discriminator_ACGAN
from dataloader import iclevrDataset, get_test_label
from evaluator import Evaluation_model
from Parser import parse_args


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(args, dataloader, generator, discriminator, device):
    adversial_Criterion = nn.BCELoss()
    classify_Criterion = nn.BCEWithLogitsLoss() if args.use_wgan else nn.BCELoss()
    if args.use_wgan:
        optimizer_g = torch.optim.RMSprop(generator.parameters(), args.lr)
        optimizer_d = torch.optim.RMSprop(discriminator.parameters() ,args.lr)
    else:
        optimizer_g = torch.optim.Adam(generator.parameters(), args.lr,betas=(args.beta1,0.999))
        optimizer_d = torch.optim.Adam(discriminator.parameters() ,args.lr,betas=(args.beta1,0.999))
    evaluation_model = Evaluation_model()

    test_label = get_test_label(args.test_file).to(device)
    test_latent =  torch.randn(len(test_label), args.latent_size).to(device)
    best_score = 0

    for epoch in range(1, 1 + args.epoch):
        total_loss_g = 0
        total_loss_d = 0
        for i, (images, conditions) in enumerate(dataloader):
            generator.train()
            discriminator.train()
            images = images.to(device)
            conditions = conditions.to(device)

            real = torch.ones(args.batch_size).to(device)
            fake = torch.zeros(args.batch_size).to(device)

            ### Train discriminator
            optimizer_d.zero_grad()

            z = torch.randn(args.batch_size, args.latent_size).to(device)
            gen_imgs = generator(z, conditions)

            adversial, classify = discriminator(images)
            if args.use_wgan:
                loss_real = -torch.mean(adversial) + args.lambda_cls * classify_Criterion(classify, conditions)
            else:
                loss_real = adversial_Criterion(adversial, real) + args.lambda_cls * classify_Criterion(classify, conditions)

            z = torch.randn(args.batch_size, args.latent_size).to(device)
            gen_imgs = generator(z, conditions)
            adversial, classify = discriminator(gen_imgs.detach())
            if args.use_wgan:
                loss_fake = torch.mean(adversial)# + classify_Criterion(classify, conditions)
            else:
                loss_fake = adversial_Criterion(adversial, fake)

            loss_d = loss_real + loss_fake
            loss_d.backward()
            optimizer_d.step()

            if args.use_wgan:
                for parm in discriminator.parameters():
                    parm.data.clamp_(-args.clamp_num,args.clamp_num)


            ### Train generator
            for _ in range(4):
                optimizer_g.zero_grad()

                z = torch.randn(args.batch_size, args.latent_size).to(device)
                gen_imgs = generator(z, conditions)
                adversial, classify = discriminator(gen_imgs.detach())
                if args.use_wgan:
                    loss_g = -torch.mean(adversial) + args.lambda_cls * classify_Criterion(classify, conditions)
                else:
                    loss_g = adversial_Criterion(adversial, real) + args.lambda_cls * classify_Criterion(classify, conditions)

                loss_g.backward()
                optimizer_g.step()
                
            print(f'Current epoch: {epoch}/{args.epoch}, step: {i+1}/{len(dataloader)}  loss_g: {loss_g.item():.3f}  loss_d: {loss_d.item():.3f}', end='\r')
            total_loss_g += loss_g.item()
            total_loss_d += loss_d.item()

        generator.eval()
        discriminator.eval()

        with torch.no_grad():
            gen_imgs = generator(test_latent, test_label)
            score = evaluation_model.eval(gen_imgs, test_label)
        if score >= best_score:
            best_score =  score
        print()
        print(f'avg loss_g: {total_loss_g/len(dataloader):.3f}  avg_loss_d: {total_loss_d/len(dataloader):.3f}')
        print(f'testing score: {score:.2f}')
        print('---------------------------------------------')

        if epoch % 10 == 0:
            torch.save(generator.state_dict(), args.log_root + "gen-e" + str(epoch) + ".pth")
            torch.save(discriminator.state_dict(), args.log_root + "dis-e" + str(epoch) + ".pth")


def test(args, generator, device):
    evaluation_model = Evaluation_model()
    generator.eval()
    test_label = get_test_label(args.test_file).to(device)
    test_latent =  torch.randn(len(test_label), args.latent_size).to(device)
    with torch.no_grad():
        gen_imgs = generator(test_latent, test_label)
        score = evaluation_model.eval(gen_imgs, test_label)
    print('---------------------------------------------')
    print(f'testing score: {score:.2f}')
    print('---------------------------------------------')
    save_image(gen_imgs+0.5, "a.jpg", padding = 2)




def main():
    args = parse_args()
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    dataset = iclevrDataset("./iclevr")
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    generator = Generator(args.latent_size).to(device)
    discriminator = Discriminator_ACGAN(is_wgan=args.use_wgan).to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    if not args.test_only:
        train(args, data_loader, generator, discriminator, device)
    test(args, generator, device)

if __name__ == "__main__":
    main()