import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf
import pytorch_lightning as pl

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=3000, help="epochs for training")
parser.add_argument("--batch_size", type=int, default=1000, help="size of the batches")
parser.add_argument("--lr_D", type=float, default=1e-7, help="learning rate for Discriminator")
parser.add_argument("--lr_G", type=float, default=1e-6, help="learning rate for Generator")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=1000, help="dimensionality of the latent space")
parser.add_argument("--n_rows", type=int, default=5, help="number of rows")
parser.add_argument("--n_cols", type=int, default=100, help="number of columns")
parser.add_argument("--tickers", type=list, default=['AGG', 'VMBS', 'VCIT', 'LQD', 'EMB', 'BNDX', 'JNK', 'VCLT', 'MUB', 'HYG', 'TLT', 'TIP', 'MBB', 'IEF', 'IEI', 'GOVT'], help="tickers")
parser.add_argument("--noise_name", type=str, default='t5', help="noise name")
parser.add_argument("--numNN", type=int, default=5, help="number of NNs")

opt = parser.parse_args()
print(opt)

R_shape = (opt.n_rows, opt.n_cols)

# Use current directory for saving data and models
current_dir = os.getcwd()
gen_data_path = os.path.join(current_dir, "generated_data")
model_path = os.path.join(current_dir, "models")
os.makedirs(gen_data_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(R_shape))),
        )

    def forward(self, z):
        img = self.model(z)
        img = torch.clamp(img, min=-1, max=1)
        img = img.view(img.shape[0], *R_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.n_rows * opt.n_cols, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity

class GAN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.criterion = nn.BCELoss()
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()

        # Unpack the batch
        real_R, = batch

        # Debugging print statements
        print(f"Shape of real_R: {real_R.shape}")
        
        # Sample noise
        z = torch.randn(real_R.shape[0], opt.latent_dim, device=self.device)

        # Generate a batch of data
        gen_R = self.generator(z)

        # Train Discriminator
        opt_d.zero_grad()
        real_validity = self.discriminator(real_R)
        fake_validity = self.discriminator(gen_R.detach())
        real_loss = self.criterion(real_validity, torch.ones_like(real_validity))
        fake_loss = self.criterion(fake_validity, torch.zeros_like(fake_validity))
        d_loss = (real_loss + fake_loss) / 2
        self.manual_backward(d_loss)
        opt_d.step()

        # Train Generator
        opt_g.zero_grad()
        gen_validity = self.discriminator(gen_R)
        g_loss = self.criterion(gen_validity, torch.ones_like(gen_validity))
        self.manual_backward(g_loss)
        opt_g.step()

        # Log losses
        self.log('g_loss', g_loss, prog_bar=True)
        self.log('d_loss', d_loss, prog_bar=True)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=opt.lr_G, betas=(opt.b1, opt.b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=opt.lr_D, betas=(opt.b1, opt.b2))
        return [opt_g, opt_d]

def Train(opt):
    start_date = "2010-01-01"
    end_date = "2023-07-08"

    # Load data
    returns = load_data(opt.tickers, start_date, end_date)
    
    tensor_data = torch.FloatTensor(returns)
    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

    for iii in range(opt.numNN):
        seed = np.random.randint(low=1, high=10000)
        print(f"------ Model {iii} Starts with Random Seed {seed} ------")
        
        pl.seed_everything(seed)
        
        model = GAN()
        
        trainer = pl.Trainer(
            max_epochs=opt.n_epochs, 
            accelerator='auto',
            devices=1,
            callbacks=[pl.callbacks.ModelCheckpoint(
                dirpath=model_path, 
                filename=f"model_{iii}"+"_{epoch:02d}_{g_loss:.2f}",
                save_top_k=3,
                monitor="g_loss"
            )],
            log_every_n_steps=1
        )
        
        trainer.fit(model, dataloader)

        # Generate samples after training
        gen_size = 1000
        z = torch.randn(gen_size, opt.latent_dim, device=model.device)

        model.generator.eval()
        with torch.no_grad():
            gen_R = model.generator(z)

        np.save(os.path.join(gen_data_path, f"Fake_id{iii}_final.npy"), gen_R.cpu().numpy())

def load_data(symbols, start_date, end_date):
    data = yf.download(symbols, start=start_date, end=end_date)['Adj Close']
    returns = data.pct_change().dropna()

    # Flatten the data to a 2D array where each row is a flattened return series
    num_elements = opt.n_rows * opt.n_cols
    flattened_returns = returns.values.flatten()
    total_samples = len(flattened_returns) // num_elements
    reshaped_returns = flattened_returns[:total_samples * num_elements].reshape(total_samples, opt.n_rows, opt.n_cols)

    return reshaped_returns

if __name__ == "__main__":
    Train(opt)
