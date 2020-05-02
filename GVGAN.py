import torch.nn as nn
import torch.optim as optim
import warnings
import os

warnings.filterwarnings("ignore")

from models import *
from options import Options


def get_options():
    opt = Options().initialize()
    return opt


opt = get_options()
z_out_size = opt.z_size + opt.av_size

# TO MAKE EVERYTHING EASY, NO CLASS GVGAN() HERE...

G = Generator(
    av_size=opt.av_size,
    d_size=opt.d_size,
    gc_size=opt.gc_size,
    z_size=opt.z_size,
    z_out_size=z_out_size,
    rep_size=opt.rep_size
).cuda()

D = Discriminator(
    av_size=opt.av_size,
    d_size=opt.d_size,
    gc_size=opt.gc_size,
    rep_size=opt.rep_size
).cuda()

criterion_bce = nn.BCELoss()
criterion_bce.cuda()

# This three are for A A' loss
loss_MSE = nn.MSELoss()
loss_MSE.cuda()

loss_BCE_logits = nn.BCEWithLogitsLoss()  # size_average=False)
loss_BCE_logits.cuda()

loss_BCE = nn.BCELoss()  # size_average=False)
loss_BCE.cuda()

opt_enc = optim.Adam(G.encoder.parameters(), lr=opt.lr)
opt_dec = optim.Adam(G.decoder.parameters(), lr=opt.lr)
opt_dis = optim.Adam(D.parameters(), lr=opt.lr * opt.alpha)


def save_models(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    name_G = os.path.join(save_path, 'model_G.ckpt')
    name_D = os.path.join(save_path, 'model_D.ckpt')
    torch.save(G.state_dict(), name_G)
    torch.save(D.state_dict(), name_D)


