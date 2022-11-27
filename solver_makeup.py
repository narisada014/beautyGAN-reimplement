import torch
import torch.nn.init as init
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
from torch.utils.tensorboard import SummaryWriter

import os
import time
import datetime

import net
from ops.histogram_matching import *
from ops.loss_added import GANLoss
from tqdm import tqdm as tqdm

import torchvision.models as models
from utils import to_var, de_norm, get_mask
import cv2
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms
import glcm

class Solver_makeupGAN(object):
    def __init__(self, data_loaders, config, dataset_config):
        # dataloader
        self.checkpoint = config.checkpoint
        # Hyper-parameteres
        self.g_lr = config.G_LR
        self.d_lr = config.D_LR
        self.ndis = config.ndis
        self.num_epochs = config.num_epochs  # set 200
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size
        self.whichG = config.whichG
        self.norm = config.norm

        # Training settings
        self.snapshot_step = config.snapshot_step
        self.log_step = config.log_step
        self.vis_step = config.vis_step

        #training setting
        self.task_name = config.task_name

        # Data loader
        self.data_loader_train = data_loaders[0]
        self.data_loader_test = data_loaders[1]

        # Model hyper-parameters
        self.img_size = config.img_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lips = config.lips
        self.skin = config.skin
        self.eye = config.eye

        # Hyper-parameteres
        self.lambda_idt = config.lambda_idt
        self.lambda_A = config.lambda_A
        self.lambda_B = config.lambda_B
        self.lambda_his_lip = config.lambda_his_lip
        self.lambda_his_skin_1 = config.lambda_his_skin_1
        self.lambda_his_skin_2 = config.lambda_his_skin_2
        self.lambda_his_eye = config.lambda_his_eye
        self.lambda_vgg = config.lambda_vgg

        self.beta1 = config.beta1
        self.beta2 = config.beta2

        self.cls = config.cls_list
        self.content_layer = config.content_layer
        self.direct = config.direct
        # Test settings
        self.test_model = config.test_model

        # Path
        self.log_path = config.log_path + '_' + config.task_name
        self.vis_path = config.vis_path + '_' + config.task_name
        self.snapshot_path = config.snapshot_path + '_' + config.task_name
        self.result_path = config.vis_path + '_' + config.task_name

        # Experiment Condition
        self.is_added_lumiL1: bool = config.is_added_lumi_L1
        self.is_added_lumi_L1_to_his: bool = config.is_added_lumi_L1_to_his
        self.is_added_lumi_L2_to_his: bool = config.is_added_lumi_L2_to_his
        self.is_added_lumi_L1_lambda_5_to_his: bool = config.is_added_lumi_L1_lambda_5_to_his
        self.is_added_l_hist: bool = config.is_added_l_hist
        self.is_luminance_matching_to_fake: bool = config.is_luminance_matching_to_fake
        self.is_HS_matching: bool = config.is_HS_matching

        self.is_HLS_match_all: bool = config.is_HLS_match_all
        self.is_HS_match_skin_HLS_match_other: bool = config.is_HS_match_skin_HLS_match_other
        self.is_AB_match_all: bool = config.is_AB_match_all
        self.is_AB_match_skin_Lab_match_other: bool = config.is_AB_match_skin_Lab_match_other
        self.is_grad_loss = config.is_grad_loss
        self.is_glcm_loss = config.is_glcm_loss
        self.is_glcm_loss_with_count = config.is_glcm_loss_with_count
        self.is_glcm_to_org_loss = config.is_glcm_to_org_loss

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if not os.path.exists(self.vis_path):
            os.makedirs(self.vis_path)
        if not os.path.exists(self.snapshot_path):
            os.makedirs(self.snapshot_path)

        self.build_model()
        # Start with trained model
        if self.checkpoint:
            print('Loaded pretrained model from: ', self.checkpoint)
            self.load_checkpoint()

        writer_path = os.path.join('./', 'runs', config.task_name)
        print('TensorBoard will be saved in: ', writer_path)
        self.writer = SummaryWriter(writer_path)
        if not os.path.isdir(os.path.join('./', 'runs', config.task_name)):
            os.makedirs(os.path.join('./runs', config.task_name))
        #for recording
        self.start_time = time.time()
        self.e = 0
        self.i = 0
        self.loss = {}

        if not os.path.exists(self.snapshot_path):
            os.makedirs(self.snapshot_path)

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def update_lr(self, g_lr, d_lr):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for i in self.cls:
            for param_group in getattr(self, "d_" + i + "_optimizer").param_groups:
                param_group['lr'] = d_lr

    def log_terminal(self):
        elapsed = time.time() - self.start_time
        elapsed = str(datetime.timedelta(seconds=elapsed))

        log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
            elapsed, self.e+1, self.num_epochs, self.i+1, self.iters_per_epoch)

        for tag, value in self.loss.items():
            log += ", {}: {:.4f}".format(tag, value)
        print(log)

    def save_models(self):
        torch.save(self.G.state_dict(),
                   os.path.join(self.snapshot_path, '{}_{}_G.pth'.format(self.e + 1, self.i + 1)))
        for i in self.cls:
            torch.save(getattr(self, "D_" + i).state_dict(),
                       os.path.join(self.snapshot_path, '{}_{}_D_'.format(self.e + 1, self.i + 1) + i + '.pth'))

    def weights_init_xavier(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.xavier_normal(m.weight.data, gain=1.0)
        elif classname.find('Linear') != -1:
            init.xavier_normal(m.weight.data, gain=1.0)

    def to_var(self, x, requires_grad=True):
        if torch.cuda.is_available():
            x = x.cuda()
        if not requires_grad:
            return Variable(x, requires_grad=requires_grad)
        else:
            return Variable(x)

    def de_norm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def load_checkpoint(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.snapshot_path, '{}_G.pth'.format(self.checkpoint))))
        for i in self.cls:
            getattr(self, "D_" + i).load_state_dict(torch.load(os.path.join(
                self.snapshot_path, '{}_D_'.format(self.checkpoint) + i + '.pth')))
        print('loaded trained models (step: {})..!'.format(self.checkpoint))

    def build_model(self):
        # Define generators and discriminators
        if self.whichG=='normal':
            self.G = net.Generator_makeup(self.g_conv_dim, self.g_repeat_num)
        if self.whichG=='branch':
            self.G = net.Generator_branch(self.g_conv_dim, self.g_repeat_num)
        for i in self.cls:
            setattr(self, "D_" + i, net.Discriminator(self.img_size, self.d_conv_dim, self.d_repeat_num, self.norm))

        self.criterionL1 = torch.nn.L1Loss()
        self.criterionL2 = torch.nn.MSELoss()
        self.criterionGAN = GANLoss(use_lsgan=True, tensor =torch.cuda.FloatTensor)
        self.vgg=models.vgg16(pretrained=True)
        # Optimizers
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        for i in self.cls:
            setattr(self, "d_" + i + "_optimizer", \
                    torch.optim.Adam(filter(lambda p: p.requires_grad, getattr(self, "D_" + i).parameters()), \
                                     self.d_lr, [self.beta1, self.beta2]))

        # Weights initialization
        self.G.apply(self.weights_init_xavier)
        for i in self.cls:
            getattr(self, "D_" + i).apply(self.weights_init_xavier)

        if torch.cuda.is_available():
            self.G.cuda()
            self.vgg.cuda()
            for i in self.cls:
                getattr(self, "D_" + i).cuda()

    def vgg_forward(self, model, x):
        for i in range(18):
            x=model.features[i](x)
        return x

    def rebound_box(self, mask_A, mask_B, mask_A_face):
        index_tmp = mask_A.nonzero()
        x_A_index = index_tmp[:, 2]
        y_A_index = index_tmp[:, 3]
        index_tmp = mask_B.nonzero()
        x_B_index = index_tmp[:, 2]
        y_B_index = index_tmp[:, 3]
        mask_A_temp = mask_A.copy_(mask_A)
        mask_B_temp = mask_B.copy_(mask_B)
        mask_A_temp[: ,: ,min(x_A_index)-10:max(x_A_index)+11, min(y_A_index)-10:max(y_A_index)+11] =\
                            mask_A_face[: ,: ,min(x_A_index)-10:max(x_A_index)+11, min(y_A_index)-10:max(y_A_index)+11]
        mask_B_temp[: ,: ,min(x_B_index)-10:max(x_B_index)+11, min(y_B_index)-10:max(y_B_index)+11] =\
                            mask_A_face[: ,: ,min(x_B_index)-10:max(x_B_index)+11, min(y_B_index)-10:max(y_B_index)+11]
        mask_A_temp = self.to_var(mask_A_temp, requires_grad=False)
        mask_B_temp = self.to_var(mask_B_temp, requires_grad=False)
        return mask_A_temp, mask_B_temp

    def mask_preprocess(self, mask_A, mask_B):
        index_tmp = mask_A.nonzero()
        x_A_index = index_tmp[:, 2] # 画像のX座標
        y_A_index = index_tmp[:, 3] # 画像のY座標
        index_tmp = mask_B.nonzero()
        x_B_index = index_tmp[:, 2]
        y_B_index = index_tmp[:, 3]
        mask_A = self.to_var(mask_A, requires_grad=False)
        mask_B = self.to_var(mask_B, requires_grad=False)
        index = [x_A_index, y_A_index, x_B_index, y_B_index]
        index_2 = [x_B_index, y_B_index, x_A_index, y_A_index]
        return mask_A, mask_B, index, index_2
    
    def criterion_make_up_his_with_lumi_his(self, original_data, fake_data, target_data, mask_src, mask_tar, index):
        original_data = (self.de_norm(original_data) * 255).squeeze()
        fake_data = (self.de_norm(fake_data) * 255).squeeze()
        target_data = (self.de_norm(target_data) * 255).squeeze()
        mask_src = mask_src.expand(1, 3, mask_src.size(2), mask_src.size(2)).squeeze()
        mask_tar = mask_tar.expand(1, 3, mask_tar.size(2), mask_tar.size(2)).squeeze()
        original_data_masked = original_data * mask_src
        fake_data_masked = fake_data * mask_src
        target_masked = target_data * mask_tar

        # original画像のHSL空間のL値を抽出する
        original_data_masked_image = np.array(to_pil_image(original_data_masked), dtype=np.uint8)
        original_data_masked_hls_image = cv2.cvtColor(original_data_masked_image, cv2.COLOR_RGB2HLS)
        original_data_masked_L_channel = original_data_masked_hls_image[:,:,1]

        # histogram_matchingをfake -> refで行う
        fake_data_match = histogram_matching(fake_data_masked, target_masked, index, 3)

        # 生成画像とヒストグラムマッチングを施した生成画像の一次ノルムを誤差として計算する
        # loss = self.criterionL1(fake_data_masked, self.to_var(fake_data_match, requires_grad=False))

        # オリジナル画像とヒストグラムマッチングを施した生成画像のL値の一次ノルムを誤差として上記の誤差に追加する
        # 生成画像にヒストグラムマッチングを施した画像のHLS空間のL値を抽出する
        fake_data_match_image = np.array(to_pil_image(fake_data_match), dtype=np.uint8)
        fake_data_match_hls_image = cv2.cvtColor(fake_data_match_image, cv2.COLOR_RGB2HLS)
        fake_data_match_L_channel = fake_data_match_hls_image[:,:,1]

        # histogram_matchingをHLSのLのみで（fake -> ref）-> originalとして行う
        matched_to_input = match_histograms(fake_data_match_L_channel, original_data_masked_L_channel)

        # fakeとmatched_to_inputの誤差を計算する
        loss = self.criterionL1(
            self.to_var(torch.from_numpy(matched_to_input.astype(np.float32)).clone(), requires_grad=False),
            self.to_var(torch.from_numpy(fake_data_match_L_channel.astype(np.float32)).clone(), requires_grad=False),
        )
        return loss

    # HSL空間のL成分を誤差として加える
    def criterionLumi(self, fake_data, original_data, mask, mask_type_string, number):
        # Tensorを逆正規化して画像に戻す
        fake_data = (self.de_norm(fake_data) * 255).squeeze()
        original_data = (self.de_norm(original_data) * 255).squeeze()
        mask = mask.expand(1, 3, mask.size(2), mask.size(2)).squeeze()
        fake_data_masked = fake_data * mask
        original_data_masked = original_data * mask
        fake_masked_image = np.array(to_pil_image(fake_data_masked), dtype=np.uint8)
        org_masked_image = np.array(to_pil_image(original_data_masked), dtype=np.uint8)
        # RGB->HLSの変換を行う。Pillowでは色の順番はRGB（赤、緑、青）を前提としている
        fake_masked_hls_image = cv2.cvtColor(fake_masked_image, cv2.COLOR_RGB2HLS)
        org_masked_hls_image = cv2.cvtColor(org_masked_image, cv2.COLOR_RGB2HLS)

        # 輝度値の成分Lを抽出する
        fake_masked_L_channel = fake_masked_hls_image[:,:,1]
        org_masked_L_channel = org_masked_hls_image[:,:,1]
        # result = cv2.imwrite(f"./experiments/org_masked_L_channel_{mask_type_string}_{number}.jpg",org_masked_L_channel)
        
        # 生成された画像とオリジナル画像のL1ロスを計算する。画像がndarray形式なのでTensorに変換してPytorchが扱えるように変換する
        loss = self.criterionL1(
            self.to_var(torch.from_numpy(fake_masked_L_channel.astype(np.float32)).clone(), requires_grad=False),
            self.to_var(torch.from_numpy(org_masked_L_channel.astype(np.float32)).clone(), requires_grad=False)
        )
        return loss
    
    def criterionHisLumiL1(self, original_data, fake_data, target_data, mask_src, mask_tar, index):
        original_data = (self.de_norm(original_data) * 255).squeeze()
        fake_data = (self.de_norm(fake_data) * 255).squeeze()
        target_data = (self.de_norm(target_data) * 255).squeeze()
        mask_src = mask_src.expand(1, 3, mask_src.size(2), mask_src.size(2)).squeeze()
        mask_tar = mask_tar.expand(1, 3, mask_tar.size(2), mask_tar.size(2)).squeeze()
        original_data_masked = original_data * mask_src
        fake_data_masked = fake_data * mask_src
        target_masked = target_data * mask_tar

        # original画像のHSL空間のL値を抽出する
        original_data_masked_image = np.array(to_pil_image(original_data_masked), dtype=np.uint8)
        original_data_masked_hls_image = cv2.cvtColor(original_data_masked_image, cv2.COLOR_RGB2HLS)
        original_data_masked_L_channel = original_data_masked_hls_image[:,:,1]

        fake_data_match = histogram_matching(fake_data_masked, target_masked, index, 3)

        # 生成画像にヒストグラムマッチングを施した画像のHLS空間のL値を抽出する
        fake_data_match_image = np.array(to_pil_image(fake_data_match), dtype=np.uint8)
        fake_data_match_hls_image = cv2.cvtColor(fake_data_match_image, cv2.COLOR_RGB2HLS)
        fake_data_match_L_channel = fake_data_match_hls_image[:,:,1]

        # 生成画像とヒストグラムマッチングを施した生成画像の一次ノルムを誤差として計算する
        loss = self.criterionL1(fake_data_masked, self.to_var(fake_data_match, requires_grad=False))

        # オリジナル画像とヒストグラムマッチングを施した生成画像のL値の一次ノルムを誤差として上記の誤差に追加する
        loss = loss + self.criterionL1(
            self.to_var(torch.from_numpy(fake_data_match_L_channel.astype(np.float32)).clone(), requires_grad=False),
            self.to_var(torch.from_numpy(original_data_masked_L_channel.astype(np.float32)).clone(), requires_grad=False)
        )
        return loss
    
    def criterionHisLumiL1Lambda5(self, original_data, fake_data, target_data, mask_src, mask_tar, index):
        original_data = (self.de_norm(original_data) * 255).squeeze()
        fake_data = (self.de_norm(fake_data) * 255).squeeze()
        target_data = (self.de_norm(target_data) * 255).squeeze()
        mask_src = mask_src.expand(1, 3, mask_src.size(2), mask_src.size(2)).squeeze()
        mask_tar = mask_tar.expand(1, 3, mask_tar.size(2), mask_tar.size(2)).squeeze()
        original_data_masked = original_data * mask_src
        fake_data_masked = fake_data * mask_src
        target_masked = target_data * mask_tar

        # original画像のHSL空間のL値を抽出する
        original_data_masked_image = np.array(to_pil_image(original_data_masked), dtype=np.uint8)
        original_data_masked_hls_image = cv2.cvtColor(original_data_masked_image, cv2.COLOR_RGB2HLS)
        original_data_masked_L_channel = original_data_masked_hls_image[:,:,1]

        fake_data_match = histogram_matching(fake_data_masked, target_masked, index, 3)

        # 生成画像にヒストグラムマッチングを施した画像のHLS空間のL値を抽出する
        fake_data_match_image = np.array(to_pil_image(fake_data_match), dtype=np.uint8)
        fake_data_match_hls_image = cv2.cvtColor(fake_data_match_image, cv2.COLOR_RGB2HLS)
        fake_data_match_L_channel = fake_data_match_hls_image[:,:,1]

        # 生成画像とヒストグラムマッチングを施した生成画像の一次ノルムを誤差として計算する
        loss = self.criterionL1(fake_data_masked, self.to_var(fake_data_match, requires_grad=False))

        # オリジナル画像とヒストグラムマッチングを施した生成画像のL値の一次ノルムを誤差として上記の誤差に追加する
        loss = loss + self.criterionL1(
            self.to_var(torch.from_numpy(fake_data_match_L_channel.astype(np.float32)).clone(), requires_grad=False),
            self.to_var(torch.from_numpy(original_data_masked_L_channel.astype(np.float32)).clone(), requires_grad=False)
        ) * 5 # lambda = 5
        return loss
    
    def criterionHisLumiL2(self, original_data, fake_data, target_data, mask_src, mask_tar, index):
        original_data = (self.de_norm(original_data) * 255).squeeze()
        fake_data = (self.de_norm(fake_data) * 255).squeeze()
        target_data = (self.de_norm(target_data) * 255).squeeze()
        mask_src = mask_src.expand(1, 3, mask_src.size(2), mask_src.size(2)).squeeze()
        mask_tar = mask_tar.expand(1, 3, mask_tar.size(2), mask_tar.size(2)).squeeze()
        original_data_masked = original_data * mask_src
        fake_data_masked = fake_data * mask_src
        target_masked = target_data * mask_tar

        # original画像のHSL空間のL値を抽出する
        original_data_masked_image = np.array(to_pil_image(original_data_masked), dtype=np.uint8)
        original_data_masked_hls_image = cv2.cvtColor(original_data_masked_image, cv2.COLOR_RGB2HLS)
        original_data_masked_L_channel = original_data_masked_hls_image[:,:,1]

        fake_data_match = histogram_matching(fake_data_masked, target_masked, index, 3)

        # 生成画像にヒストグラムマッチングを施した画像のHLS空間のL値を抽出する
        fake_data_match_image = np.array(to_pil_image(fake_data_match), dtype=np.uint8)
        fake_data_match_hls_image = cv2.cvtColor(fake_data_match_image, cv2.COLOR_RGB2HLS)
        fake_data_match_L_channel = fake_data_match_hls_image[:,:,1]

        # 生成画像とヒストグラムマッチングを施した生成画像の一次ノルムを誤差として計算する
        loss = self.criterionL1(fake_data_masked, self.to_var(fake_data_match, requires_grad=False))

        # オリジナル画像とヒストグラムマッチングを施した生成画像のL値の一次ノルムを誤差として上記の誤差に追加する
        l2_loss = self.criterionL2(
            self.to_var(torch.from_numpy(fake_data_match_L_channel.astype(np.float32)).clone(), requires_grad=False),
            self.to_var(torch.from_numpy(original_data_masked_L_channel.astype(np.float32)).clone(), requires_grad=False)
        )
        return loss + l2_loss

    def criterionHis(self, original_data, input_data, target_data, mask_src, mask_tar, index):
        if self.is_added_lumi_L1_to_his==True:
            return self.criterionHisLumiL1(original_data, input_data, target_data, mask_src, mask_tar, index)
        elif self.is_added_lumi_L2_to_his==True:
            return self.criterionHisLumiL2(original_data, input_data, target_data, mask_src, mask_tar, index)
        elif self.is_added_lumi_L1_lambda_5_to_his==True:
            return self.criterionHisLumiL1Lambda5(original_data, input_data, target_data, mask_src, mask_tar, index)
        else:
            input_data = (self.de_norm(input_data) * 255).squeeze()
            target_data = (self.de_norm(target_data) * 255).squeeze()
            mask_src = mask_src.expand(1, 3, mask_src.size(2), mask_src.size(2)).squeeze()
            mask_tar = mask_tar.expand(1, 3, mask_tar.size(2), mask_tar.size(2)).squeeze()
            input_masked = input_data * mask_src
            target_masked = target_data * mask_tar
            # dstImg = (input_masked.data).cpu().clone()
            # refImg = (target_masked.data).cpu().clone()
            input_match = histogram_matching(input_masked, target_masked, index, 3)
            input_match = self.to_var(input_match, requires_grad=False)
            loss = self.criterionL1(input_masked, input_match)
            return loss
    
    def criterionLHis(self, original_data, fake_data, mask_src, index):
        original_data = (self.de_norm(original_data) * 255).squeeze()
        fake_data = (self.de_norm(fake_data) * 255).squeeze()
        mask_src = mask_src.expand(1, 3, mask_src.size(2), mask_src.size(2)).squeeze()
        original_data_masked = original_data * mask_src
        fake_data_masked = fake_data * mask_src

        # original画像のHSL空間のL値を抽出する
        original_data_masked_image = np.array(to_pil_image(original_data_masked), dtype=np.uint8)
        original_data_masked_hls_image = cv2.cvtColor(original_data_masked_image, cv2.COLOR_RGB2HLS)
        original_data_masked_L_channel = original_data_masked_hls_image[:,:,1]

        # 生成画像にヒストグラムマッチングを施した画像のHLS空間のL値を抽出する
        fake_data_masked_image = np.array(to_pil_image(fake_data_masked), dtype=np.uint8)
        fake_data_masked_hls_image = cv2.cvtColor(fake_data_masked_image, cv2.COLOR_RGB2HLS)
        fake_data_masked_L_channel = fake_data_masked_hls_image[:,:,1]

        # Lに対するヒストグラムマッチング
        fake_data_match = luminanse_histogram_matching(fake_data_masked_L_channel, original_data_masked_L_channel, index)
        fake_data_match = self.to_var(fake_data_match, requires_grad=False)
        loss = self.criterionL1(
            fake_data_match, 
            self.to_var(torch.from_numpy(fake_data_masked_L_channel.astype(np.float32)).clone(), requires_grad=False)
        )
        return loss
    
    # Hueとサチュレーションのみに対してヒストグラムマッチングを行う関数
    def criterionHueSatHis(self, input_data, target_data, mask_src, mask_tar, index):
        input_data = (self.de_norm(input_data) * 255).squeeze()
        target_data = (self.de_norm(target_data) * 255).squeeze()
        mask_src = mask_src.expand(1, 3, mask_src.size(2), mask_src.size(2)).squeeze()
        mask_tar = mask_tar.expand(1, 3, mask_tar.size(2), mask_tar.size(2)).squeeze()
        input_masked = input_data * mask_src
        target_masked = target_data * mask_tar

        # fake画像のHSL空間のL値以外を抽出する
        input_masked_image = np.array(to_pil_image(input_masked), dtype=np.uint8)
        input_masked_hls_image = cv2.cvtColor(input_masked_image, cv2.COLOR_RGB2HLS)
        input_masked_H_channel = input_masked_hls_image[:,:,0]
        input_masked_S_channel = input_masked_hls_image[:,:,2]

        # ref画像のHSL空間のL値以外を抽出する
        target_masked_image = np.array(to_pil_image(target_masked), dtype=np.uint8)
        target_masked_hls_image = cv2.cvtColor(target_masked_image, cv2.COLOR_RGB2HLS)
        target_masked_H_channel = target_masked_hls_image[:,:,0]
        target_masked_S_channel = target_masked_hls_image[:,:,2]

        # HSに対するヒストグラムマッチング
        input_H_match = one_axis_histogram_matching(input_masked_H_channel, target_masked_H_channel, index)
        input_H_match = self.to_var(input_H_match, requires_grad=False)
        input_S_match = one_axis_histogram_matching(input_masked_S_channel, target_masked_S_channel, index)
        input_S_match = self.to_var(input_S_match, requires_grad=False)

        loss_H = self.criterionL1(self.to_var(torch.from_numpy(input_masked_H_channel.astype(np.float32)).clone(), requires_grad=False), input_H_match)
        loss_S = self.criterionL1(self.to_var(torch.from_numpy(input_masked_S_channel.astype(np.float32)).clone(), requires_grad=False), input_S_match)
        return loss_H + loss_S
    
    # Refに対してLもヒストグラムマッチングを行う
    def criterionLHisToRef(self, input_data, target_data, mask_src, mask_tar, index):
        input_data = (self.de_norm(input_data) * 255).squeeze()
        target_data = (self.de_norm(target_data) * 255).squeeze()
        mask_src = mask_src.expand(1, 3, mask_src.size(2), mask_src.size(2)).squeeze()
        mask_tar = mask_tar.expand(1, 3, mask_tar.size(2), mask_tar.size(2)).squeeze()
        input_masked = input_data * mask_src
        target_masked = target_data * mask_tar

        # fake画像のL値を抽出する
        input_masked_image = np.array(to_pil_image(input_masked), dtype=np.uint8)
        input_masked_hls_image = cv2.cvtColor(input_masked_image, cv2.COLOR_RGB2HLS)
        input_masked_L_channel = input_masked_hls_image[:,:,1]

        # ref画像のL値を抽出する
        target_masked_image = np.array(to_pil_image(target_masked), dtype=np.uint8)
        target_masked_hls_image = cv2.cvtColor(target_masked_image, cv2.COLOR_RGB2HLS)
        target_masked_L_channel = target_masked_hls_image[:,:,1]

        # HSに対するヒストグラムマッチング
        input_L_match = one_axis_histogram_matching(input_masked_L_channel, target_masked_L_channel, index)
        input_L_match = self.to_var(input_H_match, requires_grad=False)

        loss = self.criterionL1(self.to_var(torch.from_numpy(input_masked_L_channel.astype(np.float32)).clone(), requires_grad=False), input_L_match)
        return loss
    
    def criterionABHis(self, input_data, target_data, mask_src, mask_tar, index):
        input_data = (self.de_norm(input_data) * 255).squeeze()
        target_data = (self.de_norm(target_data) * 255).squeeze()
        mask_src = mask_src.expand(1, 3, mask_src.size(2), mask_src.size(2)).squeeze()
        mask_tar = mask_tar.expand(1, 3, mask_tar.size(2), mask_tar.size(2)).squeeze()
        input_masked = input_data * mask_src
        target_masked = target_data * mask_tar

        # fake画像のHSL空間のL値以外を抽出する
        input_masked_image = np.array(to_pil_image(input_masked), dtype=np.uint8)
        input_masked_hls_image = cv2.cvtColor(input_masked_image, cv2.COLOR_RGB2Lab)
        input_masked_a_channel = input_masked_hls_image[:,:,1]
        input_masked_b_channel = input_masked_hls_image[:,:,2]

        # ref画像のHSL空間のL値以外を抽出する
        target_masked_image = np.array(to_pil_image(target_masked), dtype=np.uint8)
        target_masked_hls_image = cv2.cvtColor(target_masked_image, cv2.COLOR_RGB2Lab)
        target_masked_a_channel = target_masked_hls_image[:,:,1]
        target_masked_b_channel = target_masked_hls_image[:,:,2]

        # abに対するヒストグラムマッチング
        input_a_match = one_axis_histogram_matching(input_masked_a_channel, target_masked_a_channel, index)
        input_a_match = self.to_var(input_a_match, requires_grad=False)
        input_b_match = one_axis_histogram_matching(input_masked_b_channel, target_masked_b_channel, index)
        input_b_match = self.to_var(input_b_match, requires_grad=False)

        loss_a = self.criterionL1(self.to_var(torch.from_numpy(input_masked_a_channel.astype(np.float32)).clone(), requires_grad=False), input_a_match)
        loss_b = self.criterionL1(self.to_var(torch.from_numpy(input_masked_b_channel.astype(np.float32)).clone(), requires_grad=False), input_b_match)
        return loss_a + loss_b
    
    def criterionLabHis(self, input_data, target_data, mask_src, mask_tar, index):
        input_data = (self.de_norm(input_data) * 255).squeeze()
        target_data = (self.de_norm(target_data) * 255).squeeze()
        mask_src = mask_src.expand(1, 3, mask_src.size(2), mask_src.size(2)).squeeze()
        mask_tar = mask_tar.expand(1, 3, mask_tar.size(2), mask_tar.size(2)).squeeze()
        input_masked = input_data * mask_src
        target_masked = target_data * mask_tar

        input_masked_image = np.array(to_pil_image(input_masked), dtype=np.uint8)
        input_masked_lab_image = cv2.cvtColor(input_masked_image, cv2.COLOR_RGB2Lab)
        input_masked_l_channel = input_masked_lab_image[:,:,0]
        input_masked_a_channel = input_masked_lab_image[:,:,1]
        input_masked_b_channel = input_masked_lab_image[:,:,2]

        # ref画像のHSL空間のL値以外を抽出する
        target_masked_image = np.array(to_pil_image(target_masked), dtype=np.uint8)
        target_masked_lab_image = cv2.cvtColor(target_masked_image, cv2.COLOR_RGB2Lab)
        target_masked_l_channel = target_masked_lab_image[:,:,0]
        target_masked_a_channel = target_masked_lab_image[:,:,1]
        target_masked_b_channel = target_masked_lab_image[:,:,2]

        # labに対するヒストグラムマッチング
        input_l_match = one_axis_histogram_matching(input_masked_l_channel, target_masked_l_channel, index)
        input_l_match = self.to_var(input_l_match, requires_grad=False)
        input_a_match = one_axis_histogram_matching(input_masked_a_channel, target_masked_a_channel, index)
        input_a_match = self.to_var(input_a_match, requires_grad=False)
        input_b_match = one_axis_histogram_matching(input_masked_b_channel, target_masked_b_channel, index)
        input_b_match = self.to_var(input_b_match, requires_grad=False)

        loss_l = self.criterionL1(self.to_var(torch.from_numpy(input_masked_l_channel.astype(np.float32)).clone(), requires_grad=False), input_l_match)
        loss_a = self.criterionL1(self.to_var(torch.from_numpy(input_masked_a_channel.astype(np.float32)).clone(), requires_grad=False), input_a_match)
        loss_b = self.criterionL1(self.to_var(torch.from_numpy(input_masked_b_channel.astype(np.float32)).clone(), requires_grad=False), input_b_match)
        return loss_l + loss_a + loss_b
    
    def criterionHLSHis(self, input_data, target_data, mask_src, mask_tar, index):
        input_data = (self.de_norm(input_data) * 255).squeeze()
        target_data = (self.de_norm(target_data) * 255).squeeze()
        mask_src = mask_src.expand(1, 3, mask_src.size(2), mask_src.size(2)).squeeze()
        mask_tar = mask_tar.expand(1, 3, mask_tar.size(2), mask_tar.size(2)).squeeze()
        input_masked = input_data * mask_src
        target_masked = target_data * mask_tar

        input_masked_image = np.array(to_pil_image(input_masked), dtype=np.uint8)
        input_masked_hls_image = cv2.cvtColor(input_masked_image, cv2.COLOR_RGB2HLS)
        input_masked_h_channel = input_masked_hls_image[:,:,0]
        input_masked_l_channel = input_masked_hls_image[:,:,1]
        input_masked_s_channel = input_masked_hls_image[:,:,2]

        # ref画像のHSL空間を抽出する
        target_masked_image = np.array(to_pil_image(target_masked), dtype=np.uint8)
        target_masked_hls_image = cv2.cvtColor(target_masked_image, cv2.COLOR_RGB2HLS)
        target_masked_h_channel = target_masked_hls_image[:,:,0]
        target_masked_l_channel = target_masked_hls_image[:,:,1]
        target_masked_s_channel = target_masked_hls_image[:,:,2]

        # HLSに対するヒストグラムマッチング
        input_h_match = one_axis_histogram_matching(input_masked_h_channel, target_masked_h_channel, index)
        input_h_match = self.to_var(input_h_match, requires_grad=False)
        input_l_match = one_axis_histogram_matching(input_masked_l_channel, target_masked_l_channel, index)
        input_l_match = self.to_var(input_l_match, requires_grad=False)
        input_s_match = one_axis_histogram_matching(input_masked_s_channel, target_masked_s_channel, index)
        input_s_match = self.to_var(input_s_match, requires_grad=False)

        loss_h = self.criterionL1(self.to_var(torch.from_numpy(input_masked_h_channel.astype(np.float32)).clone(), requires_grad=False), input_h_match)
        loss_l = self.criterionL1(self.to_var(torch.from_numpy(input_masked_l_channel.astype(np.float32)).clone(), requires_grad=False), input_l_match)
        loss_s = self.criterionL1(self.to_var(torch.from_numpy(input_masked_s_channel.astype(np.float32)).clone(), requires_grad=False), input_s_match)
        return loss_h + loss_l + loss_s

    def criterionLightGrad(self, fake_data, original_data, mask):
        fake = (self.de_norm(fake_data) * 255).squeeze()
        org = (self.de_norm(original_data) * 255).squeeze()
        mask = mask.expand(1, 3, mask.size(2), mask.size(2)).squeeze()

        fake_masked = fake * mask
        org_masked = org * mask

        org_masked_image = np.array(to_pil_image(org_masked), dtype=np.uint8)
        org_masked_lab_image = cv2.cvtColor(org_masked_image, cv2.COLOR_RGB2Lab)
        org_masked_L_channel = org_masked_lab_image[:,:,0]

        grad_org_x = np.gradient(org_masked_L_channel, axis=1)
        grad_org_y = np.gradient(org_masked_L_channel, axis=0)

        fake_masked_image = np.array(to_pil_image(fake_masked), dtype=np.uint8)
        fake_masked_lab_image = cv2.cvtColor(fake_masked_image, cv2.COLOR_RGB2Lab)
        fake_masked_L_channel = fake_masked_lab_image[:,:,0]

        grad_fake_x = np.gradient(fake_masked_L_channel, axis=1)
        grad_fake_y = np.gradient(fake_masked_L_channel, axis=0)

        loss = self.criterionL1(
            self.to_var(torch.from_numpy(grad_org_x.astype(np.float32)).clone(), requires_grad=False),
            self.to_var(torch.from_numpy(grad_fake_x.astype(np.float32)).clone(), requires_grad=False),
        ) + self.criterionL1(
            self.to_var(torch.from_numpy(grad_org_y.astype(np.float32)).clone(), requires_grad=False),
            self.to_var(torch.from_numpy(grad_fake_y.astype(np.float32)).clone(), requires_grad=False),
        )
        return loss
    
    def criterionGLCM(self, input_data, target_data, mask_input, mask_target) -> torch.Tensor:
        # 画像をマスキングして、CIELab空間に変換してLチャネルのみ抽出する
        target_data = (self.de_norm(target_data) * 255).squeeze()
        input_data = (self.de_norm(input_data) * 255).squeeze()
        mask_input = mask_input.expand(1, 3, mask_input.size(2), mask_input.size(2)).squeeze()
        mask_target = mask_target.expand(1, 3, mask_target.size(2), mask_target.size(2)).squeeze()
        input_masked = input_data * mask_input
        target_masked = target_data * mask_target

        input_masked_image = np.array(to_pil_image(input_masked), dtype=np.uint8)
        input_masked_lab_image = cv2.cvtColor(input_masked_image, cv2.COLOR_RGB2Lab)
        input_masked_l_channel = input_masked_lab_image[:,:,0]
        # 生成された画像のマスキングのフラグ数と、参照画像のマスクのフラグ数をカウントする
        input_masked_l_non_zero_count = np.count_nonzero(input_masked_l_channel)

        target_masked_image = np.array(to_pil_image(target_masked), dtype=np.uint8)
        target_masked_lab_image = cv2.cvtColor(target_masked_image, cv2.COLOR_RGB2Lab)
        target_masked_l_channel = target_masked_lab_image[:,:,0]
        # 生成された画像のマスキングのフラグ数と、参照画像のマスクのフラグ数をカウントする
        target_masked_l_non_zero_count = np.count_nonzero(target_masked_l_channel)
        # glcmモジュールの損失計算関数で損失を計算する
        loss = glcm.one_channel_glcm_loss(
            masked_org_img_l_channel=input_masked_l_channel,
            masked_fake_img_l_channel=target_masked_l_channel,
            masked_org_pixel_count=input_masked_l_non_zero_count,
            masked_fake_pixel_count=target_masked_l_non_zero_count,
            gray_levels=256,
            distance=10
        )
        return loss
    
    def criterionGLCMWithCount(self, input_data, target_data, mask_input, mask_target) -> torch.Tensor:
        # 画像をマスキングして、CIELab空間に変換してLチャネルのみ抽出する
        target_data = (self.de_norm(target_data) * 255).squeeze()
        input_data = (self.de_norm(input_data) * 255).squeeze()
        mask_input = mask_input.expand(1, 3, mask_input.size(2), mask_input.size(2)).squeeze()
        mask_target = mask_target.expand(1, 3, mask_target.size(2), mask_target.size(2)).squeeze()
        input_masked = input_data * mask_input
        target_masked = target_data * mask_target

        input_masked_image = np.array(to_pil_image(input_masked), dtype=np.uint8)
        input_masked_lab_image = cv2.cvtColor(input_masked_image, cv2.COLOR_RGB2Lab)
        input_masked_l_channel = input_masked_lab_image[:,:,0]
        # 生成された画像のマスキングのフラグ数と、参照画像のマスクのフラグ数をカウントする
        input_masked_l_non_zero_count = np.count_nonzero(input_masked_l_channel)

        target_masked_image = np.array(to_pil_image(target_masked), dtype=np.uint8)
        target_masked_lab_image = cv2.cvtColor(target_masked_image, cv2.COLOR_RGB2Lab)
        target_masked_l_channel = target_masked_lab_image[:,:,0]
        # 生成された画像のマスキングのフラグ数と、参照画像のマスクのフラグ数をカウントする
        target_masked_l_non_zero_count = np.count_nonzero(target_masked_l_channel)
        # glcmモジュールの損失計算関数で損失を計算する
        loss = glcm.one_channel_glcm_loss_with_count(
            masked_org_img_l_channel=input_masked_l_channel,
            masked_fake_img_l_channel=target_masked_l_channel,
            masked_org_pixel_count=input_masked_l_non_zero_count,
            masked_fake_pixel_count=target_masked_l_non_zero_count,
            gray_levels=256,
            distance=10
        )
        return loss

    def train(self):
        """Train StarGAN within a single dataset."""
        # The number of iterations per epoch
        self.iters_per_epoch = len(self.data_loader_train)
        # Start with trained model if exists
        cls_A = self.cls[0]
        cls_B = self.cls[1]
        g_lr = self.g_lr
        d_lr = self.d_lr
        if self.checkpoint:
            start = int(self.checkpoint.split('_')[0])
        else:
            start = 0
        # Start training
        self.start_time = time.time()
        print("num_epoch", self.num_epochs)
        for self.e in tqdm(range(start, self.num_epochs)):
            """
            img_Aが無化粧画像。img_Bが参照画像
            """
            for self.i, (img_A, img_B, mask_A, mask_B) in enumerate(tqdm(self.data_loader_train)):
                # By face-parsing.Pytorch
                # 1:skin, 2:l_brow, 3:r_brow, 4:l_eye, 5:r_eye, 6:eye_g, 7:l_ear, 8:r_ear, 9:ear_r,
                # 10:nose, 11:mouth, 12:u_lip, 13:l_lip, 14:neck, 15:neck_l, 16:cloth, 17:hair, 18:hat

                if self.checkpoint or self.direct:
                    if self.lips==True:
                        mask_A_lip = (mask_A==12).float() + (mask_A==13).float()
                        mask_B_lip = (mask_B==12).float() + (mask_B==13).float()
                        mask_A_lip, mask_B_lip, index_A_lip, index_B_lip = self.mask_preprocess(mask_A_lip, mask_B_lip)
                    if self.skin==True:
                        mask_A_skin = (mask_A==1).float() + (mask_A==10).float() + (mask_A==14).float()
                        mask_B_skin = (mask_B==1).float() + (mask_B==10).float() + (mask_B==14).float()
                        mask_A_skin, mask_B_skin, index_A_skin, index_B_skin = self.mask_preprocess(mask_A_skin, mask_B_skin)
                    if self.eye==True:
                        mask_A_eye_left = (mask_A==4).float()
                        mask_A_eye_right = (mask_A==5).float()
                        mask_B_eye_left = (mask_B==4).float()
                        mask_B_eye_right = (mask_B==5).float()
                        mask_A_face = (mask_A==1).float() + (mask_A==10).float()
                        mask_B_face = (mask_B==1).float() + (mask_B==10).float()
                        # avoid the situation that images with eye closed
                        if not ((mask_A_eye_left>0).any() and (mask_B_eye_left>0).any() and \
                            (mask_A_eye_right > 0).any() and (mask_B_eye_right > 0).any()):
                            print("skip")
                            continue
                        mask_A_eye_left, mask_A_eye_right = self.rebound_box(mask_A_eye_left, mask_A_eye_right, mask_A_face)
                        mask_B_eye_left, mask_B_eye_right = self.rebound_box(mask_B_eye_left, mask_B_eye_right, mask_B_face)
                        mask_A_eye_left, mask_B_eye_left, index_A_eye_left, index_B_eye_left = \
                            self.mask_preprocess(mask_A_eye_left, mask_B_eye_left)
                        mask_A_eye_right, mask_B_eye_right, index_A_eye_right, index_B_eye_right = \
                            self.mask_preprocess(mask_A_eye_right, mask_B_eye_right)

                org_A = self.to_var(img_A, requires_grad=False)
                ref_B = self.to_var(img_B, requires_grad=False)
                # ================== Train D ================== #
                # training D_A, D_A aims to distinguish class B
                # Real
                out = getattr(self, "D_" + cls_A)(ref_B) # D_Aは化粧画像を本物の化粧画像と認識するように訓練される
                d_loss_real = self.criterionGAN(out, True)
                # Fake
                """
                fake_Aは化粧画像になるように、fake_Bはすっぴんになるように学習されていく前提がある。
                本論文で達成したいのは化粧画像と化粧元画像の照明の同一性の担保なので、
                対応がorg_A : fake_A(org_Aの化粧画像), ref_B : fake_B(ref_Bのすっぴん)を考えて、
                criterion_normal(org_A, fake_A) or criterion_color_grad(org_A, fake_A)のような誤差関数を追加すればよい可能性がある。
                criterion~(fake_B, ref_B)も同様に計算してやり誤差に加える必要があるかも？->ある場合とない場合で実験してみる。
                1. オリジナル画像と出力されたfakeAをHLS空間の画像にOpenCVを使って変換する
                2. lip, skin, nose, eye_left, eye_rightでマスキングする
                （マスキングした不定形の画像の勾配の計算方法がわからない）→ マスキングした領域以外はオリジナルもfake_Aも0なので誤差は発生しないので何も考えずに勾配計算して良い。
                3. マスキングしたオリジナル画像とfakeAの輝度値の勾配をそれぞれ計算する
                4. 計算したオリジナル画像の輝度勾配AとfakeAの輝度勾配BのL1lossを誤差とする。小さければ小さいほどよい
                """
                fake_A, fake_B = self.G(org_A, ref_B) # 生成画像A,Bを得る
                fake_A = Variable(fake_A.data).detach()
                fake_B = Variable(fake_B.data).detach()
                out = getattr(self, "D_" + cls_A)(fake_A) # D_Aは生成した化粧画像をFakeの化粧画像と認識するように訓練される
                #d_loss_fake = self.get_D_loss(out, "fake")
                d_loss_fake =  self.criterionGAN(out, False)
               
                # Backward + Optimize
                d_loss = (d_loss_real + d_loss_fake) * 0.5
                getattr(self, "d_" + cls_A + "_optimizer").zero_grad()
                d_loss.backward(retain_graph=True)
                getattr(self, "d_" + cls_A + "_optimizer").step()

                # Logging
                self.loss = {}
                # self.loss['D-A-loss_real'] = d_loss_real.item()

                # training D_B, D_B aims to distinguish class A
                # Real
                out = getattr(self, "D_" + cls_B)(org_A) # D_Bは化粧前のターゲット画像を本物画像と認識するように訓練される
                d_loss_real = self.criterionGAN(out, True)
                # Fake
                out = getattr(self, "D_" + cls_B)(fake_B) # D_Bは生成した化粧前のFake画像をFakeと判定するように訓練される
                #d_loss_fake = self.get_D_loss(out, "fake")
                d_loss_fake =  self.criterionGAN(out, False)
               
                # Backward + Optimize
                d_loss = (d_loss_real + d_loss_fake) * 0.5
                getattr(self, "d_" + cls_B + "_optimizer").zero_grad()
                d_loss.backward(retain_graph=True)
                getattr(self, "d_" + cls_B + "_optimizer").step()

                # Logging
                # self.loss['D-B-loss_real'] = d_loss_real.item()

                # ================== Train G ================== #
                if (self.i + 1) % self.ndis == 0:
                    # adversarial loss, i.e. L_trans,v in the paper 

                    # identity loss
                    if self.lambda_idt > 0:
                        # G should be identity if ref_B or org_A is fed
                        idt_A1, idt_A2 = self.G(org_A, org_A)
                        idt_B1, idt_B2 = self.G(ref_B, ref_B)
                        loss_idt_A1 = self.criterionL1(idt_A1, org_A) * self.lambda_A * self.lambda_idt
                        loss_idt_A2 = self.criterionL1(idt_A2, org_A) * self.lambda_A * self.lambda_idt
                        loss_idt_B1 = self.criterionL1(idt_B1, ref_B) * self.lambda_B * self.lambda_idt
                        loss_idt_B2 = self.criterionL1(idt_B2, ref_B) * self.lambda_B * self.lambda_idt
                        # loss_idt
                        loss_idt = (loss_idt_A1 + loss_idt_A2 + loss_idt_B1 + loss_idt_B2) * 0.5
                    else:
                        loss_idt = 0
                        
                    # GAN loss D_A(G_A(A))
                    # fake_A in class B, 
                    fake_A, fake_B = self.G(org_A, ref_B)
                    pred_fake = getattr(self, "D_" + cls_A)(fake_A)
                    g_A_loss_adv = self.criterionGAN(pred_fake, True)
                    #g_loss_adv = self.get_G_loss(out)
                    # GAN loss D_B(G_B(B))
                    pred_fake = getattr(self, "D_" + cls_B)(fake_B)
                    g_B_loss_adv = self.criterionGAN(pred_fake, True)
                    rec_B, rec_A = self.G(fake_B, fake_A) # サイクル一貫性損失を計算するための生成器訓練

                    # color_histogram loss
                    g_A_loss_his = 0
                    g_B_loss_his = 0
                    # lumi loss
                    g_A_loss_lumi = 0
                    g_B_loss_lumi = 0
                    if (self.checkpoint or self.direct) and self.is_HS_matching==False and self.is_HLS_match_all==False and \
                        self.is_HS_match_skin_HLS_match_other==False and self.is_AB_match_all==False and self.is_AB_match_skin_Lab_match_other==False:
                        if self.lips==True:
                            g_A_lip_loss_his = self.criterionHis(org_A, fake_A, ref_B, mask_A_lip, mask_B_lip, index_A_lip) * self.lambda_his_lip
                            g_B_lip_loss_his = self.criterionHis(ref_B, fake_B, org_A, mask_B_lip, mask_A_lip, index_B_lip) * self.lambda_his_lip
                            g_A_loss_his += g_A_lip_loss_his
                            g_B_loss_his += g_B_lip_loss_his
                            if self.is_added_lumiL1==True:
                                g_A_lip_loss_lumi = self.criterionLumi(fake_A, org_A, mask_A_lip, 'lip_A', self.i)
                                g_B_lip_loss_lumi = self.criterionLumi(fake_B, ref_B, mask_B_lip, 'lip_B', self.i)
                                g_A_loss_lumi += g_A_lip_loss_lumi
                                g_B_loss_lumi += g_B_lip_loss_lumi
                        if self.skin==True:
                            g_A_skin_loss_his = self.criterionHis(org_A, fake_A, ref_B, mask_A_skin, mask_B_skin, index_A_skin) * self.lambda_his_skin_1
                            g_B_skin_loss_his = self.criterionHis(ref_B, fake_B, org_A, mask_B_skin, mask_A_skin, index_B_skin) * self.lambda_his_skin_2
                            g_A_loss_his += g_A_skin_loss_his
                            g_B_loss_his += g_B_skin_loss_his
                            if self.is_added_lumiL1==True:
                                g_A_skin_loss_lumi = self.criterionLumi(fake_A, org_A, mask_A_skin, 'skin_A', self.i)
                                g_B_skin_loss_lumi = self.criterionLumi(fake_B, ref_B, mask_B_skin, 'skin_B', self.i)
                                g_A_loss_lumi += g_A_skin_loss_lumi
                                g_B_loss_lumi += g_B_skin_loss_lumi
                            if self.is_added_lumi_L1_lambda_5_to_his==True:
                                g_A_skin_loss_lumi = self.criterionHisLumiL1Lambda5(fake_A, org_A, mask_A_skin, 'skin_A', self.i)
                                g_B_skin_loss_lumi = self.criterionHisLumiL1Lambda5(fake_B, ref_B, mask_B_skin, 'skin_B', self.i)
                                g_A_loss_lumi += g_A_skin_loss_lumi
                                g_B_loss_lumi += g_B_skin_loss_lumi
                        if self.eye==True:
                            g_A_eye_left_loss_his = self.criterionHis(org_A, fake_A, ref_B, mask_A_eye_left, mask_B_eye_left, index_A_eye_left) * self.lambda_his_eye
                            g_B_eye_left_loss_his = self.criterionHis(ref_B, fake_B, org_A, mask_B_eye_left, mask_A_eye_left, index_B_eye_left) * self.lambda_his_eye
                            g_A_eye_right_loss_his = self.criterionHis(org_A, fake_A, ref_B, mask_A_eye_right, mask_B_eye_right, index_A_eye_right) * self.lambda_his_eye
                            g_B_eye_right_loss_his = self.criterionHis(ref_B, fake_B, org_A, mask_B_eye_right, mask_A_eye_right, index_B_eye_right) * self.lambda_his_eye
                            g_A_loss_his += g_A_eye_left_loss_his + g_A_eye_right_loss_his
                            g_B_loss_his += g_B_eye_left_loss_his + g_B_eye_right_loss_his
                            if self.is_added_lumiL1==True:
                                g_A_eye_left_loss_lumi = self.criterionLumi(fake_A, org_A, mask_A_eye_left, 'mask_A_eye_left', self.i)
                                g_B_eye_left_loss_lumi = self.criterionLumi(fake_B, ref_B, mask_B_eye_left, 'mask_B_eye_left', self.i)
                                g_A_eye_right_loss_lumi = self.criterionLumi(fake_A, org_A, mask_A_eye_right, 'mask_A_eye_right', self.i)
                                g_B_eye_right_loss_lumi = self.criterionLumi(fake_B, ref_B, mask_B_eye_right, 'mask_B_eye_right', self.i)
                                g_A_loss_lumi += g_A_eye_left_loss_lumi + g_A_eye_right_loss_lumi
                                g_B_loss_lumi += g_B_eye_left_loss_lumi + g_B_eye_right_loss_lumi

                    # luminance gradient loss
                    # lambdaは同一で実験してみる
                    g_A_loss_lumi_match = 0
                    g_B_loss_lumi_match = 0
                    if (self.checkpoint or self.direct) and self.is_luminance_matching_to_fake:
                        if self.lips==True:
                            g_A_lip_loss_lumi_his = self.criterionLHis(org_A, fake_A, mask_A_lip, index_A_lip) * self.lambda_his_lip
                            g_B_lip_loss_lumi_his = self.criterionLHis(ref_B, fake_B, mask_B_lip, index_B_lip) * self.lambda_his_lip
                            g_A_loss_lumi_match += g_A_lip_loss_lumi_his
                            g_B_loss_lumi_match += g_B_lip_loss_lumi_his
                        if self.skin==True:
                            g_A_skin_loss_lumi_his = self.criterionLHis(org_A, fake_A, mask_A_skin, index_A_skin) * self.lambda_his_skin_1
                            g_B_skin_loss_lumi_his = self.criterionLHis(ref_B, fake_B, mask_B_skin, index_B_skin) * self.lambda_his_skin_2
                            g_A_loss_lumi_match += g_A_skin_loss_lumi_his
                            g_B_loss_lumi_match += g_B_skin_loss_lumi_his
                        if self.eye==True:
                            g_A_eye_left_loss_lumi_his = self.criterionLHis(org_A, fake_A, mask_A_eye_left, index_A_eye_left) * self.lambda_his_eye
                            g_B_eye_left_loss_lumi_his = self.criterionLHis(ref_B, fake_B, mask_B_eye_left, index_B_eye_left) * self.lambda_his_eye
                            g_A_eye_right_loss_lumi_his = self.criterionLHis(org_A, fake_A, mask_A_eye_right, index_A_eye_right) * self.lambda_his_eye
                            g_B_eye_right_loss_lumi_his = self.criterionLHis(ref_B, fake_B, mask_B_eye_right, index_B_eye_right) * self.lambda_his_eye
                            g_A_loss_lumi_match += g_A_eye_left_loss_lumi_his + g_A_eye_right_loss_lumi_his
                            g_B_loss_lumi_match += g_B_eye_left_loss_lumi_his + g_B_eye_right_loss_lumi_his
                    
                    # HS loss
                    # lambdaは同一で実験してみる
                    g_A_loss_HS_match = 0
                    g_B_loss_HS_match = 0
                    if (self.checkpoint or self.direct) and self.is_HS_matching==True:
                        if self.lips==True:
                            g_A_lip_loss_HS_his = self.criterionHueSatHis(fake_A, ref_B, mask_A_lip, mask_B_lip, index_A_lip) * self.lambda_his_lip
                            g_B_lip_loss_HS_his = self.criterionHueSatHis(fake_B, org_A, mask_B_lip, mask_A_lip, index_B_lip) * self.lambda_his_lip
                            g_A_loss_HS_match += g_A_lip_loss_HS_his
                            g_B_loss_HS_match += g_B_lip_loss_HS_his
                        if self.skin==True:
                            g_A_skin_loss_HS_his = self.criterionHueSatHis(fake_A, ref_B, mask_A_skin, mask_B_skin, index_A_skin) * self.lambda_his_skin_1
                            g_B_skin_loss_HS_his = self.criterionHueSatHis(fake_B, org_A, mask_B_skin, mask_A_skin, index_B_skin) * self.lambda_his_skin_2
                            g_A_loss_HS_match += g_A_skin_loss_HS_his
                            g_B_loss_HS_match += g_B_skin_loss_HS_his
                        if self.eye==True:
                            g_A_eye_left_loss_HS_his = self.criterionHueSatHis(fake_A, ref_B, mask_A_eye_left, mask_B_eye_left, index_A_eye_left) * self.lambda_his_eye
                            g_B_eye_left_loss_HS_his = self.criterionHueSatHis(fake_B, org_A, mask_B_eye_left, mask_A_eye_left, index_B_eye_left) * self.lambda_his_eye
                            g_A_eye_right_loss_HS_his = self.criterionHueSatHis(fake_A, ref_B, mask_A_eye_right, mask_B_eye_right, index_A_eye_right) * self.lambda_his_eye
                            g_B_eye_right_loss_HS_his = self.criterionHueSatHis(fake_B, org_A, mask_B_eye_right, mask_A_eye_right, index_B_eye_right) * self.lambda_his_eye
                            g_A_loss_HS_match += g_A_eye_left_loss_HS_his + g_A_eye_right_loss_HS_his
                            g_B_loss_HS_match += g_B_eye_left_loss_HS_his + g_B_eye_right_loss_HS_his
                    
                    # 8/17以降
                    # HLS All loss
                    g_A_loss_HLS_match = 0
                    g_B_loss_HLS_match = 0
                    if (self.checkpoint or self.direct) and self.is_HLS_match_all==True:
                        if self.lips==True:
                            g_A_lip_loss_HLS_his = self.criterionHLSHis(fake_A, ref_B, mask_A_lip, mask_B_lip, index_A_lip) * self.lambda_his_lip
                            g_B_lip_loss_HLS_his = self.criterionHLSHis(fake_B, org_A, mask_B_lip, mask_A_lip, index_B_lip) * self.lambda_his_lip
                            g_A_loss_HLS_match += g_A_lip_loss_HLS_his
                            g_B_loss_HLS_match += g_B_lip_loss_HLS_his
                        if self.skin==True:
                            g_A_skin_loss_HLS_his = self.criterionHLSHis(fake_A, ref_B, mask_A_skin, mask_B_skin, index_A_skin) * self.lambda_his_skin_1
                            g_B_skin_loss_HLS_his = self.criterionHLSHis(fake_B, org_A, mask_B_skin, mask_A_skin, index_B_skin) * self.lambda_his_skin_2
                            g_A_loss_HLS_match += g_A_skin_loss_HLS_his
                            g_B_loss_HLS_match += g_B_skin_loss_HLS_his
                        if self.eye==True:
                            g_A_eye_left_loss_HLS_his = self.criterionHLSHis(fake_A, ref_B, mask_A_eye_left, mask_B_eye_left, index_A_eye_left) * self.lambda_his_eye
                            g_B_eye_left_loss_HLS_his = self.criterionHLSHis(fake_B, org_A, mask_B_eye_left, mask_A_eye_left, index_B_eye_left) * self.lambda_his_eye
                            g_A_eye_right_loss_HLS_his = self.criterionHLSHis(fake_A, ref_B, mask_A_eye_right, mask_B_eye_right, index_A_eye_right) * self.lambda_his_eye
                            g_B_eye_right_loss_HLS_his = self.criterionHLSHis(fake_B, org_A, mask_B_eye_right, mask_A_eye_right, index_B_eye_right) * self.lambda_his_eye
                            g_A_loss_HLS_match += g_A_eye_left_loss_HLS_his + g_A_eye_right_loss_HLS_his
                            g_B_loss_HLS_match += g_B_eye_left_loss_HLS_his + g_B_eye_right_loss_HLS_his

                    # HS for Skin, HLS for Other
                    g_A_loss_HS_skin_HLS_other_match = 0
                    g_B_loss_HS_skin_HLS_other_match = 0
                    if (self.checkpoint or self.direct) and self.is_HS_match_skin_HLS_match_other==True:
                        if self.lips==True:
                            g_A_lip_loss_HLS_his = self.criterionHLSHis(fake_A, ref_B, mask_A_lip, mask_B_lip, index_A_lip) * self.lambda_his_lip
                            g_B_lip_loss_HLS_his = self.criterionHLSHis(fake_B, org_A, mask_B_lip, mask_A_lip, index_B_lip) * self.lambda_his_lip
                            g_A_loss_HS_skin_HLS_other_match += g_A_lip_loss_HLS_his
                            g_B_loss_HS_skin_HLS_other_match += g_B_lip_loss_HLS_his
                        if self.skin==True:
                            g_A_skin_loss_HS_his = self.criterionHueSatHis(fake_A, ref_B, mask_A_skin, mask_B_skin, index_A_skin) * self.lambda_his_skin_1
                            g_B_skin_loss_HS_his = self.criterionHueSatHis(fake_B, org_A, mask_B_skin, mask_A_skin, index_B_skin) * self.lambda_his_skin_2
                            g_A_loss_HS_skin_HLS_other_match += g_A_skin_loss_HS_his
                            g_B_loss_HS_skin_HLS_other_match += g_B_skin_loss_HS_his
                        if self.eye==True:
                            g_A_eye_left_loss_HLS_his = self.criterionHLSHis(fake_A, ref_B, mask_A_eye_left, mask_B_eye_left, index_A_eye_left) * self.lambda_his_eye
                            g_B_eye_left_loss_HLS_his = self.criterionHLSHis(fake_B, org_A, mask_B_eye_left, mask_A_eye_left, index_B_eye_left) * self.lambda_his_eye
                            g_A_eye_right_loss_HLS_his = self.criterionHLSHis(fake_A, ref_B, mask_A_eye_right, mask_B_eye_right, index_A_eye_right) * self.lambda_his_eye
                            g_B_eye_right_loss_HLS_his = self.criterionHLSHis(fake_B, org_A, mask_B_eye_right, mask_A_eye_right, index_B_eye_right) * self.lambda_his_eye
                            g_A_loss_HS_skin_HLS_other_match += g_A_eye_left_loss_HLS_his + g_A_eye_right_loss_HLS_his
                            g_B_loss_HS_skin_HLS_other_match += g_B_eye_left_loss_HLS_his + g_B_eye_right_loss_HLS_his
                    
                    # AB All loss
                    g_A_loss_AB_match = 0
                    g_B_loss_AB_match = 0
                    if (self.checkpoint or self.direct) and self.is_AB_match_all==True:
                        if self.lips==True:
                            g_A_lip_loss_AB_his = self.criterionABHis(fake_A, ref_B, mask_A_lip, mask_B_lip, index_A_lip) * self.lambda_his_lip
                            g_B_lip_loss_AB_his = self.criterionABHis(fake_B, org_A, mask_B_lip, mask_A_lip, index_B_lip) * self.lambda_his_lip
                            g_A_loss_AB_match += g_A_lip_loss_AB_his
                            g_B_loss_AB_match += g_B_lip_loss_AB_his
                        if self.skin==True:
                            g_A_skin_loss_AB_his = self.criterionABHis(fake_A, ref_B, mask_A_skin, mask_B_skin, index_A_skin) * self.lambda_his_skin_1
                            g_B_skin_loss_AB_his = self.criterionABHis(fake_B, org_A, mask_B_skin, mask_A_skin, index_B_skin) * self.lambda_his_skin_2
                            g_A_loss_AB_match += g_A_skin_loss_AB_his
                            g_B_loss_AB_match += g_B_skin_loss_AB_his
                        if self.eye==True:
                            g_A_eye_left_loss_AB_his = self.criterionABHis(fake_A, ref_B, mask_A_eye_left, mask_B_eye_left, index_A_eye_left) * self.lambda_his_eye
                            g_B_eye_left_loss_AB_his = self.criterionABHis(fake_B, org_A, mask_B_eye_left, mask_A_eye_left, index_B_eye_left) * self.lambda_his_eye
                            g_A_eye_right_loss_AB_his = self.criterionABHis(fake_A, ref_B, mask_A_eye_right, mask_B_eye_right, index_A_eye_right) * self.lambda_his_eye
                            g_B_eye_right_loss_AB_his = self.criterionABHis(fake_B, org_A, mask_B_eye_right, mask_A_eye_right, index_B_eye_right) * self.lambda_his_eye
                            g_A_loss_AB_match += g_A_eye_left_loss_AB_his + g_A_eye_right_loss_AB_his
                            g_B_loss_AB_match += g_B_eye_left_loss_AB_his + g_B_eye_right_loss_AB_his
                    
                    # ab for Skin, Lab for Other
                    g_A_loss_ab_skin_Lab_other_match = 0
                    g_B_loss_ab_skin_Lab_other_match = 0
                    if (self.checkpoint or self.direct) and self.is_AB_match_skin_Lab_match_other==True:
                        if self.lips==True:
                            g_A_lip_loss_Lab_his = self.criterionLabHis(fake_A, ref_B, mask_A_lip, mask_B_lip, index_A_lip) * self.lambda_his_lip
                            g_B_lip_loss_Lab_his = self.criterionLabHis(fake_B, org_A, mask_B_lip, mask_A_lip, index_B_lip) * self.lambda_his_lip
                            g_A_loss_ab_skin_Lab_other_match += g_A_lip_loss_Lab_his
                            g_B_loss_ab_skin_Lab_other_match += g_B_lip_loss_Lab_his
                        if self.skin==True:
                            g_A_skin_loss_AB_his = self.criterionABHis(fake_A, ref_B, mask_A_skin, mask_B_skin, index_A_skin) * self.lambda_his_skin_1
                            g_B_skin_loss_AB_his = self.criterionABHis(fake_B, org_A, mask_B_skin, mask_A_skin, index_B_skin) * self.lambda_his_skin_2
                            g_A_loss_ab_skin_Lab_other_match += g_A_skin_loss_AB_his
                            g_B_loss_ab_skin_Lab_other_match += g_B_skin_loss_AB_his
                        if self.eye==True:
                            g_A_eye_left_loss_Lab_his = self.criterionLabHis(fake_A, ref_B, mask_A_eye_left, mask_B_eye_left, index_A_eye_left) * self.lambda_his_eye
                            g_B_eye_left_loss_Lab_his = self.criterionLabHis(fake_B, org_A, mask_B_eye_left, mask_A_eye_left, index_B_eye_left) * self.lambda_his_eye
                            g_A_eye_right_loss_Lab_his = self.criterionLabHis(fake_A, ref_B, mask_A_eye_right, mask_B_eye_right, index_A_eye_right) * self.lambda_his_eye
                            g_B_eye_right_loss_Lab_his = self.criterionLabHis(fake_B, org_A, mask_B_eye_right, mask_A_eye_right, index_B_eye_right) * self.lambda_his_eye
                            g_A_loss_ab_skin_Lab_other_match += g_A_eye_left_loss_Lab_his + g_A_eye_right_loss_Lab_his
                            g_B_loss_ab_skin_Lab_other_match += g_B_eye_left_loss_Lab_his + g_B_eye_right_loss_Lab_his
                    
                    g_A_grad_loss = 0
                    g_B_grad_loss = 0
                    if (self.checkpoint or self.direct) and self.is_grad_loss==True:
                        if self.lips==True:
                            g_A_lip_loss_grad = self.criterionLightGrad(fake_A, org_A, mask_A_lip) * 0.1
                            g_B_lip_loss_grad = self.criterionLightGrad(fake_B, ref_B, mask_B_lip) * 0.1
                            g_A_grad_loss += g_A_lip_loss_grad
                            g_B_grad_loss += g_B_lip_loss_grad
                        if self.skin==True:
                            g_A_skin_loss_grad = self.criterionLightGrad(fake_A, org_A, mask_A_skin) * 0.1
                            g_B_skin_loss_grad = self.criterionLightGrad(fake_B, ref_B, mask_B_skin) * 0.1
                            g_A_grad_loss += g_A_skin_loss_grad
                            g_B_grad_loss += g_B_skin_loss_grad
                        if self.eye==True:
                            g_A_eye_left_loss_grad = self.criterionLightGrad(fake_A, org_A, mask_A_eye_left) * 0.1
                            g_B_eye_left_loss_grad = self.criterionLightGrad(fake_B, ref_B, mask_B_eye_left) * 0.1
                            g_A_eye_right_loss_grad = self.criterionLightGrad(fake_A, org_A, mask_A_eye_right) * 0.1
                            g_B_eye_right_loss_grad = self.criterionLightGrad(fake_B, ref_B, mask_B_eye_right) * 0.1
                            g_A_grad_loss += g_A_eye_left_loss_grad + g_A_eye_right_loss_grad
                            g_B_grad_loss += g_B_eye_left_loss_grad + g_B_eye_right_loss_grad
                    
                    # GLCMの損失を誤差として加え、生成される画像の肌質がより化粧参照画像に近づくか、シミなどが隠れるようになるかを検証するための損失項
                    g_A_glcm_loss = 0
                    g_B_glcm_loss = 0
                    if (self.checkpoint or self.direct) and self.is_glcm_loss==True:
                        if self.lips==True:
                            g_A_lip_glcm_loss = self.criterionGLCM(fake_A, ref_B, mask_A_lip, mask_B_lip) * 10
                            g_B_lip_glcm_loss = self.criterionGLCM(fake_B, org_A, mask_B_lip, mask_A_lip) * 10
                            g_A_glcm_loss += g_A_lip_glcm_loss
                            g_B_glcm_loss += g_B_lip_glcm_loss
                        if self.skin==True:
                            g_A_skin_glcm_loss = self.criterionGLCM(fake_A, ref_B, mask_A_skin, mask_B_skin) * 10
                            g_B_skin_glcm_loss = self.criterionGLCM(fake_B, org_A, mask_B_skin, mask_A_skin) * 10
                            g_A_glcm_loss += g_A_skin_glcm_loss
                            g_B_glcm_loss += g_B_skin_glcm_loss
                        if self.eye==True:
                            g_A_eye_left_glcm_loss= self.criterionGLCM(fake_A, ref_B, mask_A_eye_left, mask_B_eye_left) * 10
                            g_B_eye_left_glcm_loss = self.criterionGLCM(fake_B, org_A, mask_B_eye_left, mask_A_eye_left) * 10
                            g_A_eye_right_glcm_loss = self.criterionGLCM(fake_A, ref_B, mask_A_eye_right, mask_B_eye_right) * 10
                            g_B_eye_right_glcm_loss = self.criterionGLCM(fake_B, org_A, mask_B_eye_right, mask_A_eye_right) * 10
                            g_A_glcm_loss += g_A_eye_left_glcm_loss + g_A_eye_right_glcm_loss
                            g_B_glcm_loss += g_B_eye_left_glcm_loss + g_B_eye_right_glcm_loss
                    
                    # ピクセルの出現確率でGLCMの損失を計算
                    g_A_glcm_with_count_loss = 0
                    g_B_glcm_with_count_loss = 0
                    if (self.checkpoint or self.direct) and self.is_glcm_loss_with_count==True:
                        if self.lips==True:
                            g_A_lip_glcm_with_count_loss = self.criterionGLCMWithCount(fake_A, ref_B, mask_A_lip, mask_B_lip) * 10
                            g_B_lip_glcm_with_count_loss = self.criterionGLCMWithCount(fake_B, org_A, mask_B_lip, mask_A_lip) * 10
                            g_A_glcm_with_count_loss += g_A_lip_glcm_with_count_loss
                            g_B_glcm_with_count_loss += g_B_lip_glcm_with_count_loss
                        if self.skin==True:
                            g_A_skin_glcm_with_count_loss = self.criterionGLCMWithCount(fake_A, ref_B, mask_A_skin, mask_B_skin) * 10
                            g_B_skin_glcm_with_count_loss = self.criterionGLCMWithCount(fake_B, org_A, mask_B_skin, mask_A_skin) * 10
                            g_A_glcm_with_count_loss += g_A_skin_glcm_with_count_loss
                            g_B_glcm_with_count_loss += g_B_skin_glcm_with_count_loss
                        if self.eye==True:
                            g_A_eye_left_glcm_with_count_loss= self.criterionGLCMWithCount(fake_A, ref_B, mask_A_eye_left, mask_B_eye_left) * 10
                            g_B_eye_left_glcm_with_count_loss = self.criterionGLCMWithCount(fake_B, org_A, mask_B_eye_left, mask_A_eye_left) * 10
                            g_A_eye_right_glcm_with_count_loss = self.criterionGLCMWithCount(fake_A, ref_B, mask_A_eye_right, mask_B_eye_right) * 10
                            g_B_eye_right_glcm_with_count_loss = self.criterionGLCMWithCount(fake_B, org_A, mask_B_eye_right, mask_A_eye_right) * 10
                            g_A_glcm_with_count_loss += g_A_eye_left_glcm_with_count_loss + g_A_eye_right_glcm_with_count_loss
                            g_B_glcm_with_count_loss += g_B_eye_left_glcm_with_count_loss + g_B_eye_right_glcm_with_count_loss
                    
                    # オリジナルに対してのGLCM
                    g_A_glcm_to_org_loss = 0
                    g_B_glcm_to_org_loss = 0
                    if (self.checkpoint or self.direct) and self.is_glcm_to_org_loss==True:
                        if self.lips==True:
                            g_A_lip_glcm_to_org_loss = self.criterionGLCMWithCount(fake_A, org_A, mask_A_lip, mask_A_lip)
                            g_B_lip_glcm_to_org_loss = self.criterionGLCMWithCount(fake_B, ref_B, mask_B_lip, mask_B_lip)
                            g_A_glcm_to_org_loss += g_A_lip_glcm_to_org_loss
                            g_B_glcm_to_org_loss += g_B_lip_glcm_to_org_loss
                        if self.skin==True:
                            g_A_skin_glcm_to_org_loss = self.criterionGLCMWithCount(fake_A, org_A, mask_A_skin, mask_A_skin)
                            g_B_skin_glcm_to_org_loss = self.criterionGLCMWithCount(fake_B, ref_B, mask_B_skin, mask_B_skin)
                            g_A_glcm_to_org_loss += g_A_skin_glcm_to_org_loss
                            g_B_glcm_to_org_loss += g_B_skin_glcm_to_org_loss
                        if self.eye==True:
                            g_A_eye_left_glcm_to_org_loss= self.criterionGLCMWithCount(fake_A, org_A, mask_A_eye_left, mask_A_eye_left)
                            g_B_eye_left_glcm_to_org_loss = self.criterionGLCMWithCount(fake_B, ref_B, mask_B_eye_left, mask_B_eye_left)
                            g_A_eye_right_glcm_to_org_loss = self.criterionGLCMWithCount(fake_A, org_A, mask_A_eye_right, mask_A_eye_right)
                            g_B_eye_right_glcm_to_org_loss = self.criterionGLCMWithCount(fake_B, ref_B, mask_B_eye_right, mask_B_eye_right)
                            g_A_glcm_to_org_loss += g_A_eye_left_glcm_to_org_loss + g_A_eye_right_glcm_to_org_loss
                            g_B_glcm_to_org_loss += g_B_eye_left_glcm_to_org_loss + g_B_eye_right_glcm_to_org_loss

                    # cycle loss
                    g_loss_rec_A = self.criterionL1(rec_A, org_A) * self.lambda_A
                    g_loss_rec_B = self.criterionL1(rec_B, ref_B) * self.lambda_B

                    # vgg loss
                    vgg_org=self.vgg_forward(self.vgg,org_A)
                    vgg_org = Variable(vgg_org.data).detach()
                    vgg_fake_A=self.vgg_forward(self.vgg,fake_A)
                    g_loss_A_vgg = self.criterionL2(vgg_fake_A, vgg_org) * self.lambda_A * self.lambda_vgg

                    vgg_ref=self.vgg_forward(self.vgg, ref_B)
                    vgg_ref = Variable(vgg_ref.data).detach()
                    vgg_fake_B=self.vgg_forward(self.vgg,fake_B)
                    g_loss_B_vgg = self.criterionL2(vgg_fake_B, vgg_ref) * self.lambda_B * self.lambda_vgg

                    loss_rec = (g_loss_rec_A + g_loss_rec_B + g_loss_A_vgg + g_loss_B_vgg) * 0.5
                    
                    # Combined loss
                    g_loss = g_A_loss_adv + g_B_loss_adv + loss_rec + loss_idt
                    if self.checkpoint or self.direct:
                        g_loss = g_A_loss_adv + g_B_loss_adv + loss_rec + loss_idt + g_A_loss_his + g_B_loss_his + \
                            g_A_loss_lumi_match + g_B_loss_lumi_match + g_A_loss_HS_match + g_B_loss_HS_match + \
                            g_A_loss_HLS_match + g_B_loss_HLS_match + g_A_loss_HS_skin_HLS_other_match + g_B_loss_HS_skin_HLS_other_match + \
                            g_A_loss_AB_match + g_B_loss_AB_match + g_A_loss_ab_skin_Lab_other_match + g_B_loss_ab_skin_Lab_other_match + g_A_grad_loss + g_B_grad_loss
                        if self.is_added_lumiL1==True:
                            g_loss = g_loss + g_A_loss_lumi + g_B_loss_lumi
                        if self.is_glcm_loss==True:
                            g_loss = g_loss + g_A_glcm_loss + g_B_glcm_loss
                        if self.is_glcm_loss_with_count==True:
                            g_loss = g_loss + g_A_glcm_with_count_loss + g_B_glcm_with_count_loss
                        if self.is_glcm_to_org_loss==True:
                            g_loss = g_loss + g_A_glcm_to_org_loss + g_B_glcm_to_org_loss
                    
                    self.g_optimizer.zero_grad()
                    g_loss.backward(retain_graph=True)
                    self.g_optimizer.step()

                    # # Logging
                    self.loss['G-A-loss-adv'] = g_A_loss_adv.item()
                    self.loss['G-B-loss-adv'] = g_A_loss_adv.item()
                    self.loss['G-loss-org'] = g_loss_rec_A.item()
                    self.loss['G-loss-ref'] = g_loss_rec_B.item()
                    self.loss['G-loss-idt'] = loss_idt.item()
                    self.loss['G-loss-img-rec'] = (g_loss_rec_A + g_loss_rec_B).item()
                    self.loss['G-loss-vgg-rec'] = (g_loss_A_vgg + g_loss_B_vgg).item()
                    if self.direct and self.is_HS_matching==False and self.is_HLS_match_all==False and \
                        self.is_HS_match_skin_HLS_match_other==False and self.is_AB_match_all==False and self.is_AB_match_skin_Lab_match_other==False:
                        self.loss['G-A-loss-his'] = g_A_loss_his.item()
                        self.loss['G-B-loss-his'] = g_B_loss_his.item()
                    if self.is_luminance_matching_to_fake:
                        self.loss['G-A-loss-lumi-his-match'] = g_A_loss_lumi_match.item()
                        self.loss['G-B-loss-lumi-his-match'] = g_B_loss_lumi_match.item()
                    if self.is_HS_matching:
                        self.loss['G-A-loss-HS-his-match'] = g_A_loss_HS_match.item()
                        self.loss['G-B-loss-HS-his-match'] = g_B_loss_HS_match.item()
                    if self.is_HLS_match_all:
                        self.loss['G-A-loss-HLS-match'] = g_A_loss_HLS_match.item()
                        self.loss['G-B-loss-HLS-match'] = g_B_loss_HLS_match.item()
                    if self.is_HS_match_skin_HLS_match_other:
                        self.loss['G-A-loss-HS-skin-HLS-other-match'] = g_A_loss_HS_skin_HLS_other_match.item()
                        self.loss['G-B-loss-HS-skin-HLS-other-match'] = g_B_loss_HS_skin_HLS_other_match.item()
                    if self.is_AB_match_all:
                        self.loss['G-A-loss-AB-match'] = g_A_loss_AB_match.item()
                        self.loss['G-B-loss-AB-match'] = g_B_loss_AB_match.item()
                    if self.is_AB_match_skin_Lab_match_other:
                        self.loss['G-A-loss-ab-skin-Lab-other-match'] = g_A_loss_ab_skin_Lab_other_match.item()
                        self.loss['G-B-loss-ab-skin-Lab-other-match'] = g_B_loss_ab_skin_Lab_other_match.item()
                    if self.is_added_lumiL1==True:
                            self.loss['G-A-loss-lumi'] = g_A_loss_lumi.item()
                            self.loss['G-B-loss-lumi'] = g_B_loss_lumi.item()
                    if self.is_grad_loss==True:
                        self.loss['G-A-loss-grad'] = g_A_grad_loss.item()
                        print('g_A_grad_loss:',g_A_grad_loss.item())
                        self.loss['G-B-loss-grad'] = g_B_grad_loss.item()
                        print('g_B_grad_loss:',g_B_grad_loss.item())
                    if self.is_glcm_loss==True:
                        self.loss['G-A-glcm-loss'] = g_A_glcm_loss.item()
                        self.loss['G-B-glcm-loss'] = g_B_glcm_loss.item()
                    if self.is_glcm_loss_with_count==True:
                        self.loss['G-A-glcm-with-count-loss'] = g_A_glcm_with_count_loss.item()
                        self.loss['G-B-glcm-with-countloss'] = g_B_glcm_with_count_loss.item()
                    if self.is_glcm_to_org_loss==True:
                        self.loss['G-A-glcm-to-org-loss'] = g_A_glcm_to_org_loss.item()
                        self.loss['G-B-glcm-to-org-loss'] = g_B_glcm_to_org_loss.item()
                # Print out log info


                #plot the figures
                # for key_now in self.loss.keys():
                #     plot_fig.plot(key_now, self.loss[key_now])

                #save the images
                if (self.i + 1) % self.vis_step == 0:
                    print("Saving middle output...")
                    self.vis_train([org_A, ref_B, fake_A, fake_B, rec_A, rec_B])

                if self.i%10==0:
                    self.writer.add_scalar('losses/GA-loss-adv', g_A_loss_adv.item(), self.i)
                    self.writer.add_scalar('losses/GB-loss-adv', g_B_loss_adv.item(), self.i)
                    self.writer.add_scalar('losses/rec-org', g_loss_rec_A.item(), self.i)
                    self.writer.add_scalar('losses/rec-ref', g_loss_rec_B.item(), self.i)
                    self.writer.add_scalar('losses/vgg-A', g_loss_A_vgg.item(), self.i)
                    self.writer.add_scalar('losses/vgg-B', g_loss_B_vgg.item(), self.i)
                    # if self.lambda_spl>0:
                    #     self.writer.add_scalar('mkup-spl/SPL-A', spl_loss_A.item(), self.i)
                    #     self.writer.add_scalar('mkup-spl/SPL-B', spl_loss_B.item(), self.i)
                    #     self.writer.add_scalar('mkup-spl/GPL-A', gpl_value_A.item(), self.i)
                    #     self.writer.add_scalar('mkup-spl/GPL-B', gpl_value_B.item(), self.i)
                    #     self.writer.add_scalar('mkup-spl/CPL-A', cpl_value_A.item(), self.i)
                    #     self.writer.add_scalar('mkup-spl/CPL-B', cpl_value_B.item(), self.i)
                    if self.eye:
                        if self.is_added_lumiL1==True:
                            self.writer.add_scalar('mkup-lumi/eyes', (g_A_eye_left_loss_lumi + g_A_eye_right_loss_lumi).item(), self.i)
                        if self.is_added_lumi_L1_to_his==True:
                            self.writer.add_scalar('mkup-his-lumi/eyes', (g_A_eye_left_loss_his + g_A_eye_right_loss_his).item(), self.i)
                        if self.is_luminance_matching_to_fake==True:
                            self.writer.add_scalar('mkup-lumi-his-match/eyes', (g_A_eye_left_loss_lumi_his + g_A_eye_right_loss_lumi_his).item(), self.i)
                        if self.is_HS_matching==True:
                            self.writer.add_scalar('mkup-HS-his-match/eyes', (g_A_eye_left_loss_HS_his + g_A_eye_right_loss_HS_his + g_B_eye_left_loss_HS_his + g_B_eye_right_loss_HS_his).item(), self.i)
                        
                        # 8/17
                        if self.is_HLS_match_all==True:
                            self.writer.add_scalar('mkup-is_HLS_match_all/eyes', (g_A_eye_left_loss_HLS_his + g_B_eye_left_loss_HLS_his + g_A_eye_right_loss_HLS_his + g_B_eye_right_loss_HLS_his).item(), self.i)
                        if self.is_HS_match_skin_HLS_match_other==True:
                            self.writer.add_scalar('mkup-is_HS_match_skin_HLS_match_other/eyes', (g_A_eye_left_loss_HLS_his + g_B_eye_left_loss_HLS_his + g_A_eye_right_loss_HLS_his + g_B_eye_right_loss_HLS_his).item(), self.i)
                        if self.is_AB_match_all==True:
                            self.writer.add_scalar('mkup-is_AB_match_all/eyes', (g_A_eye_left_loss_AB_his + g_B_eye_left_loss_AB_his + g_A_eye_right_loss_AB_his + g_B_eye_right_loss_AB_his).item(), self.i)
                        if self.is_AB_match_skin_Lab_match_other==True:
                            self.writer.add_scalar('mkup-is_AB_match_skin_Lab_match_other/eyes', (g_A_eye_left_loss_Lab_his + g_B_eye_left_loss_Lab_his + g_A_eye_right_loss_Lab_his + g_B_eye_right_loss_Lab_his).item(), self.i)

                        if self.is_HS_matching==False and self.is_HLS_match_all==False and \
                            self.is_HS_match_skin_HLS_match_other==False and self.is_AB_match_all==False and self.is_AB_match_skin_Lab_match_other==False:
                            self.writer.add_scalar('mkup-hist/eyes', (g_A_eye_left_loss_his + g_A_eye_right_loss_his).item(), self.i)
                        
                        if self.is_grad_loss==True:
                            self.writer.add_scalar('mkup-grad/eyes', (g_A_eye_left_loss_grad + g_A_eye_right_loss_grad).item(), self.i)
                            print('g_A_eye_left_loss_grad + g_B_eye_left_loss_grad', (g_A_eye_left_loss_grad + g_A_eye_right_loss_grad).item())
                        
                        if self.is_glcm_loss==True:
                            self.writer.add_scalar('mkup-glcm/eyes', (g_A_eye_left_glcm_loss + g_A_eye_right_glcm_loss + g_B_eye_left_glcm_loss + g_B_eye_right_glcm_loss).item(), self.i)

                        if self.is_glcm_loss_with_count==True:
                            self.writer.add_scalar('mkup-glcm-with-count/eyes', (g_A_eye_left_glcm_with_count_loss + g_A_eye_right_glcm_with_count_loss + g_B_eye_left_glcm_with_count_loss + g_B_eye_right_glcm_with_count_loss).item(), self.i)
                        
                        if self.is_glcm_to_org_loss==True:
                            self.writer.add_scalar('mkup-glcm-to-org/eyes', (g_A_eye_left_glcm_to_org_loss + g_A_eye_right_glcm_to_org_loss + g_B_eye_left_glcm_to_org_loss + g_B_eye_right_glcm_to_org_loss).item(), self.i)
                    if self.lips:
                        if self.is_added_lumiL1==True:
                            self.writer.add_scalar('mkup-lumi/lips', (g_A_lip_loss_lumi + g_B_lip_loss_lumi).item(), self.i)
                        if self.is_added_lumi_L1_to_his==True:
                            self.writer.add_scalar('mkup-his-lumi/lips', (g_A_lip_loss_his + g_B_lip_loss_his).item(), self.i)
                        if self.is_luminance_matching_to_fake==True:
                            self.writer.add_scalar('mkup-lumi-his-match/lips', (g_A_lip_loss_lumi_his+g_B_lip_loss_lumi_his).item(), self.i)
                        if self.is_HS_matching==True:
                            self.writer.add_scalar('mkup-HS-his-match/lips', (g_A_lip_loss_HS_his + g_B_lip_loss_HS_his).item(), self.i)
                        
                        # 8/17
                        if self.is_HLS_match_all==True:
                            self.writer.add_scalar('mkup-is_HLS_match_all/lips', (g_A_lip_loss_HLS_his + g_B_lip_loss_HLS_his).item(), self.i)
                        if self.is_HS_match_skin_HLS_match_other==True:
                            self.writer.add_scalar('mkup-is_HS_match_skin_HLS_match_other/lips', (g_A_lip_loss_HLS_his + g_B_lip_loss_HLS_his).item(), self.i)
                        if self.is_AB_match_all==True:
                            self.writer.add_scalar('mkup-is_AB_match_all/lips', (g_A_lip_loss_AB_his + g_B_lip_loss_AB_his).item(), self.i)
                        if self.is_AB_match_skin_Lab_match_other==True:
                            self.writer.add_scalar('mkup-is_AB_match_skin_Lab_match_other/lips', (g_A_lip_loss_Lab_his + g_B_lip_loss_Lab_his).item(), self.i)

                        if self.is_HS_matching==False and self.is_HLS_match_all==False and \
                            self.is_HS_match_skin_HLS_match_other==False and self.is_AB_match_all==False and self.is_AB_match_skin_Lab_match_other==False:
                            self.writer.add_scalar('mkup-hist/lips', (g_A_lip_loss_his+g_B_lip_loss_his).item(), self.i)
                        
                        if self.is_grad_loss==True:
                            self.writer.add_scalar('mkup-grad/lips', (g_A_lip_loss_grad + g_B_lip_loss_grad).item(), self.i)
                            print('g_A_lip_loss_grad + g_B_lip_loss_grad', (g_A_lip_loss_grad + g_B_lip_loss_grad).item())

                        if self.is_glcm_loss==True:
                            self.writer.add_scalar('mkup-glcm/lips', (g_A_lip_glcm_loss + g_B_lip_glcm_loss).item(), self.i)
                        
                        if self.is_glcm_loss_with_count==True:
                            self.writer.add_scalar('mkup-glcm-with-count/lips', (g_A_lip_glcm_with_count_loss + g_B_lip_glcm_with_count_loss).item(), self.i)
                        
                        if self.is_glcm_to_org_loss==True:
                            self.writer.add_scalar('mkup-glcm-to-org/lips', (g_A_lip_glcm_to_org_loss + g_B_lip_glcm_to_org_loss).item(), self.i)

                    if self.skin:
                        if self.is_added_lumiL1==True:
                            self.writer.add_scalar('mkup-lumi/skin', (g_A_skin_loss_lumi + g_B_skin_loss_lumi).item(), self.i)
                        if self.is_added_lumi_L1_to_his==True:
                            self.writer.add_scalar('mkup-his-lumi/skin', (g_A_skin_loss_his + g_B_skin_loss_his).item(), self.i)
                        if self.is_luminance_matching_to_fake==True:
                            self.writer.add_scalar('mkup-lumi-his-match/skin', (g_A_skin_loss_lumi_his+g_B_skin_loss_lumi_his).item(), self.i)
                        if self.is_HS_matching==True:
                            self.writer.add_scalar('mkup-HS-his-match/skin', (g_A_skin_loss_HS_his + g_B_skin_loss_HS_his).item(), self.i)

                        # 8/17
                        if self.is_HLS_match_all==True:
                            self.writer.add_scalar('mkup-is_HLS_match_all/skin', (g_A_skin_loss_HLS_his + g_B_skin_loss_HLS_his).item(), self.i)
                        if self.is_HS_match_skin_HLS_match_other==True:
                            self.writer.add_scalar('mkup-is_HS_match_skin_HLS_match_other/skin', (g_A_skin_loss_HS_his + g_B_skin_loss_HS_his).item(), self.i)
                        if self.is_AB_match_all==True:
                            self.writer.add_scalar('mkup-is_AB_match_all/skin', (g_A_skin_loss_AB_his + g_B_skin_loss_AB_his).item(), self.i)
                        if self.is_AB_match_skin_Lab_match_other==True:
                            self.writer.add_scalar('mkup-is_AB_match_skin_Lab_match_other/skin', (g_A_skin_loss_AB_his + g_B_skin_loss_AB_his).item(), self.i)

                        if self.is_HS_matching==False and self.is_HLS_match_all==False and \
                            self.is_HS_match_skin_HLS_match_other==False and self.is_AB_match_all==False and self.is_AB_match_skin_Lab_match_other==False:
                            self.writer.add_scalar('mkup-hist/skin', (g_A_skin_loss_his+g_B_skin_loss_his).item(), self.i)
                        
                        if self.is_grad_loss==True:
                            self.writer.add_scalar('mkup-grad/skin', (g_A_skin_loss_grad + g_B_skin_loss_grad).item(), self.i)
                            print('g_A_skin_loss_grad + g_B_skin_loss_grad', (g_A_skin_loss_grad + g_B_skin_loss_grad).item())
                        
                        if self.is_glcm_loss==True:
                            self.writer.add_scalar('mkup-glcm/skin', (g_A_skin_glcm_loss + g_B_skin_glcm_loss).item(), self.i)
                        
                        if self.is_glcm_loss_with_count==True:
                            self.writer.add_scalar('mkup-glcm-with-count/skin', (g_A_skin_glcm_with_count_loss + g_B_skin_glcm_with_count_loss).item(), self.i)
                        
                        if self.is_glcm_to_org_loss==True:
                            self.writer.add_scalar('mkup-glcm-to-org/skin', (g_A_skin_glcm_to_org_loss + g_B_skin_glcm_to_org_loss).item(), self.i)
                    #-- Images
                    self.writer.add_images('Original/org_A', de_norm(org_A), self.i)
                    self.writer.add_images('Original/ref_B', de_norm(ref_B), self.i)
                    self.writer.add_images('Fake/fake_A', de_norm(fake_A), self.i)
                    self.writer.add_images('Fake/fake_B', de_norm(fake_B), self.i)
                    self.writer.add_images('Rec/rec_A', de_norm(rec_A), self.i)
                    self.writer.add_images('Rec/rec_B', de_norm(rec_B), self.i)
                
                # Save model checkpoints
                if (self.i + 1) % self.snapshot_step == 0:
                    self.save_models()

            # Decay learning rate
            if (self.e+1) > (self.num_epochs - self.num_epochs_decay):
                g_lr -= (self.g_lr / float(self.num_epochs_decay))
                d_lr -= (self.d_lr / float(self.num_epochs_decay))
                self.update_lr(g_lr, d_lr)
                print('Decay learning rate to g_lr: {}, d_lr:{}.'.format(g_lr, d_lr))

    def vis_train(self, img_train_list):
        # saving training results
        mode = "train_vis"
        img_train_list = torch.cat(img_train_list, dim=3)
        result_path_train = os.path.join(self.result_path, mode)
        if not os.path.exists(result_path_train):
            os.mkdir(result_path_train)
        save_path = os.path.join(result_path_train, '{}_{}_fake.jpg'.format(self.e, self.i))
        save_image(self.de_norm(img_train_list.data), save_path, normalize=True)