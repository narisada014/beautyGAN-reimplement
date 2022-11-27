import numpy as np
from skimage.feature import graycomatrix
from typing import List
from torch.autograd import Variable
import torch

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    if not requires_grad:
        return Variable(x, requires_grad=requires_grad)
    else:
        return Variable(x)

def one_channel_glcm_loss(
    masked_org_img_l_channel: np.ndarray, masked_fake_img_l_channel: np.ndarray, masked_org_pixel_count: float, masked_fake_pixel_count: float, gray_levels: int, distance: int
) -> torch.Tensor:
    masked_org_glcm = eight_way_glcm_wrapper(masked_org_img_l_channel, gray_levels, distance)
    masked_fake_glcm = eight_way_glcm_wrapper(masked_fake_img_l_channel, gray_levels, distance)
    criterionL1 = torch.nn.L1Loss()
    loss: torch.Tensor = torch.Tensor(0)
    for idx, org in enumerate(masked_org_glcm):
        loss = criterionL1(
            to_var(torch.from_numpy((org).astype(np.float32)).clone(), requires_grad=False), 
            to_var(torch.from_numpy((masked_fake_glcm[idx]).astype(np.float32)).clone(), requires_grad=False),
        )
    return loss

def one_channel_glcm_loss_with_count(
    masked_org_img_l_channel: np.ndarray, masked_fake_img_l_channel: np.ndarray, masked_org_pixel_count: float, masked_fake_pixel_count: float, gray_levels: int, distance: int
) -> torch.Tensor:
    masked_org_glcm = eight_way_glcm_wrapper(masked_org_img_l_channel, gray_levels, distance)
    masked_fake_glcm = eight_way_glcm_wrapper(masked_fake_img_l_channel, gray_levels, distance)
    criterionL1 = torch.nn.L1Loss()
    loss: torch.Tensor = torch.Tensor(0)
    for idx, org in enumerate(masked_org_glcm):
        loss = criterionL1(
            to_var(torch.from_numpy((org/masked_org_pixel_count).astype(np.float32)).clone(), requires_grad=False), 
            to_var(torch.from_numpy((masked_fake_glcm[idx]/masked_fake_pixel_count).astype(np.float32)).clone(), requires_grad=False),
        )
    # print(loss * 100000)
    return loss * 100000


# 白黒画像に対してのwrapper
def eight_way_glcm_wrapper(one_channel_img: np.ndarray, gray_levels: int, distance: int) -> List[np.ndarray]:
    # 右0度
    glcm_0 = np.squeeze(graycomatrix(one_channel_img, levels=gray_levels+1, distances=[distance], angles=[0]))[1:,1:]
    # 斜め45度
    glcm_45 = np.squeeze(graycomatrix(one_channel_img, levels=gray_levels+1, distances=[distance], angles=[np.pi/4]))[1:,1:]
    # 90度
    glcm_90 = np.squeeze(graycomatrix(one_channel_img, levels=gray_levels+1, distances=[distance], angles=[np.pi/2]))[1:,1:]
    # 135度
    glcm_135 = np.squeeze(graycomatrix(one_channel_img, levels=gray_levels+1, distances=[distance], angles=[3*(np.pi/4)]))[1:,1:]
    # 180
    glcm_180 = np.squeeze(graycomatrix(one_channel_img, levels=gray_levels+1, distances=[distance], angles=[np.pi]))[1:,1:]
    # 225
    glcm_225 = np.squeeze(graycomatrix(one_channel_img, levels=gray_levels+1, distances=[distance], angles=[5*(np.pi/4)]))[1:,1:]
    # 270
    glcm_270 = np.squeeze(graycomatrix(one_channel_img, levels=gray_levels+1, distances=[distance], angles=[3*(np.pi/2)]))[1:,1:]
    # 315
    glcm_315 = np.squeeze(graycomatrix(one_channel_img, levels=gray_levels+1, distances=[distance], angles=[7*(np.pi/4)]))[1:,1:]

    return [glcm_0, glcm_45, glcm_90, glcm_135, glcm_180, glcm_225, glcm_270, glcm_315]

# RGB画像に対してのwrapper
def rgb_glcm_wrapper(arr):
    pass