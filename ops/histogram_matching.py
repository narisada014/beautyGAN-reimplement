import numpy as np
import torch
import copy

def cal_hist(image, channel_num):
    """
        cal cumulative hist for channel list
    """
    hists = []
    for i in range(0, channel_num):
        channel = image[i]
        # channel = image[i, :, :]
        channel = torch.from_numpy(channel)
        # hist, _ = np.histogram(channel, bins=256, range=(0,255))
        hist = torch.histc(channel, bins=256, min=0, max=256)
        hist = hist.numpy()
        # refHist=hist.view(256,1)
        sum = hist.sum()
        pdf = [v / sum for v in hist]
        for i in range(1, 256):
            pdf[i] = pdf[i - 1] + pdf[i]
        hists.append(pdf)
    return hists

def cal_hist_one_dim(image):
    tensor_lumi_img = torch.from_numpy(image)
    # hist = torch.histc(tensor_lumi_img, bins=256, min=0, max=256)
    # hist = hist.numpy()
    hist, _ = np.histogram(tensor_lumi_img, bins=256, range=(0,255))
    sum = hist.sum()
    # print('sum',sum)
    pdf = [v / sum for v in hist]
    for i in range(1, 256):
        pdf[i] = pdf[i - 1] + pdf[i]
    # print('pdf', pdf)
    return pdf

# histgram_matchingのアルゴ
def cal_trans(ref, adj):
    """
        calculate transfer function
        algorithm refering to wiki item: Histogram matching
    """
    table = list(range(0, 256))
    for i in list(range(1, 256)):
        for j in list(range(1, 256)):
            if ref[i] >= adj[j - 1] and ref[i] <= adj[j]:
                table[i] = j
                break
    table[255] = 255
    return table

# 入力は三次元の行列
def histogram_matching(dstImg, refImg, index, channel_num):
    """
        perform histogram matching
        dstImg is transformed to have the same the histogram with refImg's
        index[0], index[1]: the index of pixels that need to be transformed in dstImg
        index[2], index[3]: the index of pixels that to compute histogram in refImg
    """
    index = [x.cpu().numpy() for x in index]

    # Tensor > Numpy
    dstImg = dstImg.detach().cpu().numpy()
    refImg = refImg.detach().cpu().numpy()
    # image[channel, x座標, y座標]
    # fake -> orgの場合はインデックスは一つのペアでいいはず
    dst_align = [dstImg[i, index[0], index[1]] for i in range(0, channel_num)] # マスキングされて残った部分のXY座標のみを使って各チャネルの値を取得する
    ref_align = [refImg[i, index[2], index[3]] for i in range(0, channel_num)]
    hist_ref = cal_hist(ref_align, channel_num)
    hist_dst = cal_hist(dst_align, channel_num)
    tables = [cal_trans(hist_dst[i], hist_ref[i]) for i in range(0, channel_num)]

    mid = copy.deepcopy(dst_align)
    for i in range(0, channel_num):
        for k in range(0, len(index[0])):
            dst_align[i][k] = tables[i][int(mid[i][k])]

    for i in range(0, channel_num):
        dstImg[i, index[0], index[1]] = dst_align[i]

    dstImg = torch.FloatTensor(dstImg).cuda()
    return dstImg

# ここにHLS verのhis_matchを実装する
# dstImg=fakeHLS->refImg=orgHLS or dstImg=matched fakeHLS -> refImg=orgImg
# loss = self.criterionL1(input_masked, input_match) fakeとfakeのマッチング後を比較する
# 独自実装が必要なのは、マスキングした領域のみでヒストグラムマッチングを行わなければならないから。ライブラリ使うと画像全体でヒストグラムマッチングされてしまう

# 入力は二次元行列
# dstはfake画像 or fakeをヒストグラムマッチングした画像。refImgはオリジナル画像
# 対象のジオメトリが同じなのでマスクはdstもalignも同じ
def luminanse_histogram_matching(dstImg, refImg, index):
    dstImg = copy.deepcopy(dstImg)
    refImg = copy.deepcopy(refImg)
    index = [x.cpu().numpy() for x in index]

    dst_align = dstImg[index[0], index[1]]
    # print(dst_align)
    ref_align = refImg[index[0], index[1]]
    hist_ref = cal_hist_one_dim(ref_align)
    hist_dst = cal_hist_one_dim(dst_align)

    table = cal_trans(hist_dst, hist_ref)
    mid = copy.deepcopy(dst_align)
    for i in range(0, len(index[0])):
        dst_align[i] = table[int(mid[i])]
    dstImg[index[0], index[1]] = dst_align
    dstImg = torch.FloatTensor(dstImg).cuda()
    return dstImg

def one_axis_histogram_matching(dstImg, refImg, index):
    dstImg = copy.deepcopy(dstImg)
    refImg = copy.deepcopy(refImg)
    index = [x.cpu().numpy() for x in index]

    # image[channel, x座標, y座標]
    dst_align = dstImg[index[0], index[1]]
    ref_align = refImg[index[2], index[3]]
    hist_ref = cal_hist_one_dim(ref_align)
    hist_dst = cal_hist_one_dim(dst_align)
    table = cal_trans(hist_dst, hist_ref)

    mid = copy.deepcopy(dst_align)
    for i in range(0, len(index[0])):
        dst_align[i] = table[int(mid[i])]
    dstImg[index[0], index[1]] = dst_align
    dstImg = torch.FloatTensor(dstImg).cuda()
    return dstImg