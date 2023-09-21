import numpy as np
import cv2
import torch
import torch.nn.functional as F

class WBsRGB:
    def __init__(self, gamut_mapping=2, upgraded=0):
        if upgraded == 1:
            self.features = torch.from_numpy(np.load('nerfstudio/models/models/features+.npy')).to(torch.float32)
            self.mappingFuncs = torch.from_numpy(np.load('nerfstudio/models/models/mappingFuncs+.npy')).to(torch.float32)
            self.encoderWeights = torch.from_numpy(np.load('nerfstudio/models/models/encoderWeights+.npy')).to(torch.float32)
            self.encoderBias = torch.from_numpy(np.load('nerfstudio/models/models/encoderBias+.npy')).to(torch.float32)
            self.K = 75
        else:
            self.features = torch.from_numpy(np.load('nerfstudio/models/models/features.npy')).to(torch.float32)
            self.mappingFuncs = torch.from_numpy(np.load('nerfstudio/models/models/mappingFuncs.npy')).to(torch.float32)
            self.encoderWeights = torch.from_numpy(np.load('nerfstudio/models/models/encoderWeights.npy')).to(torch.float32)
            self.encoderBias = torch.from_numpy(np.load('nerfstudio/models/models/encoderBias.npy')).to(torch.float32)
            self.K = 25

        self.sigma = 0.25
        self.h = 60
        self.gamut_mapping = gamut_mapping

    def encode(self, hist):
        histR_reshaped = hist[:, :, 0].transpose(1, 0).reshape(1, -1, order="F")
        histG_reshaped = hist[:, :, 1].transpose(1, 0).reshape(1, -1, order="F")
        histB_reshaped = hist[:, :, 2].transpose(1, 0).reshape(1, -1, order="F")
        hist_reshaped = torch.cat((histR_reshaped, histG_reshaped, histB_reshaped), dim=1)
        feature = torch.mm(hist_reshaped - self.encoderBias.t(), self.encoderWeights)
        return feature

    def rgb_uv_hist(self, image):
        if image.dim() == 3:
            sz = image.shape
            if sz[0] * sz[1] > 202500:
                factor = torch.sqrt(torch.tensor(202500 / (sz[0] * sz[1]), dtype=torch.float32))
                newH = torch.floor(sz[0] * factor).int()
                newW = torch.floor(sz[1] * factor).int()
                image = F.interpolate(image.permute(2, 0, 1).unsqueeze(0), size=(newH, newW), mode='nearest').squeeze(0).permute(1, 2, 0)

            I_reshaped = image[(image > 0).all(dim=2)]
            eps = 6.4 / self.h
            hist = torch.zeros((self.h, self.h, 3), dtype=torch.float32)
            Iy = torch.norm(I_reshaped, dim=1)

            for i in range(3):
                r = [j for j in range(3) if j != i]
                Iu = torch.log(I_reshaped[:, i] / I_reshaped[:, r[1]])
                Iv = torch.log(I_reshaped[:, i] / I_reshaped[:, r[0]])
                hist[:, :, i], _, _ = np.histogram2d(Iu.cpu().numpy(), Iv.cpu().numpy(), bins=self.h, range=[[-3.2 - eps / 2, 3.2 - eps / 2], [-3.2 - eps / 2, 3.2 - eps / 2]], weights=Iy.cpu().numpy())
                norm_ = hist[:, :, i].sum()
                hist[:, :, i] = torch.sqrt(hist[:, :, i] / norm_)

            return hist
        else:
            raise ValueError("Input image must be a 3-dimensional tensor. ")


    def correctImage(self, image):
        image = image.flip(-1)
        image = im2double(image)
        feature = self.encode(self.rgb_uv_hist(image))
        D_sq = torch.einsum('ij, ij ->i', self.features, self.features).unsqueeze(1) + torch.einsum('ij, ij ->i', feature, feature) - 2 * torch.mm(self.features, feature.t())

        idH = torch.argsort(D_sq, dim=0)[:self.K]
        mappingFuncs = self.mappingFuncs.index_select(0, idH)
        dH = torch.sqrt(torch.gather(D_sq, 0, idH.unsqueeze(1)).squeeze(1))
        weightsH = torch.exp(-(dH ** 2) / (2 * self.sigma ** 2))
        weightsH = weightsH / torch.sum(weightsH)
        mapping_functions = torch.sum(weightsH.view(-1, 1).expand(-1, 33) * mappingFuncs, dim=0)
        mapping_functions = mapping_functions.reshape(11, 3, 33).transpose(2, 0).contiguous().view(33, 11, 3)
        image_corr = self.colorCorrection(image, mapping_functions)
        return image_corr

    def colorCorrection(self, image, mapping_functions):
        sz = image.shape
        I_reshaped = image.reshape(-1, 3, order='F')
        kernel_out = kernelP(I_reshaped)
        out = torch.mm(kernel_out, mapping_functions)
        if self.gamut_mapping == 1:
            out = normScaling(I_reshaped, out)
        elif self.gamut_mapping == 2:
            out = outOfGamutClipping(out)
        else:
            raise Exception('Wrong gamut_mapping value')
        out = out.reshape((sz[0], sz[1], sz[2]))
        out = out.flip(-1)
        return out

def normScaling(I, I_corr):
    norm_I_corr = torch.norm(I_corr, dim=1)
    inds = norm_I_corr != 0
    norm_I_corr = norm_I_corr[inds]
    norm_I = torch.norm(I[inds, :], dim=1)
    I_corr[inds, :] = I_corr[inds, :] / norm_I_corr.view(-1, 1) * norm_I.view(-1, 1)
    return I_corr

def outOfGamutClipping(I):
    I[I > 1] = 1
    I[I < 0] = 0
    return I

def im2double(im):
    if im.dim() == 2:  # Grayscale image
        return torch.tensor(im.float() / 255.0)
    elif im.dim() == 3:  # RGB image
        return im.permute(2, 0, 1).float() / 255.0
    else:
        raise ValueError("Unsupported image format. Expected 2 or 3 dimensions.")

