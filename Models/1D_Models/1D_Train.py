#!/usr/bin/env python
# coding: utf-8


from fastai.vision.all import *
import numpy as np

# from fourier_1d import *
from fastai.vision.all import *
from scipy import signal
#from ipyexperiments import *
import warnings
from torch_audiomentations import Compose, Gain, PolarityInversion

import torch_audiomentations as tA

import julius
# import pytorch_warmup as warmup
import nnAudio
warnings.filterwarnings('ignore')


set_seed(42,reproducible=True)




#####################

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, downsample=None):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.downsample = downsample




    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        if self.downsample is not None:
#             out1 = self.maxpool(out)
#             print(out1.shape)
            out = gem_1d_res(out)
#             print(out2.shape)


            identity = self.downsample(x)

        out += identity
        # print(out.shape)

        return out




class ECGNet(nn.Module):

    def __init__(self, struct=[14, 16, 18, 20], in_channels=3, fixed_kernel_size=17, num_classes=1):
        super(ECGNet, self).__init__()
        self.struct = struct
        self.planes = 16
        self.parallel_conv = nn.ModuleList()
        for i, kernel_size in enumerate(struct):
            sep_conv = nn.Conv1d(in_channels=in_channels, out_channels=self.planes, kernel_size=kernel_size,
                               stride=1, padding=0, bias=False)
            self.parallel_conv.append(sep_conv)
        # self.parallel_conv.append(nn.Sequential(
        #     nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
        #     nn.Conv1d(in_channels=1, out_channels=self.planes, kernel_size=1,
        #                        stride=1, padding=0, bias=False)
        # ))

        self.bn1 = nn.BatchNorm1d(num_features=self.planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv1d(in_channels=self.planes, out_channels=self.planes, kernel_size=fixed_kernel_size,
                               stride=2, padding=2, bias=False)
        self.block = self._make_layer(kernel_size=fixed_kernel_size, stride=1, padding=8)
        self.bn2 = nn.BatchNorm1d(num_features=self.planes)
#         self.avgpool = nn.AvgPool1d(kernel_size=8, stride=8, padding=2)
        self.rnn = nn.LSTM(input_size=3, hidden_size=40, num_layers=1, bidirectional=True)
#         self.rnn = nn.GRU(input_size=3, hidden_size=40, num_layers=1, bidirectional=True)  
        self.fc = nn.Linear(in_features=128, out_features=num_classes)        
#         self.fc = nn.Linear(in_features=296, out_features=num_classes)
#         self.fc = nn.Linear(in_features=1576, out_features=num_classes)


    def _make_layer(self, kernel_size, stride, blocks=11, padding=0):
        layers = []
        downsample = None
        base_width = self.planes

        for i in range(blocks):
            if (i + 1) % 4 == 0:
                downsample = nn.Sequential(
                    nn.Conv1d(in_channels=self.planes, out_channels=self.planes + base_width, kernel_size=1,
                               stride=1, padding=0, bias=False),
                    nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
                )
                layers.append(ResBlock(in_channels=self.planes, out_channels=self.planes + base_width, kernel_size=kernel_size,
                                       stride=stride, padding=padding, downsample=downsample))
                self.planes += base_width
            elif (i + 1) % 2 == 0:
                downsample = nn.Sequential(
                    nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
                )
                layers.append(ResBlock(in_channels=self.planes, out_channels=self.planes, kernel_size=kernel_size,
                                       stride=stride, padding=padding, downsample=downsample))
            else:
                downsample = None
                layers.append(ResBlock(in_channels=self.planes, out_channels=self.planes, kernel_size=kernel_size,
                                       stride=stride, padding=padding, downsample=downsample))

        return nn.Sequential(*layers)



    def forward(self, x):
        out_sep = []

        for i in range(len(self.struct)):
            sep = self.parallel_conv[i](x)
            out_sep.append(sep)

        out = torch.cat(out_sep, dim=2)
        
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1(out)  # out => [b, 16, 9960]

        out1 = self.block(out)
#         print(out1.shape)
        out2 = self.block(out)
        out3 = torch.cat((out1,out2),-1)
#         print(out3.shape)
        out = self.bn2(out3)
        out = self.relu(out)
        out = gem_1d(out)
#         out = self.avgpool(out)  # out => [b, 64, 10]
        out = out.reshape(out.shape[0], -1)  # out => [b, 640]
#         print(out.shape)


#         rnn_out,rnn_h = self.rnn(x.permute(2, 0, 1))    
        rnn_out, (rnn_h, rnn_c) = self.rnn(x.permute(2, 0, 1))
        new_rnn_h = rnn_h[-1, :, :]  # rnn_h => [b, 40]
        new_rnn_c = rnn_c[-1, :, :]
#         print(new_rnn_c)
        new_out = torch.cat([out, new_rnn_h,new_rnn_c], dim=1)  # out => [b, 680]
#         print(new_out.shape)
        result = self.fc(new_out)  # out => [b, 20]

        # print(out.shape)

        return result

#####################
""
from sklearn.metrics import roc_auc_score
def roc_auc(preds,targ):
    return roc_auc_score(targ.cpu(), preds.sigmoid().cpu())


# from torch _audiomentations import Compose, Gain, PolarityInversion,Shift,AddColoredNoise,ShuffleChannels

class LoadG2Waves(torch.utils.data.Dataset):
    
    def __init__(self, df,noisy_df,data_dir, train = False,augmentations = None):
        self.df = df
        self.noisy = noisy_df
        self.augmentations = augmentations
        self.train = train
        self.input_path = Path(data_dir)
        bandpass_lowcut  = 25
        bandpass_highcut = 500
        bandpass_order   = 8
        
        lf, hf, order, sr = bandpass_lowcut, bandpass_highcut, bandpass_order, 2048
        self.sos = signal.butter(order, [lf, hf], btype="bandpass", output="sos", fs=sr)
        self.normalization = np.sqrt((hf - lf) / (sr / 2))
        
        
    def __len__(self):
        return len(self.df)
    
    def load_file(self, id_):
        path = self.input_path / id_[0] / id_[1] / id_[2] / f"{id_}.npy"
        waves = np.load(path)
        return waves
    
    def apply_bandpass(self, strain):
        return torch.from_numpy(signal.sosfiltfilt(self.sos, strain) / self.normalization)
        
    def __getitem__(self, index):
        row  = self.df.iloc[index]
        if self.noisy is not None:
            row2 = self.noisy.iloc[index]
        samples = self.load_file(row.id)

        samples*= signal.tukey(4096, 0.2)
        samples = self.apply_bandpass(samples) #/ 1e-19
        norm_by =[7.729773e-21,8.228142e-21, 8.750003e-21]
#         norm_by =[7.9081245e-21,7.7202e-21, 8.7608e-21]


#         norm_by  = 1e-19
        
        for i in range(samples.shape[0]):
            samples[i] = samples[i] / norm_by[i]
        samples = self.augmentations(samples.unsqueeze(0)).squeeze()

        samples = np.stack(samples, axis=0)

#         if np.random.random() < 0.2:
#             np.random.shuffle(samples)

        samples = quantize_data(samples,1)
        # print(samples)
        samples = torch.from_numpy(samples).float().view(3,4096)
        label = torch.tensor(row.target).float()

        if self.train:
            label2 = torch.tensor(row2.target).float()
            return samples, (label , label2)


        else:
            label2 = torch.tensor(row.target).float()
            return samples, label
# ====================================================
# Transforms
# ====================================================
def get_transforms(*, data):
    
    if data == 'train':
        return tA.Compose(
                transforms=[
#                      tA.ShuffleChannels(p=0.25,mode="per_example",p_mode="per_example", sample_rate= 2048),


#                      tA.AddColoredNoise(p=0.15,mode="per_channel",p_mode="per_channel", sample_rate=2048,max_snr_in_db = 15),
                     tA.Shift(p=0.75,mode="per_example",p_mode="per_example", sample_rate=2048,max_shift=0.025, min_shift=-0.025),
                ])

    elif data == 'valid':
        return tA.Compose([
        ])

def gem_1d(x, p=3, eps=1e-6):
    return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1),)).pow(1./p )

def gem_1d_res(x, p=3, eps=1e-6):
    return F.avg_pool1d(x.clamp(min=eps).pow(p), 2,2).pow(1./p)

def gem_2d(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


def gem_3d(x, p=3, eps=1e-6):
    return F.avg_pool3d(x.clamp(min=eps).pow(p), (x.size(-3), x.size(-2), x.size(-1))).pow(1./p)


_GEM_FN = {
    1: gem_1d, 2: gem_2d, 3: gem_3d
}

# class depthwise_separable_conv(nn.Module):
#     def __init__(self, nin, kernels_per_layer, nout): 
#         super(depthwise_separable_conv, self).__init__() 
#         self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=nin) 
#         self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1) 

#     def forward(self, x): 
#         out = self.depthwise(x) 
#         out = self.pointwise(out) 
#         return out

class depthwise_separable_conv1D(Module):
    def __init__(self, filters, kernel_size):
        super(TCSConv1d, self).__init__()
        self.depthwise = Conv1d(filters, filters, kernel_size=kernel_size, groups=filters)
        self.pointwise = Conv1d(filters, filters, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# class GeM(nn.Module):

#     def __init__(self, p=3, eps=1e-6, dim=2):
#         super().__init__()
#         self.p = nn.Parameter(torch.ones(1)*p)
#         self.eps = eps
#         self.dim = dim

#     def forward(self, x):
#         return _GEM_FN[self.dim](x, p=self.p, eps=self.eps)

class GeM(nn.Module):
    '''
    Code modified from the 2d code in
    https://amaarora.github.io/2020/08/30/gempool.html
    '''
    def __init__(self, kernel_size=2, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.kernel_size = kernel_size
        self.eps = eps
        self.stride
    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool1d(x.clamp(min=eps).pow(p), self.kernel_size).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'

class AdaptiveConcatPool1d(nn.Module):

    def forward(self, x):
        return torch.cat((F.adaptive_avg_pool1d(x, 1), F.adaptive_max_pool1d(x, 1)), dim=1)



""
def quantize_data(data, classes):
    mu_x = mu_law_encoding(data, classes)
    # bins = np.linspace(-1, 1, classes)
    # quantized = np.digitize(mu_x, bins) - 1
    return mu_x#quantized

def mu_law_encoding(data, mu):
    mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)
    return mu_x

def mu_law_expansion(data, mu):
    s = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
    return s

class CombinedLoss:
    "Dice and Focal combined"
    def __init__(self, axis=1, smooth=1., alpha=0.5):
        store_attr()
        self.focal_loss = BCEWithLogitsLossFlat()
        self.dice_loss =  BCEWithLogitsLossFlat()
        
    def __call__(self, pred, targ):
        if type(targ) is list:
            loss = self.focal_loss(pred, targ[0])*(1-self.alpha) + self.alpha * self.dice_loss(pred, targ[1])
        else:
            loss = self.focal_loss(pred, targ)#*(1-self.alpha) 


        return loss
    
    def decodes(self, x):    return x.argmax(dim=self.axis)
    def activation(self, x): return F.softmax(x, dim=self.axis)
    
    
modes = 16
width = 1

# myModel = FNO1d(modes,width)

        
def run():
    for fold_num in [2,3,4]:    
        print('*****************************************')
        print(f'Training Fold {fold_num}')
        print('*****************************************')

    #with IPyExperimentsPytorch() as exp:
        kernel_type = 'nis_1D_v1_noisytry_last2'
        OUTPUT_DIR = f'/home/models/{kernel_type}/'
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        data_dir = 'Data/train/'
        test_data_dir = 'Data/test/'
        csv_path = 'Data/folds-3.csv'
        noisy = True
        noisy_path = 'Data/oof_for1d.csv'
        batch_size = 126
        n_epochs = 16

        bandpass_lowcut  = 30
        bandpass_highcut = 400
        bandpass_order   = 8

        df = pd.read_csv(csv_path)[['id', 'target', 'fold']]
        df = df.sort_values(by='id').reset_index()
        df['is_valid'] = df.fold.apply(lambda x: x==fold_num)

        df2 = pd.read_csv(noisy_path)[['id', 'target']]
        df2 = df2.merge(df[['id','fold']],on='id')
        df2 = df2.sort_values(by='id').reset_index()


        df2['is_valid'] = df.fold.apply(lambda x: x==fold_num)       
#         df = df.head(10000)
#         df2 = df2.head(10000)


#         training_fold = df.query('is_valid==False').reset_index(drop=True, inplace=False)
#         train_ds = LoadG2Waves(training_fold,data_dir,augmentations = get_transforms(data='train'))

#         validation_fold = df.query('is_valid==True').reset_index(drop=True, inplace=False)
#         valid_ds = LoadG2Waves(validation_fold,data_dir,augmentations = get_transforms(data='valid'))
        training_fold = df.query('is_valid==False').reset_index(drop=True, inplace=False)
        training_fold_ns = df2.query('is_valid==False').reset_index(drop=True, inplace=False)


        train_ds = LoadG2Waves(training_fold,training_fold_ns,data_dir,train=True,augmentations = get_transforms(data='train'))

        validation_fold = df.query('is_valid==True').reset_index(drop=True, inplace=False)
        validation_fold_ns = df2.query('is_valid==True').reset_index(drop=True, inplace=False)


        valid_ds = LoadG2Waves(validation_fold,validation_fold_ns,data_dir,train=False,augmentations = get_transforms(data='valid'))

        print(f'- Training samples: {len(train_ds)}\n- Validation Samples : {len(valid_ds)}')

        bs = batch_size
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=bs, num_workers=8,pin_memory=False)
        valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=bs*4, num_workers=8,shuffle=False,pin_memory=False)

        dls = DataLoaders(train_dl, valid_dl)
#         model = myModel()#WaveNetModel()#myModel()
        model = ECGNet()#myModel(1) #FNO1d(modes,width) # # ,myModel()#WaveNetModel()#myModel()

        learn = Learner(dls, model, loss_func=CombinedLoss(), metrics=[roc_auc],cbs=[SaveModelCallback('roc_auc', every_epoch=True),CSVLogger(f'/home/models/{kernel_type}/{fold_num}logs.csv')])#.to_fp16()
#         learn.model = torch.nn.DataParallel(learn.model, device_ids=[0, 1])


        learn.fit_one_cycle(n_epochs, 1e-2, wd=1e-03)

        learn = learn.to_fp16()
        learn.save(f'/home/models/{kernel_type}/fold_{fold_num}')

        learn = learn.load(f'/home/models/{kernel_type}/fold_{fold_num}')
        learn.model.eval()
        test_df = pd.read_csv('Data/sample_submission.csv')
        test_ds = LoadG2Waves(test_df,None,test_data_dir,False,augmentations = get_transforms(data='valid'))
#         test_ds = LoadG2Waves(test_df,test_data_dir,augmentations = get_transforms(data='valid'))


        test_ds.input_path = Path('Data/test/')

        bs = batch_size
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size=bs, num_workers=8, shuffle=False)

        preds = []
        for xb, _ in progress_bar(test_dl):
            with torch.no_grad(): output = learn.model(xb.cuda())
            preds.append(torch.sigmoid(output.float()).squeeze().cpu())
        preds = torch.cat(preds)

        sample_df = pd.read_csv('Data/sample_submission.csv')
        sample_df['target'] = preds
        sample_df.to_csv(f'/home/models/{kernel_type}-fold_{fold_num}.csv', index=False)


if __name__ == '__main__':
    run()



