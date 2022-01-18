import torch
import torch.nn as nn
from pytorchyolo import models
from torch.nn.modules import transformer
import torchsummary

class YOLO(nn.Module):
    #input: (batch, seq_len, channels, w, h)
    def __init__(self):
        super().__init__()
        self.bboxPredictor = models.load_model("./yolov3-custom.cfg")

    def forward(self, x):
        out = self.bboxPredictor(x)
        return out


if __name__ == "__main__":
    seq_len, batch_size, img_size = 47, 1, 512 
    test_img = torch.randn((batch_size,seq_len,1,img_size,img_size))
    # conv = Convnet(seq_len=seq_len)
    # out = conv(test_img)
    # print(out.shape) #(batch, seq_len, 4096)
    # pos_emb = PositionalEmbedding1D(seq_len=seq_len,dim=out.shape[2])
    # out = pos_emb(out)
    # print(out.shape)
    # encoder = TransformerEncoder(d_model=out.shape[2])
    # out = encoder(out)
    # print(out.shape) #(batch, seq_len, 4096)
    # out = out.reshape(batch_size, seq_len, int(out.shape[2]**(1/2)),  int(out.shape[2]**(1/2)))
    # decoder = TransposeConvnet(img_size=out.shape[2], seq_len=seq_len)
    # out = decoder(out)
    # print(out.shape) #(batch, seq_len, img_size**2)
    