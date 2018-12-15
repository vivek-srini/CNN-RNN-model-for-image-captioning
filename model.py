import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.hidden_size=hidden_size
        self.word_embed=nn.Embedding(vocab_size,embed_size)
        self.LSTM=nn.LSTM(input_size=embed_size,hidden_size=hidden_size,num_layers=2,bias=True,batch_first=True,dropout=0)
        self.linear = nn.Linear(hidden_size, vocab_size)  
        self.softmax=nn.Softmax(dim=2)
    
    def forward(self, features, captions):
        batch_size=features.shape[0]
        captions=self.word_embed(captions[:,:-1])
        inputs=torch.cat((features.unsqueeze(1),captions),1)
        lstm_output,_=self.LSTM(inputs,None)
        output=self.linear(lstm_output)
        return output
                                 

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        tokens = []
        for i in range(max_len):
            lstm_output, states = self.LSTM(inputs, states)
            out = self.linear(lstm_output.squeeze(1))
            argmax = out.max(1)
            token = argmax[1].item()
            tokens.append(token)
            inputs = self.word_embed(argmax[1].long()).unsqueeze(1)
            if token == 1: 
                break
        return tokens