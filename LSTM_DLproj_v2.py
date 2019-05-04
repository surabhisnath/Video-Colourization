import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import PIL
from PIL import Image
from torchvision import transforms
import pickle
torch.manual_seed(1)
frames = os.listdir("../original/")
frames.sort()
imgs = []
scale_percent = 10 # percent of original size
dim = (64, 36) #width, height
# resize image
 
# print('Resized Dimensions : ',resized.shape)
 
# cv2.imshow("Resized image", resized)￼
# train_transforms = transforms.Compose([ 
#     transforms.ToTensor(), 
#     transforms.Normalize(mean=[ 0.406], std=[0.225])])
# conv = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

for i in frames:
	img = cv2.imread("../original/" + i)
	# print(img)
	img = cv2.resize(img, dim)
	# cv2.imshow("img", img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	img = np.reshape(img, (img.shape[0]*img.shape[1]*img.shape[2]))
	# print(img.shape)
	imgs.append(img/255)	
	# break


imgs = np.array(imgs)

var = 6912

# class LSTM(nn.Module):

#     def __init__(self, input_dim, hidden_dim, batch_size, output_dim=var, num_layers=2):
#         super(LSTM, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.batch_size = batch_size
#         self.num_layers = num_layers

#         # Define the LSTM layer
#         self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

#         # Define the output layer
#         self.linear = nn.Linear(self.hidden_dim, output_dim)

#     def init_hidden(self):
#         # This is what we'll initialise our hidden state as
#         return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
#                 torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

#     def forward(self, input):
#         # Forward pass through LSTM layer
#         # shape of lstm_out: [input_size, batch_size, hidden_dim]
#         # shape of self.hidden: (a, b), where a and b both 
#         # have shape (num_layers, batch_size, hidden_dim).
#         lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
        
#         # Only take the output from the final timetep
#         # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
#         y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
#         return y_pred.view(-1)

class Sequence(nn.Module):
    def __init__(self):
        super(Sequence,self).__init__()
        self.lstm_enc = nn.LSTM(var,hidden_size=var)
        self.fc_enc = nn.Linear(var,1)
        self.lstm_dec = nn.LSTM(1,var)
        self.fc_dec = nn.Linear(var,var)

    def forward(self,input,input_reverse):
        outputs = []
        o_t_enc = Variable(torch.zeros(input.size(0),var).cuda(), requires_grad=False)
        h_t_enc = Variable(torch.zeros(input.size(0),var).cuda(), requires_grad=False)

        for i,input_t in enumerate(input.chunk(input.size(1),dim=1)):
            o_t_enc,h_t_enc = self.lstm_enc(input_t,(o_t_enc,h_t_enc))
            output = self.fc_enc(h_t_enc)

        outputs += [output]

        for i, input_t in enumerate(input_reverse.chunk(input_reverse.size(1),dim=1)):
            for i in range( input_reverse.size(1)-1):
                o_t_dec,h_t_dec = self.lstm_dec(input_t,(o_t_enc,h_t_enc))
                output = self.fc_dec(h_t_dec)
                outputs += [output]

        outputs = torch.stack(outputs,1).squeeze(2)

        return outputs




model = Sequence()
# model = LSTM(var, 1000, 1, var, 2)

learning_rate = 0.0001

loss_fn = torch.nn.L1Loss()
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

#####################
# Train model
#####################
num_epochs = 10
X_train = imgs[:-1,:]
y_train = imgs[1:,:]
X_test = imgs[len(imgs)-1,:]

y_train = torch.from_numpy(y_train).type(torch.Tensor)
#y_train = torch.from_numpy(y_train).type(torch.Tensor)
y_train = y_train.view([len(y_train), 1, -1])
#y_test = torch.from_numpy(y_test).type(torch.Tensor).view(-1)

X_train = torch.from_numpy(X_train).type(torch.Tensor)
# X_train = train_transforms(X_train)
X_train = X_train.view([len(X_train), 1, -1])

X_test = torch.from_numpy(X_test).type(torch.Tensor)
X_test = X_test.view([len(X_test), 1, -1])

hist = np.zeros(num_epochs)

for t in range(num_epochs):
	# Clear stored gradient
	model.zero_grad()

	# Initialise hidden state
	# Don't do this if you want your LSTM to be stateful
	# model.hidden = model.init_hidden()
	output = model.forward(data)
	# Forward pass
	loss = 0
	# print(X_train.size())
	for o in range(len(X_train)):

	    # print("size ", X_train[o].size())
	    # im = conv(X_train[o])
	    # print("new size = ", im.shape)
	    y_pred = model(X_train[o])

	    # print(y_pred.size())

	    loss += loss_fn(y_pred, y_train[o].squeeze(0))
	    
	# if t % 100 == 0:	
	print("Epoch ", t, "MSE: ", loss.item())

	hist[t] = loss.item()

	# Zero out gradient, else they will accumulate between epochs
	optimiser.zero_grad()

	# Backward pass
	loss.backward()

	# Update parameters
	optimiser.step()

	pred_img = model(X_train[-50]).detach().numpy()
	pred_img = pred_img*255
	pred_img = np.resize(pred_img, (72, 128, 3))
	# cv2.imshow("Generated_Image.jpg", pred_img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	# pred_img = model(X_test[0]).detach().numpy()
	# pred_img = np.resize(pred_img, (72, 128, 3))
	# cv2.imshow("Generated_Image.jpg", pred_img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

#torch.save(model.state_dict(), open("model_10k.pth", 'wb'))

#pickle.dump(hist, open('LSTM_Loss_10k.pkl', 'wb'))
pred_img = model(X_train[-1]).detach().numpy()
pred_img = np.resize(pred_img, (72, 128, 3))
cv2.imwrite("Generated_Image.jpg", pred_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
