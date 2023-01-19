import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyper-parameters
INUNITS = 177
h_dim = 32
z_dim = 16
num_epochs = 100
batch_size = 1
learning_rate = 1e-3
TESTNUM_TOTAL = 206



f1 = open("Coverage_Info/covMatrix.txt",'r')
f2 = open("Coverage_Info/error.txt",'r')
f3 = open("Coverage_Info/covMatrix_new.txt",'w')

first_ele = True
for data in f1.readlines():
        data = data.strip('\n')
        nums = data.split()
        if first_ele:
            nums = [float(x) for x in nums]
            matrix_x = np.array(nums)
            first_ele = False
        else:
            nums = [float(x) for x in nums]
            matrix_x = np.c_[matrix_x,nums]
f1.close()
    
first_ele = True
for data in f2.readlines():
        data = data.strip('\n')
        nums = data.split()
        if first_ele:
            nums = [float(x) for x in nums]
            matrix_y = np.array(nums)
            first_ele = False
        else:
            nums = [float(x) for x in nums]
            matrix_y = np.c_[matrix_y,nums]
f2.close()
    
    
matrix_x = matrix_x.transpose()
matrix_y = matrix_y.transpose()
inputs_pre = []
testcase_fail_num = 0
for testcase_num in range(len(matrix_y)):
        if matrix_y[testcase_num][0] == 1:
            inputs_pre.append(matrix_x[testcase_num])
            testcase_fail_num = testcase_fail_num + 1
TESTNUM = testcase_fail_num
inputs = torch.FloatTensor(inputs_pre)
labels = torch.FloatTensor(matrix_y)
inputs = inputs.to(device)
labels = labels.to(device)

""" 
build minimum suspicious set
"""

minimum_suspicious_set = Variable(torch.zeros(INUNITS))
for element_num in range(len(inputs[0])):
        flag = 0
        for item in inputs:
            if item[element_num] == 1:
                flag = flag +1
        if flag == testcase_fail_num:
            minimum_suspicious_set[element_num] = 1

# VAE model
class VAE(nn.Module):
    def __init__(self, INUNITS = INUNITS, h_dim = h_dim, z_dim = z_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(INUNITS, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim) 
        self.fc3 = nn.Linear(h_dim, z_dim) 
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, INUNITS)
        
   
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)
    
   
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

   
    def decode(self, z):
        h = F.relu(self.fc4(z))
        return F.sigmoid(self.fc5(h))
    
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var
    
    
model = VAE().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i, x in enumerate(inputs):
        x_reconst, mu, log_var = model(x)
        reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
        loss = reconst_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}" 
                   .format(epoch+1, num_epochs, i+1, len(x), reconst_loss.item(), kl_div.item()))



shape = ((TESTNUM_TOTAL-TESTNUM*2),INUNITS)
z_mean = torch.rand(shape)
rand_z = torch.randn(shape)+z_mean
rand_z = rand_z.to(device)

out, _, _ = model(rand_z)

out = out.cpu().detach()
out = out.numpy()
for item in out:
        for element_num in range(len(item)):
            if item[element_num] <= 0.5 and minimum_suspicious_set[element_num] == 0:
                f3.write('0')
            else:
                f3.write(str(round(item[element_num],2)))
            f3.write(' ')
        f3.write('\n')
f3.close()
