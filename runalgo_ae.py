##################################################
# RUN ML MODEL ON SPARKES
# Author: Suk Yee Yong
##################################################


import pypsrfits
import evalmetricsim
import autoencoder

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from pathlib import Path
import time
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision


FNAME       = 'simplepulse_multi_01.sf'
DIR         = '/datasets/work/mlaifsp-sparkes/work/sparkesX/multi/simplepulse/'
OUTPUT_DIR  = DIR + FNAME[:-3] + '/outputs/'

seed = 42
torch.manual_seed(seed)

batch_size = 16
epochs = 199
learning_rate = 1e-6
input_shape = [96, 4096]


def main(fname):
    Path(OUTPUT_DIR).mkdir(parents = True, exist_ok=True)
    Path(OUTPUT_DIR + 'subints/').mkdir(parents = True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = autoencoder.ConvAutoencoder(input_shape=input_shape).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Load file
    psrfile = pypsrfits.PSRFITS(DIR + fname)
    # Get sim labels
    _, y_sim, sim, tframe = evalmetricsim.read_simlabel(DIR + fname, datadir='')
    print(y_sim)
    
    fout = open(OUTPUT_DIR + FNAME[:-3] + '.txt', 'w')
    fout.write('frame,loss,filename\n')
   
    #####
    # RUN ALGORITHM
    # Input bdata: 2d array with dimension [nfrequency, ntime]
    #####
    all_scores = []
    all_times = []
    for epoch in range(epochs):
        total_loss = 0
        for nrow in range(psrfile.nrows_file):
            img = psrfile.getData(nrow, None, get_ft=True, squeeze=True, transpose=True)[0]
            outputs = model(img)
            loss = criterion(outputs, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f'Epoch: {epoch} | Loss {loss:.4f} | Total loss {total_loss:.4f}')
        
        # Write to output file
        fout.write(f"{nrow},{loss},{FNAME[:-3]}\n")
    
    torch.save(model.state_dict(), './convautoencoder.pth')

    fout.close()
    

if __name__ == '__main__':
    main(FNAME)

