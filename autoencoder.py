##################################################
# AUTOENCODER
# Author: Suk Yee Yong
##################################################


class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder"""
    def __init__(self, input_shape, list_filters=[4, 8, 16], kernel_size=5, stride=2):
        super(ConvAutoencoder, self).__init__()
        
        
        conv_kwargs = {'kernel_size': kernel_size, 'stride': stride, stride=stride)}
        
        # Encoder
        self.encoder = []
        encoder_filters = [input_shape[0], *list_filters]
        for in_c, out_c in zip(encoder_filters, encoder_filters[1:]):
            self.encoder.append(nn.Sequential(
                nn.Conv2d(in_c, out_c, **conv_kwargs),
                nn.ReLU(1e-2),
                nn.MaxPool2d(kernel_size=conv_kwargs['kernel_size'], stride=1),
            ))
        self.encoder = nn.Sequential(*self.encoder)
        # Decoder
        self.decoder = []
        decoder_filters = [*list_filters[::-1], input_shape[0]]
        for in_c, out_c in zip(decoder_filters[:-1], decoder_filters[1:-1]):
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, **conv_kwargs),
                nn.ReLU(1e-2),
            ))
        self.decoder.extend([nn.ConvTranspose2d(self.decoder[-1][-2].out_channels, decoder_filters[-1], output_padding=1, **conv_kwargs), nn.Sigmoid()])
        self.decoder = nn.Sequential(*self.decoder)
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


