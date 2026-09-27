import torch
import torch.nn as nn
import torch.nn.functional as F

class FFADD(nn.Module):
    """
    Fuses SITS and AER features by concatenation and projection.
    Input sizes:
        SITS  : [64,128,256] channels at 64/32/16 resolutions
        AER   : [128,256,512] channels at 64/32/16 resolutions
    """
    def __init__(self, sits_dims=[64, 128, 256], aer_dims=[128, 256, 512], use_norm=True):
        super(FFADD, self).__init__()

        # After concatenation, input channels = sits_dim + aer_dim
        self.fuse_layers = nn.ModuleList([
            nn.Conv2d(s + a, a, kernel_size=1, bias=not use_norm)
            for s, a in zip(sits_dims, aer_dims)
        ])

        self.norm_layers = nn.ModuleList([
            nn.BatchNorm2d(a) if use_norm else nn.Identity()
            for a in aer_dims
        ])

        self.activation = nn.ReLU(inplace=True)

    def forward(self, aer_feats, sits_feats):
        assert len(sits_feats) == len(aer_feats), \
            "SITS and AER feature list lengths must match."

        fused_feats = []

        for i, (s_feat, a_feat) in enumerate(zip(sits_feats, aer_feats)):

            # Resize SITS to match AER spatial size
            if s_feat.shape[2:] != a_feat.shape[2:]:
                s_feat = F.interpolate(
                    s_feat,
                    size=a_feat.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )

            # Concatenate along channel dimension
            fused = torch.cat([s_feat, a_feat], dim=1)

            # Project to AER channel size
            fused = self.fuse_layers[i](fused)
            fused = self.norm_layers[i](fused)
            fused = self.activation(fused)

            fused_feats.append(fused)

        return fused_feats
