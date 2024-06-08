from functools import partial
from torch import nn
from model.models import SupervisedMaskedAutoencoderViT

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = SupervisedMaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_patch16_dec512d1b(**kwargs):
    model = SupervisedMaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=1, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16(**kwargs):
    model = SupervisedMaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder=False, **kwargs)
    return model

# set recommended archs
mae_vit16_base_d8 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit16_base_d1 = mae_vit_base_patch16_dec512d1b  # decoder: 512 dim, 1 blocks
mae_vit16_base = mae_vit_base_patch16  # no decoder
