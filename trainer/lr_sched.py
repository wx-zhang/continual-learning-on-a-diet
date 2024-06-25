# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math


def adjust_learning_rate(optimizer, cur_iter, total_iter, blr, min_lr, warmup_iters):
    """Decay the learning rate with half-cycle cosine after warmup"""

    if blr < min_lr:
        min_lr = blr / 2

    if total_iter < warmup_iters:
        warmup_iters = total_iter * 0.1

    if cur_iter < warmup_iters:
        lr = blr * cur_iter / warmup_iters
    else:
        lr = min_lr + (blr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (cur_iter - warmup_iters) / (total_iter - warmup_iters)))

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr

    return lr
