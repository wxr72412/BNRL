import numpy as np
import torch
from torch import nn
from torch.autograd import Function, Variable
import torch.nn.functional as F


class NearestEmbedFunc(Function):
    """
    Input:
    ------
    x - (batch_size, emb_dim, *)
        Last dimensions may be arbitrary
    emb - (emb_dim, num_emb)
    """
    @staticmethod
    def forward(ctx, input, emb):
        # print(ctx)
        # print(input) # torch.Size([n, d, l]) torch.Size([1, 2, 4])
        # print(emb) # torch.Size([d, k]) torch.Size([2, 3])
        # print(input.shape) # torch.Size([n, d, l]) torch.Size([1, 2, 4])
        # print(emb.shape) # torch.Size([d, k]) torch.Size([2, 3])
        # exit(0)
        if input.size(1) != emb.size(0):
            raise RuntimeError('invalid argument: input.size(1) ({}) must be equal to emb.size(0) ({})'.
                               format(input.size(1), emb.size(0)))
        # save sizes for backward
        ctx.batch_size = input.size(0) # n
        ctx.num_latents = int(np.prod(np.array(input.size()[2:]))) # l
        ctx.emb_dim = emb.size(0) # d
        ctx.num_emb = emb.size(1) # k
        ctx.input_type = type(input) # <class 'torch.Tensor'>
        ctx.dims = list(range(len(input.size()))) # [0, 1, 2]
        # print(ctx.batch_size)
        # print(ctx.num_latents)
        # print(ctx.emb_dim)
        # print(ctx.num_emb)
        # print(ctx.input_type)
        # print(input.size()) # torch.Size([1, 1, 8]) len(input.size() = 3
        # print(ctx.dims)
        # exit(0)


        # expand to be broadcast-able
        x_expanded = input.unsqueeze(-1)
        # print(x_expanded.shape) # torch.Size([1, 2, 4, 1])
        # exit(0)
        num_arbitrary_dims = len(ctx.dims) - 2
        # print(num_arbitrary_dims) # 1
        # exit(0)
        if num_arbitrary_dims:
            emb_expanded = emb.view(
                emb.shape[0], *([1] * num_arbitrary_dims), emb.shape[1])
        else:
            emb_expanded = emb
        # print(emb_expanded) # torch.Size([2, 1, 3])
        # print(emb_expanded.shape) # torch.Size([2, 1, 3])
        # exit(0)

        # find nearest neighbors
        dist = torch.norm(x_expanded - emb_expanded, 2, 1)
        # input (Tensor) – 输入张量
        # p (float) = 2 – 范数计算中的幂指数值
        # dim (int) = 1 – 缩减的维度
        # print(x_expanded - emb_expanded) # torch.Size([1, 2, 4, 3])
        # print(dist) # torch.Size([1, 4, 3])
        # print((x_expanded - emb_expanded).shape) # torch.Size([1, 2, 4, 3])
        # print(dist.shape) # torch.Size([1, 4, 3])
        # exit(0)

        _, argmin = dist.min(-1)
        # print(argmin) # tensor([[0, 2, 2, 2]], device='cuda:0')
        # print(argmin.shape) # torch.Size([1, 4])
        # exit(0)

        shifted_shape = [input.shape[0], *
                         list(input.shape[2:]), input.shape[1]]
        # print(shifted_shape) # [1, 4, 2]
        result = emb.t().index_select(0, argmin.view(-1)).view(shifted_shape).permute(0, ctx.dims[-1], *ctx.dims[1:-1])
        # input
        # tensor([[[ 1.0000, -0.1023, -0.1154, -0.0265],
        #          [ 0.3000,  0.0527,  0.0400, -0.0357]]], device='cuda:0',
        # emb
        #        grad_fn=<AsStridedBackward>)
        # tensor([[0.7027, 0.0979, 0.4076],
        #          [0.2948, 0.8591, 0.1427]], device='cuda:0')
        # print(result)
        # tensor([[[0.7027, 0.4076, 0.4076, 0.4076],
        #          [0.2948, 0.1427, 0.1427, 0.1427]]], device='cuda:0')
        # print(result.shape) # torch.Size([1, 2, 4])
        # exit(0)

        ctx.save_for_backward(argmin)
        return result.contiguous(), argmin

    @staticmethod
    def backward(ctx, grad_output, argmin=None):
        # print('grad_output')
        # print(grad_output)
        # tensor([[[-0.0149,  0.0255,  0.0261,  0.0217],
        #          [-0.0003,  0.0045,  0.0051,  0.0089]]], device='cuda:0')
        # print(argmin)
        # tensor([[0, 0, 0, 0]], device='cuda:0')

        grad_input = grad_emb = None
        # print(ctx.needs_input_grad) # (False, True)
        if ctx.needs_input_grad[0]:
            grad_input = grad_output
        # print(grad_input) # None

        if ctx.needs_input_grad[1]:
            argmin, = ctx.saved_variables
            # print(argmin) # tensor([[0, 2, 2, 2]], device='cuda:0')
            latent_indices = torch.arange(ctx.num_emb).type_as(argmin)
            # print(latent_indices) # tensor([0, 1, 2], device='cuda:0') # ctx.num_emb = 3
            idx_choices = (argmin.view(-1, 1) == latent_indices.view(1, -1)).type_as(grad_output.data)
            # print(argmin.view(-1, 1))
            # print(latent_indices.view(1, -1))
            # print(idx_choices)
            # print(idx_choices.shape) # torch.Size([4, 3])
            # exit(0)
            # tensor([[1., 0., 0.],
            #         [0., 0., 1.],
            #         [0., 0., 1.],
            #         [0., 0., 1.]], device='cuda:0')

            n_idx_choice = idx_choices.sum(0)
            # print(n_idx_choice) # tensor([1., 0., 3.], device='cuda:0')
            n_idx_choice[n_idx_choice == 0] = 1 # 备注： 让下一步被除数不为0
            # print(n_idx_choice) # tensor([1., 1., 3.], device='cuda:0')
            idx_avg_choices = idx_choices / n_idx_choice
            # print(idx_avg_choices)
            # tensor([[1.0000, 0.0000, 0.0000],
            #         [0.0000, 0.0000, 0.3333],
            #         [0.0000, 0.0000, 0.3333],
            #         [0.0000, 0.0000, 0.3333]], device='cuda:0')
            grad_output = grad_output.permute(0, *ctx.dims[2:], 1).contiguous()
            # print(grad_output)
            # tensor([[[-0.0149, -0.0003],
            #          [ 0.0255,  0.0045],
            #          [ 0.0261,  0.0051],
            #          [ 0.0217,  0.0089]]], device='cuda:0')
            grad_output = grad_output.view(
                ctx.batch_size * ctx.num_latents, ctx.emb_dim)
            # print(grad_output)
            # tensor([[-0.0149, -0.0003],
            #         [ 0.0255,  0.0045],
            #         [ 0.0261,  0.0051],
            #         [ 0.0217,  0.0089]], device='cuda:0')
            grad_emb = torch.sum(grad_output.data.view(-1, ctx.emb_dim, 1) * idx_avg_choices.view(-1, 1, ctx.num_emb), 0)
            # print(grad_output.data.view(-1, ctx.emb_dim, 1))
            # print(grad_output.data.view(-1, ctx.emb_dim, 1).shape) # torch.Size([4, 2, 1])
            # tensor([[[-0.0149],
            #          [-0.0003]],
            #
            #         [[ 0.0255],
            #          [ 0.0045]],
            #
            #         [[ 0.0261],
            #          [ 0.0051]],
            #
            #         [[ 0.0217],
            #          [ 0.0089]]], device='cuda:0')
            # print(idx_avg_choices.view(-1, 1, ctx.num_emb))
            # print(idx_avg_choices.view(-1, 1, ctx.num_emb).shape) # torch.Size([4, 1, 3])
            # tensor([[[1.0000, 0.0000, 0.0000]],
            #
            #         [[0.0000, 0.0000, 0.3333]],
            #
            #         [[0.0000, 0.0000, 0.3333]],
            #
            #         [[0.0000, 0.0000, 0.3333]]], device='cuda:0')
            # print((grad_output.data.view(-1, ctx.emb_dim, 1) * idx_avg_choices.view(-1, 1, ctx.num_emb)))
            # print((grad_output.data.view(-1, ctx.emb_dim, 1) * idx_avg_choices.view(-1, 1, ctx.num_emb)).shape) # torch.Size([4, 2, 3])
            # tensor([[[-0.0149, -0.0000, -0.0000],
            #          [-0.0003, -0.0000, -0.0000]],
            #
            #         [[ 0.0000,  0.0000,  0.0085],
            #          [ 0.0000,  0.0000,  0.0015]],
            #
            #         [[ 0.0000,  0.0000,  0.0087],
            #          [ 0.0000,  0.0000,  0.0017]],
            #
            #         [[ 0.0000,  0.0000,  0.0072],
            #          [ 0.0000,  0.0000,  0.0030]]], device='cuda:0')
            # print(grad_emb)
            # print(grad_emb.shape) # torch.Size([2, 3])
            # tensor([[-0.0149,  0.0000,  0.0244],
            #         [-0.0003,  0.0000,  0.0062]], device='cuda:0')
        # exit(0)
        return grad_input, grad_emb, None, None


def nearest_embed(x, emb):
    # print(x)
    # print(emb)
    # exit(0)
    return NearestEmbedFunc().apply(x, emb)


class NearestEmbed(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim):
        super(NearestEmbed, self).__init__()
        # self.weight = nn.Parameter(torch.rand(embeddings_dim, num_embeddings))
        # # print(torch.rand(embeddings_dim, num_embeddings))
        # # print(self.weight)
        # # exit(0)

        init_weight = torch.ones(embeddings_dim, num_embeddings)
        # temp = 0.1 * num_embeddings
        # co = np.linspace(0, temp, num_embeddings)
        # co = np.linspace(-1*temp, temp, num_embeddings)
        co = np.linspace(-1, 1, num_embeddings)
        for i in range(num_embeddings):
            init_weight[:,i] = co[i]
        self.weight = nn.Parameter(init_weight)
        # print(self.weight)
        # exit(0)

    def forward(self, x, weight_sg=False):
        """Input:
        ---------
        x - (batch_size, emb_size, *)
        """
        return nearest_embed(x, self.weight.detach() if weight_sg else self.weight)


# adapted from https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py#L25
# that adapted from https://github.com/deepmind/sonnet


# class NearestEmbedEMA(nn.Module):
#     def __init__(self, n_emb, emb_dim, decay=0.99, eps=1e-5):
#         super(NearestEmbedEMA, self).__init__()
#         self.decay = decay
#         self.eps = eps
#         self.embeddings_dim = emb_dim
#         self.n_emb = n_emb
#         self.emb_dim = emb_dim
#         embed = torch.rand(emb_dim, n_emb)
#         self.register_buffer('weight', embed)
#         self.register_buffer('cluster_size', torch.zeros(n_emb))
#         self.register_buffer('embed_avg', embed.clone())
#
#     def forward(self, x):
#         """Input:
#         ---------
#         x - (batch_size, emb_size, *)
#         """
#
#         dims = list(range(len(x.size())))
#         x_expanded = x.unsqueeze(-1)
#         num_arbitrary_dims = len(dims) - 2
#         if num_arbitrary_dims:
#             emb_expanded = self.weight.view(
#                 self.emb_dim, *([1] * num_arbitrary_dims), self.n_emb)
#         else:
#             emb_expanded = self.weight
#
#         # find nearest neighbors
#         dist = torch.norm(x_expanded - emb_expanded, 2, 1)
#         _, argmin = dist.min(-1)
#         shifted_shape = [x.shape[0], *list(x.shape[2:]), x.shape[1]]
#         result = self.weight.t().index_select(
#             0, argmin.view(-1)).view(shifted_shape).permute(0, dims[-1], *dims[1:-1])
#
#         if self.training:
#             latent_indices = torch.arange(self.n_emb).type_as(argmin)
#             emb_onehot = (argmin.view(-1, 1) ==
#                           latent_indices.view(1, -1)).type_as(x.data)
#             n_idx_choice = emb_onehot.sum(0)
#             n_idx_choice[n_idx_choice == 0] = 1
#             flatten = x.permute(
#                 1, 0, *dims[-2:]).contiguous().view(x.shape[1], -1)
#
#             self.cluster_size.data.mul_(self.decay).add_(
#                 1 - self.decay, n_idx_choice
#             )
#             embed_sum = flatten @ emb_onehot
#             self.embed_avg.data.mul_(self.decay).add_(
#                 1 - self.decay, embed_sum)
#
#             n = self.cluster_size.sum()
#             cluster_size = (
#                 (self.cluster_size + self.eps) /
#                 (n + self.n_emb * self.eps) * n
#             )
#             embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
#             self.weight.data.copy_(embed_normalized)
#
#         return result, argmin
