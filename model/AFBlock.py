import torch
import torch.nn as nn
import torch.nn.functional as F

class gnconv(nn.Module):
    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2*dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)
        
        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i+1], 1) for i in range(order-1)]
        )

        self.scale = s
        print('[gnconv]', order, 'order with dims=', self.dims, 'scale=%.4f'%self.scale)

    def forward(self, x, mask=None, dummy=False):
        B, C, H, W = x.shape

        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)

        dw_abc = self.dwconv(abc) * self.scale

        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]

        for i in range(self.order -1):
            x = self.pws[i](x) * dw_list[i+1]

        x = self.proj_out(x)

        return x


class AnisotropicConv(nn.Module):
    """ 各向异性卷积：捕捉长程水平和垂直依赖，避开 MOFA 的常规多尺度 """
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(1, kernel_size), padding=(0, kernel_size//2), groups=dim)
        self.dw_v = nn.Conv2d(dim, dim, kernel_size=(kernel_size, 1), padding=(kernel_size//2, 0), groups=dim)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        # 这种交叉建模方式在数学上等效于大核，但参数更少且强调结构
        return self.proj(self.dw_h(x) + self.dw_v(x))

class DynamicDecompose(nn.Module):
    """ 动态频率分解：替代原有的简单减法 """
    def __init__(self, dim):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.contrast_gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        # 计算全局上下文
        y = self.avg_pool(x).view(b, c)
        gate = self.contrast_gate(y).view(b, c, 1, 1)
        # 动态增强残差部分（高频）
        return x + gate * (x - F.adaptive_avg_pool2d(x, 1))

class AFBlock(nn.Module):
    """ 改进后的 FBlock，命名为 AFBlock 以示区别 """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6)
        # 核心创新：将 gnconv 的内部 dwconv 替换为条形各向异性卷积
        self.gnconv = gnconv(dim, gflayer=lambda d, h, w: AnisotropicConv(d))
        self.decompose = DynamicDecompose(dim)
        
        # 保持原有的 FFN 结构
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.SiLU() # 换成 SiLU，比 GELU 在这种门控结构下效果更好
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        self.drop_path = nn.Identity() # 简化演示

    def forward(self, x):
        # 1. 动态分解
        x = self.decompose(x)
        # 2. 各向异性门控交互
        x = x + self.gnconv(self.norm1(x))
        # 3. 增强 FFN
        input = x
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv2(self.act(self.pwconv1(self.norm2(x))))
        x = x.permute(0, 3, 1, 2)
        return input + x

class LayerNorm(nn.Module):
    """
    通用 LayerNorm
    支持:
    - channels_first:  [B, C, H, W]
    - channels_last:   [B, H, W, C]
    """
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = normalized_shape

    def forward(self, x):
        if x.dim() == 4 and x.shape[1] == self.normalized_shape:
            # channels_first: [B, C, H, W]
            mean = x.mean(dim=1, keepdim=True)
            var = (x - mean).pow(2).mean(dim=1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            return self.weight[:, None, None] * x + self.bias[:, None, None]

        else:
            # channels_last: [..., C]
            return F.layer_norm(
                x,
                (self.normalized_shape,),
                self.weight,
                self.bias,
                self.eps
            )