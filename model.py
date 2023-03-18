import torch
import torch.nn as nn

class LinearProjection(nn.Module):

    def __init__(self, patch_vec_size, num_patches, latent_vec_dim, drop_rate):
        super().__init__()
        self.linear_proj = nn.Linear(patch_vec_size, latent_vec_dim) #p^cxD
        self.cls_token = nn.Parameter(torch.randn(1, latent_vec_dim)) #랜덤한 값으로 #학습해야해서 다음과 같이 parameter로 설정 
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, latent_vec_dim)) #patch수에 하나 추가# 마찬가지로 학습해야해서 parameter로 
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        batch_size = x.size(0)
        #linear projection : BxNxD
        #class 토큰은 1xD이기 때문에 repeat함수를 사용하여 batch크기만큼 만들어 붙여줌
        x = torch.cat([self.cls_token.repeat(batch_size, 1, 1), self.linear_proj(x)], dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        return x

class MultiheadedSelfAttention(nn.Module):
    def __init__(self, latent_vec_dim, num_heads, drop_rate):
        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_heads = num_heads
        self.latent_vec_dim = latent_vec_dim
        self.head_dim = int(latent_vec_dim / num_heads)
        #각각을 linear로 정의
        #논문은 다음의 방법으로 한번의 정의
        #self.query = nn.Linear(latent_vec_dim,3*latent_vec_dim)

        #원래 output은 dh인데 다음과 같이 kxdh로하여 멀티헤드를 한번에 처리 
        #이와 같은 방법으로 모든 헤드의 쿼리,키,밸류를 구함
        self.query = nn.Linear(latent_vec_dim, latent_vec_dim) 
        self.key = nn.Linear(latent_vec_dim, latent_vec_dim) 
        self.value = nn.Linear(latent_vec_dim, latent_vec_dim)
        self.scale = torch.sqrt(self.head_dim*torch.ones(1)).to(device) #gpu연산이 가능하도록
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        batch_size = x.size(0)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        #위에서 정의한 q,k,v를 나눔
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3) #latent_vec_dim => 헤드 개수, 헤드 dimension
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,3,1) # k.t transpose로 바꿈
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3)
        attention = torch.softmax(q @ k / self.scale, dim=-1) # @는 matrix 곱
        x = self.dropout(attention) @ v
        x = x.permute(0,2,1,3).reshape(batch_size, -1, self.latent_vec_dim) #concatenate를 시켜주기위해 reshape

        return x, attention

class TFencoderLayer(nn.Module):
    def __init__(self, latent_vec_dim, num_heads, mlp_hidden_dim, drop_rate):
        super().__init__()
        self.ln1 = nn.LayerNorm(latent_vec_dim)
        self.ln2 = nn.LayerNorm(latent_vec_dim)
        self.msa = MultiheadedSelfAttention(latent_vec_dim=latent_vec_dim, num_heads=num_heads, drop_rate=drop_rate)
        self.dropout = nn.Dropout(drop_rate)
        self.mlp = nn.Sequential(nn.Linear(latent_vec_dim, mlp_hidden_dim),
                                 nn.GELU(), nn.Dropout(drop_rate),
                                 nn.Linear(mlp_hidden_dim, latent_vec_dim),
                                 nn.Dropout(drop_rate))

    def forward(self, x):
        z = self.ln1(x)
        z, att = self.msa(z)
        z = self.dropout(z)
        x = x + z
        z = self.ln2(x)
        z = self.mlp(z)
        x = x + z

        return x, att

class VisionTransformer(nn.Module):
    def __init__(self, patch_vec_size, num_patches, latent_vec_dim, num_heads, mlp_hidden_dim, drop_rate, num_layers, num_classes):
        super().__init__()
        self.patchembedding = LinearProjection(patch_vec_size=patch_vec_size, num_patches=num_patches,
                                               latent_vec_dim=latent_vec_dim, drop_rate=drop_rate)
        #multi-head attention부분 for문을 사용하여 layer크기만큼 ModuleList에 추가하여 학습에 사용할 수 있도록 함
        self.transformer = nn.ModuleList([TFencoderLayer(latent_vec_dim=latent_vec_dim, num_heads=num_heads,
                                                         mlp_hidden_dim=mlp_hidden_dim, drop_rate=drop_rate)
                                          for _ in range(num_layers)])

        self.mlp_head = nn.Sequential(nn.LayerNorm(latent_vec_dim), nn.Linear(latent_vec_dim, num_classes)) #첫번째 벡터를 인풋으로 받고 layer_norm , 출력은 클래수 수만큼 설정하여 linear projection 진행

    def forward(self, x):
        att_list = []
        x = self.patchembedding(x)
        for layer in self.transformer:
            x, att = layer(x)
            att_list.append(att) #층마다 나오는 attetnion을 저장
        x = self.mlp_head(x[:,0]) #완료된 후 class token에 해당하는 부분 가져옴

        return x, att_list
