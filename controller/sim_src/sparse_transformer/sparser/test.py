import numpy as np
from sim_src.sparse_transformer.sparser.nn import hashing_function
from sim_src.sparse_transformer.tokenizer.model import tokenizer_base
from sim_src.sim_env.env import WiFiNet

from sim_src.util import *

from working_dir_path import get_controller_path

np.set_printoptions(precision=3)

tk_model = tokenizer_base()

path = get_controller_path()
path = os.path.join(path, "sim_alg/train_tokenizer/selected_nn/tokenizer_base.final.pt")

tk_model.load_model(path=path)
tk_model.eval()


hf_model = to_device(hashing_function(latent_dim=5,hash_dim=10))
optimizer = torch.optim.Adam(hf_model.parameters(), lr=1e-3)    

print(hf_model.mlp)
N_TRAINING_STEP = 10000
for i in range(N_TRAINING_STEP):
    e = WiFiNet(seed=i)
    b = e.get_sta_states()
    l , r = tk_model.get_output_np_batch(b)

    points = to_tensor(l)

    target_collision_matrix = to_tensor(e.get_CH_matrix())

    soft_code = hf_model(points)
    
    loss_tot, loss_dis, loss_cor, loss_bal = hashing_function.loss_function(
        soft_code, target_collision_matrix
    )

    loss_tot.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(to_numpy(soft_code[0]))
    print(to_numpy(soft_code[1]))
    print(torch.sum(torch.abs(torch.sign(soft_code[0])-torch.sign(soft_code[1]))).item()/2,  to_numpy(target_collision_matrix[0,1]))
    print(to_numpy(points[0]))
    print(to_numpy(target_collision_matrix[0]))
    print(f"step:{i:4d}, loss_tot:{loss_tot:.4f} ,loss_dis:{loss_dis:.4f}, loss_cor:{loss_cor:.4f}, loss_bal:{loss_bal:.4f}")
