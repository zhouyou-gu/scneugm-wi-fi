import torch
import torch.nn as nn

class hashing_function(nn.Module):
    def __init__(self, latent_dim=5, hash_dim=20, single_layer = False, hidden_layer=5):
        super(hashing_function, self).__init__()
        self.latent_dim = latent_dim
        self.hash_dim = hash_dim
        
        self.hidden_layer = hidden_layer
        
        if not single_layer:
            l = nn.ModuleList([nn.Linear(latent_dim, hash_dim)])
            for _ in range(hidden_layer):
                l.extend([nn.GELU(),nn.Linear(hash_dim, hash_dim)])
            self.mlp = nn.Sequential(*l)
        else:
            self.mlp = nn.Linear(latent_dim, hash_dim, bias=True)
            torch.nn.init.xavier_uniform_(self.mlp.weight,gain=0.1)
            torch.nn.init.uniform_(self.mlp.bias)

        
    def forward(self, x):
        output = self.mlp(x)
        output = torch.tanh(output)
        return output
    
    def hard_code(self, x):
        output = self.mlp(x)
        return torch.sign(output)
    
    @staticmethod
    def loss_function(soft_code, target_collision_matrix, r_dis= 1., r_cor=0.1, r_bal=0.):
        num_soft_code = soft_code.size(0)
        num_hash_bits = soft_code.size(1)
        
        # maximize distance of soft_code
        ##simularity (simularity is the probability of two hash bits are the same. it works on soft codes in -1,+1)
        simularity = (torch.mm(soft_code,soft_code.t())+num_hash_bits)/2./num_hash_bits
        off_diagonal_mask = ~torch.eye(num_soft_code, dtype=bool, device=soft_code.device)
        # loss_dis = - target_collision_matrix[off_diagonal_mask] * torch.log(simularity[off_diagonal_mask]) - (1-target_collision_matrix[off_diagonal_mask]) * torch.log(1-simularity[off_diagonal_mask])
        loss_dis = torch.mean(torch.abs(target_collision_matrix[off_diagonal_mask]  - simularity[off_diagonal_mask])**2)
        
        # minimize correlation of different bits in each codeword
        cor_matrix = torch.mm(soft_code.t(),soft_code)/num_soft_code
        identity = torch.eye(num_hash_bits, device=soft_code.device)
        loss_cor = torch.mean(torch.abs(cor_matrix - identity)**2)
        
        # # minimize correlation of different codewords
        loss_bal = -torch.mean(torch.abs(soft_code))
        

        
        loss_tot =      r_dis * loss_dis \
                    +   r_cor * loss_cor \
                    +   r_bal * loss_bal
        return loss_tot, loss_dis, loss_cor, loss_bal
    
    
if __name__ == "__main__":
    from sim_src.util import *
    threshold = 30
    def generate_2d_points(num_points, min_val=-100, max_val=100):
        # Generate random 2D points within the specified range
        points = (max_val - min_val) * torch.rand(num_points, 2) + min_val
        return points

    def compute_collision_matrix(points, threshold=threshold):
        # Compute pairwise distances between points
        dist_matrix = torch.cdist(points, points, p=2)

        # Create a collision matrix where distance < threshold is considered a collision (1), else 0
        collision_matrix = (dist_matrix < threshold).float()
        return collision_matrix
    
    def generate_colliding_points(threshold=threshold):
        # Randomly generate a point in [-100, 100]^2
        point1 = (100 - (-100)) * torch.rand(2) - 100
        
        # Generate a second point that is within the threshold distance from the first point
        direction = torch.rand(2) - 0.5  # Random direction
        direction = direction / torch.norm(direction)  # Normalize the direction vector
        distance = torch.rand(1).item() * threshold  # Random distance within the threshold
        point2 = point1 + direction * distance
        
        return point1, point2
    
    def generate_non_colliding_points(point1 = None, threshold=threshold, min_distance=threshold+5):
        # Ensure that the two points are at least min_distance apart (greater than the threshold)
        if point1 is None:
            point1 = (100 - (-100)) * torch.rand(2) - 100

        # Generate the second point with a distance greater than the min_distance
        while True:
            point2 = (100 - (-100)) * torch.rand(2) - 100
            distance = torch.norm(point1 - point2)
            if distance <= min_distance and distance >= threshold:
                break

        return point1, point2
    
    model = to_device(hashing_function(latent_dim=2,hash_dim=20,single_layer=True))
    w_orignal = to_numpy(model.mlp.weight)
    b_orignal = to_numpy(model.mlp.bias)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)    
    batch_size = 2000
    n_step = 500
    
    # Generate 100 2D points
    points = to_device(generate_2d_points(batch_size))

    # Compute the collision matrix with a distance threshold of 20
    target_collision_matrix = to_device(compute_collision_matrix(points, threshold=20))
    for i in range(n_step):


        soft_code = model(points)
        print(soft_code[0],"soft")

        loss_tot, loss_dis, loss_cor, loss_bal = hashing_function.loss_function(
            soft_code, target_collision_matrix
        )

        
        loss_tot.backward()
        optimizer.step()
        optimizer.zero_grad()


        p1, p2 = generate_colliding_points()
        stacked_points = to_device(torch.vstack([p1, p2]))
        
        sc = model.hard_code(stacked_points)
        d_c = torch.cdist(sc,sc,p=1)[0,1].item()        
        v_c = torch.abs(sc).mean().item()
        pd_c = torch.cdist(stacked_points,stacked_points,p=2)[0,1].item()       



        p1, p2 = generate_non_colliding_points(p1)
        stacked_points = to_device(torch.vstack([p1, p2]))
        
        sc = model.hard_code(stacked_points)
        d_n = torch.cdist(sc,sc,p=1)[0,1].item()        
        v_n = torch.abs(sc).mean().item()
        pd_n = torch.cdist(stacked_points,stacked_points,p=2)[0,1].item()       
        
        
        print(f"step:{i:4d}, loss_tot:{loss_tot:.4f} ,loss_dis:{loss_dis:.4f}, loss_cor:{loss_cor:.4f}, loss_bal:{loss_bal:.4f}")
        print(f"d_c:{d_c:.4f}, d_n:{d_n:.4f}, v_c:{v_c:.4f}, v_n:{v_n:.4f}, pd_c:{pd_c:.4f}, pd_n:{pd_n:.4f}")
        # print(sc[0].data)
        
        
    w_trained = to_numpy(model.mlp.weight)
    b_trained = to_numpy(model.mlp.bias)
    
    import matplotlib.pyplot as plt
    
    
    
    fig, axs = plt.subplots(2,1,)
    for i in range(model.hash_dim):
        x = np.linspace(-100, 100, 1000)
        y = (b_orignal[i] - w_orignal[i,0] * x) / w_orignal[i,1]
        axs[0].plot(x, y)
    axs[0].set_xlim(-100, 100)
    axs[0].set_ylim(-100, 100)
    axs[0].grid()
    
    for i in range(model.hash_dim):
        x = np.linspace(-100, 100, 1000)
        y = (b_trained[i] - w_trained[i,0] * x) / w_trained[i,1]
        axs[1].plot(x, y)
    axs[1].set_xlim(-100, 100)
    axs[1].set_ylim(-100, 100)
    axs[1].grid()
    plt.show()
