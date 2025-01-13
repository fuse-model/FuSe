import os 



data_dir = '/home/paulzhou/bridge_with_digit/v3_dnw'


num_collected_trajectories = 0 
total_len = 0
for dated_dir in os.listdir(data_dir): 
    if dated_dir[:4] != "2024" or 'raw' not in os.listdir(os.path.join(data_dir, dated_dir)): 
        continue 
    dated_dir = os.path.join(data_dir, dated_dir, 'raw', 'traj_group0')
    for trajdir in os.listdir(dated_dir): 
        if trajdir[:4] != 'traj': 
            continue 
        assert os.path.exists(os.path.join(dated_dir, trajdir, 'images0', 'im_0.jpg'))
        num_collected_trajectories += 1 
        
        total_len += len(os.listdir(os.path.join(dated_dir, trajdir, 'images0')))
        
    

print(f'Collected {num_collected_trajectories} so far!')
print(f'Which corresponds to {total_len * 0.2} seconds')
print(f'Or {total_len * 0.2 / 60} minutes')
print(f'Or {total_len * 0.2 / 3600.0} hours')