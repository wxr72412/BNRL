read_cpd_rould_decimal = None

# read_cpd_rould_decimal = 3
# read_cpd_rould_decimal = 2
# read_cpd_rould_decimal = 1

# read_cpd_rould = None
# read_cpd_rould = 3
# read_cpd_rould = 2
read_cpd_rould = 1  # 除了在执行rename时候，学的时候需设置为1

print_float_round = 6

# threshold_delta = 0.06
# threshold_delta = 0.01
threshold_delta = - 1

# max_iter = 7
max_iter = 1


# seed = 22
seed = 23
# seed = 42


# bn_name = ['toy2']
# bn_data = ['data/my_toy/toy2_origin.bif']
# missingdata = ['data/my_toy/toy2_origin_L2_12.xlsx']


# bn_name = ['toy2']
# num_sample = 12
# num_latent = 2
# num_VQVAE = 1


bn = 'child'
# bn = 'water'
# bn = 'munin1'
# bn = 'pigs'

# num_sample = 10
num_sample = 100
# num_sample = 200
# num_sample = 400
# num_sample = 800
# num_sample = 1600
# num_sample = 1000
num_latent = 3
num_VQVAE = 1

# num_sample = 100
# num_latent = 1
# num_latent = 3
# num_latent = 5
# num_latent = 7
# num_latent = 9
# num_VQVAE = 1

# num_sample = 20000
# num_latent = 2
# num_VQVAE = 1

num_infer = num_sample
n_samples = 1000

bn_data = []
missingdata = []
output_file = []


bn_data.append('data\\' + bn + '\\missingdata\\' + str(num_sample) + '\\' + str(num_latent) + '\\' + bn + '.bif')
missingdata.append('data\\' + bn + '\\missingdata\\' + str(num_sample) + '\\' + str(num_latent) + '\\' + bn + '.xlsx')
output_file.append('data\\' + bn + '\\missingdata\\' + str(num_sample) + '\\' + str(num_latent) + '\\' + bn)

# for bn in bn_name:
#     bn_data.append('data\\' + bn + '\\VQVAE\\'+ str(num_sample) + '\\' + str(num_VQVAE) + '\\' + bn + '_VQVAE.bif')
#     missingdata.append('data\\' + bn + '\\VQVAE\\' + str(num_sample) + '\\' + str(num_VQVAE) + '\\' + bn + '_VQVAE_L.xlsx')
#     output_file.append('data\\' + bn + '\\VQVAE\\' + str(num_sample) + '\\' + str(num_VQVAE) + '\\' + bn + '_VQVAE')









