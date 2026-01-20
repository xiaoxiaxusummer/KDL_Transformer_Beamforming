import numpy as np 
import math, os
from scipy.io import savemat
from types import SimpleNamespace
from numpy import newaxis as NA
import pickle

def convert_to_numpy(data):
    if isinstance(data, dict):
        return {k: convert_to_numpy(v) for k, v in data.items()}
    elif isinstance(data, list):
        return np.array(data)
    else:
        return data


def dbm_to_watts(P_dbm):
    """
    Conversion from dBm to watt (W).
    @ param:
        p_dbm (float): power in dBm
    
    @ returns:
        float: power in W
    """
    return 10 ** ((P_dbm - 30) / 10)

current_directory = os.path.dirname(os.path.abspath(__file__))
PATH_SAVE = os.path.join(current_directory, 'datasets')


TRAIN_SEED = 1111
TEST_SEED = 2222

def generate_data(K, L, N, P_max_dBm=20, S_PAA = 20, space_size = 20, n_sample = 30000, saveData=False, testData=True, fileName = None):

    if fileName is not None and os.path.exists(os.path.join(PATH_SAVE, fileName)):
        with open(os.path.join(PATH_SAVE, fileName), 'rb') as f:
            data = pickle.load(f)
            # data = convert_to_numpy(data)
            # savemat(os.path.join(PATH_SAVE, f'{fileName[0:-3]}mat'), data)
            f.close()
            cfg = data["cfg"]
            loc_user = data['loc_user']
            h_tilde = data['h_tilde']
            r0 = data['r0']
            theta = data['theta']
            varsigma = data['varsigma']
            A = data['A']
            cfg.P_max_dBm = P_max_dBm
            cfg.P_max = dbm_to_watts(cfg.P_max_dBm)
        if not cfg.space_size_x == space_size:  # scale for comparison
            loc_user[:,:,0] = loc_user[:,:,0] /cfg.space_size_x * space_size
            r0 = np.sqrt(((loc_user - np.array([0,0,cfg.h_PAA]))**2).sum(-1)) # [n_s, K]
            ang_elevation = np.arcsin(cfg.h_PAA/r0) # elevation angles of users
            ang_azimuth = np.arccos(loc_user[:,:,0]/r0/np.cos(ang_elevation)) # azimuth angles of users
            theta = np.cos(ang_elevation) * np.cos(ang_azimuth)
            varsigma = 1 + (cfg.loc_PA[:,1][NA,:,NA]/r0[:,NA,:])**2 
            varsigma = varsigma - 1/r0[:,NA,:]*2*cfg.loc_PA[:,1][NA,:,NA]*np.cos(ang_elevation[:,NA,:])*np.sin(ang_azimuth[:,NA,:])
            h_tilde = np.sqrt(cfg.beta) * np.exp(1j* cfg.kappa *r0)  / r0 # \tilde{h} [n_s, K]
            A = r0*theta # [n_s,K]
            cfg.space_size_x = space_size
            cfg.S_PAA = S_PAA
        return cfg, loc_user, h_tilde, r0, theta, varsigma, A
    else:
        if testData:
            np.random.seed(TEST_SEED)
        else:
            np.random.seed(TRAIN_SEED)
        cfg = SimpleNamespace()
        cfg.space_size_x = space_size
        cfg.space_size_y = 20
        cfg.frequency = 30e9 
        cfg.light_speed = 3e8 # 3*10**8
        cfg.wavelength = cfg.light_speed/cfg.frequency
        cfg.kappa = 2*math.pi/cfg.wavelength
        cfg.minimum_spacing = cfg.wavelength/2
        cfg.n_eff = 1.4
        cfg.P_max_dBm = 20 # The maximum transmission power: 0~20 [dBm] 
        cfg.P_max = dbm_to_watts(cfg.P_max_dBm) # The maximum transmission power [W]
        cfg.noise_dBm = -90
        cfg.noise = dbm_to_watts(cfg.noise_dBm)
        cfg.sigma2 = cfg.noise/cfg.noise*1e-6 # Normalized noise power
        cfg.beta = cfg.light_speed/(4*math.pi*cfg.frequency)
        cfg.beta = cfg.beta/cfg.noise*1e-6 # Normalized channel gain
        cfg.Rmin = 1
        cfg.S_PAA = S_PAA
        cfg.h_user = 0
        cfg.h_PAA = 2.5 # height of PAA [meter]


        cfg.K = K # Number of users
        cfg.L = L # Number of pinching antennas along each waveguide
        cfg.N = N # Number of waveguides
        cfg.M = cfg.L*cfg.N # Number of pinching antennas
        cfg.n_sample = n_sample
        cfg.P_max_dBm = P_max_dBm
        cfg.P_max = dbm_to_watts(cfg.P_max_dBm)
        cfg.set_L = [[n*cfg.L+l for l in range(cfg.L)] for n in range(cfg.N)]
        cfg.dW = cfg.space_size_y/cfg.N # spacing of waveguides [meter]
        cfg.loc_feedpoint = np.array([[0, n*cfg.dW, cfg.h_PAA] for n in range(cfg.N)])
        x =  np.tile(np.linspace(0, cfg.S_PAA, cfg.L), cfg.N)
        cfg.loc_PA = np.array([[x[m], (np.ceil((m+1)/cfg.L)-1)*cfg.dW, cfg.h_PAA] for m in range(cfg.M)])
        cfg.Gamma = np.kron(np.eye(N), np.ones((L, 1))) # (M,N)
        loc_user_x_coords = np.random.uniform(cfg.space_size_x*0.5, cfg.space_size_x, cfg.K*cfg.n_sample) # [K*n_s]
        loc_user_y_coords = np.random.uniform(0, cfg.space_size_y, cfg.K*cfg.n_sample) # [K*n_s]
        loc_user = np.column_stack((loc_user_x_coords, loc_user_y_coords, np.zeros(cfg.K*cfg.n_sample))) # [K*n_s,3]
        loc_user = np.array(np.array_split(loc_user, cfg.n_sample, axis=0)) # [n_s, K, 3]
        r0 = np.sqrt(((loc_user - np.array([0,0,cfg.h_PAA]))**2).sum(-1)) # [n_s, K]
        ang_elevation = np.arcsin(cfg.h_PAA/r0) # elevation angles of users
        ang_azimuth = np.arccos(loc_user[:,:,0]/r0/np.cos(ang_elevation)) # azimuth angles of users
        theta = np.cos(ang_elevation) * np.cos(ang_azimuth)
        # assert(np.abs(loc_user[:,:,1] - r0*np.cos(ang_elevation)*np.cos(ang_azimuth)).sum() < 1e-5)
        varsigma = 1 + (cfg.loc_PA[:,1][NA,:,NA]/r0[:,NA,:])**2 
        varsigma = varsigma - 1/r0[:,NA,:]*2*cfg.loc_PA[:,1][NA,:,NA]*np.cos(ang_elevation[:,NA,:])*np.sin(ang_azimuth[:,NA,:])
        h_tilde = np.sqrt(cfg.beta) * np.exp(1j* cfg.kappa *r0)  / r0 # \tilde{h} [n_s, K]
        A = r0*theta # [n_s,K]

        # eta = np.sqrt( (x[None,:,NA]/r0[:,NA,:])**2 - 2*theta[:,NA,:]/r0[:,None,:]*x[None,:,NA] + varsigma) - 1 # [M,K]
        # R = np.sqrt(x[None,:,None]**2 - 2*A[:,None,:]*x[None,:,None] + r0[:,None,:]**2 * varsigma)
        # h_x = np.sqrt(cfg.beta) / R * np.exp(1j*cfg.kappa*R)

        if not saveData:
            return cfg, loc_user, h_tilde, r0, theta, varsigma, A
        else:
            data = {
                'loc_user': loc_user,
                'h_tilde': h_tilde,
                'r0': r0,
                'cfg': cfg, 
                'theta': theta,
                'varsigma': varsigma,
                'A': A, 
            }

            with open(os.path.join(PATH_SAVE, fileName), 'wb') as f:
                pickle.dump(data, f)
            # # save to mat file
            # savemat('example_data.mat', data)
            return cfg, loc_user, h_tilde, r0, theta, varsigma, A



if __name__ == '__main__':
    generate_data(K=6, L=8, N=4, saveData=True, testData=True)
    generate_data(K=6, L=8, N=4, saveData=True, testData=False)