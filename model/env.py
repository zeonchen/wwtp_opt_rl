import logging
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import sys

sys.path.append("D:\\rl_wwtp\\")
from model.objective import ObjectiveFunction
from utils.tool import clear_text, txt_read
import pandas as pd

logger = logging.getLogger(__name__)


class RLWWTP(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.of = ObjectiveFunction()
        self.energy_max, self.energy_min, self.cost_max, self.cost_min, self.eutro_max, \
        self.eutro_min, self.ghg_max, self.ghg_min = self.of.min_max(self.df.iloc[:, :])
        self.influent = pd.read_csv('D:\\rl_wwtp\\outputs\\influent.csv')
        self.min_, self.max_ = self.min_max()

        self.dim = 12
        observation_high = np.array([1] * self.dim)
        observation_low = np.array([0] * self.dim)
        action_high = np.array([1.0, 1.0])
        action_low = np.array([0, 0])
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        self.observation_space = spaces.Box(low=observation_low, high=observation_high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.df = pd.DataFrame([0] * 19).T
        self.do = np.array([0.0])
        self.dosage = np.array([0.0])
        self.last_do = 1.5 / 5.0
        self.last_dosage = 250 / 800.0
        self.time = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, actions):
        current_do = actions[0]
        current_dosage = actions[1]
        self.do = np.array([actions[0]])
        self.dosage = np.array([actions[1]])

        actions = np.insert(self.dosage * 800000, 0, self.do * 5)
        # actions
        action_path_1 = "D:\\rl_wwtp\\outputs\\parallel\\temp_action1.txt"
        np.savetxt(action_path_1, actions)

        # today effluent
        state_path_1 = "D:\\rl_wwtp\\outputs\\parallel\\temp_state1.txt"
        today_effluent_1 = txt_read(state_path_1)
        clear_text(action_path_1)  # clear data in temp_action.txt
        clear_text(state_path_1)

        norm_energy_1, norm_cost_1, norm_eutro_1, norm_ghg_1, reward_1 = self.reward_calculate(today_effluent_1,
                                                                                               actions)

        state_1 = (today_effluent_1 - pd.Series(self.min_[:11])) / (pd.Series(self.max_[:11] - self.min_[:11]))
        reward_1 += abs(self.last_do-current_do)
        reward_1 += abs(self.last_dosage-current_dosage)

        self.last_do = current_do
        self.last_dosage = current_dosage

        # self.of.output()
        return state_1, -reward_1, False, {}

    def reward_calculate(self, today_effluent, action):
        # unit conversion
        onsite_energy = today_effluent[24-13]
        aeration_power = today_effluent[29-13]
        recover_heat = today_effluent[31-13] / 1e8
        sludge_tss = today_effluent[21-13]
        sludge_flow = today_effluent[22-13]
        dosage = action[1] / 1000
        bio_elec = today_effluent[25-13] / 24
        sludge = today_effluent[23-13] / 1000
        outflow = today_effluent[26-13]

        process_ghg = today_effluent[27-13] / 1000
        methane_offset = today_effluent[28-13] / 1000

        energy_consumption = self.of.energy_consumption(onsite_energy=onsite_energy, aeration_power=aeration_power,
                                                        recover_heat=recover_heat,
                                                        sludge_tss=sludge_tss, sludge_flow=sludge_flow, dosage=dosage,
                                                        bio_elec=bio_elec)

        cost = self.of.cost(total_energy=energy_consumption, sludge_tss=sludge_tss, sludge_flow=sludge_flow,
                            dosage=dosage, sludge=sludge, bio_elec=bio_elec)

        cod = today_effluent[13-13]
        bod = today_effluent[14-13]
        tn = today_effluent[15-13]
        tp = today_effluent[16-13]
        nh4 = today_effluent[17-13]
        no3 = today_effluent[18-13]
        no2 = today_effluent[19-13]
        ss = today_effluent[20-13]

        eutro_po = self.of.eutrophication_potential(cod=cod, tp=tp, nh4=nh4, no3=no3, no2=no2, outflow=outflow)

        ghg = self.of.greenhouse_gas(process=process_ghg, energy=energy_consumption, sludge_flow=sludge_flow,
                                     sludge_tss=sludge_tss, bod=bod, tn=tn, dosage=dosage,
                                     outflow=outflow, sludge=sludge, bio_elec=bio_elec)

        # normalization
        norm_energy = self.of.normalization(energy_consumption, self.energy_min, self.energy_max)
        norm_cost = self.of.normalization(cost, self.cost_min, self.cost_max)
        norm_eutro = self.of.normalization(eutro_po, self.eutro_min, self.eutro_max)
        norm_ghg = self.of.normalization(ghg, self.ghg_min, self.ghg_max)

        # reward
        reward = (0.38 * norm_energy + 0.26 * norm_eutro + 0.36 * norm_ghg)
        # reward = norm_cost

        # if cod > 60 or nh4 > 8 or tn > 20 or bod > 20 or ss > 20 or tp > 1:
        if cod > 50 or nh4 > 5 or tn > 15 or bod > 10 or ss > 10 or tp > 0.5:
        # if cod > 40 or nh4 > 4 or tp > 0.3 or tn > 15:
            reward += 1

        return norm_energy, norm_cost, norm_eutro, norm_ghg, reward

    def reset(self):
        influent6 = np.array(
            [0.29744381, 0.311315646, 0.715897745, 0.391896867, 0.483235526, 0.534866349, 0.613485808, 0.445713426,
             0.586010335,
             0.089227437, 0.193617513, 0.3, 0.3125])
        self.last_do = 1.5 / 5.0
        self.last_dosage = 250 / 800.0

        return np.concatenate((influent6[:11], [0.3]))

    def min_max(self):
        tot = pd.DataFrame([0] * 10).T
        for i in range(20):
            temp = pd.read_excel('D:\\rl_wwtp\\outputs\\lgbm_data\\mc_mpc_{}.xlsx'.format(i))
            tot = pd.concat((tot, temp), axis=0)

        tot = tot.iloc[1:, :]
        tot = tot.drop(["Unnamed: 0"], axis=1)
        min_ = np.min(tot)
        max_ = np.max(tot)
        min_ = pd.Series(min_.drop([22, 36, 37, 38]).values)
        max_ = pd.Series(max_.drop([22, 36, 37, 38]).values)
        return min_, max_
