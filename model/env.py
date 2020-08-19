import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import sys

sys.path.append("D:\\rl_wwtp\\")
from model.objective import ObjectiveFunction
from model.surrogate.nn import output
from utils.tool import clear_text, txt_read
import torch
import pandas as pd

logger = logging.getLogger(__name__)


class RLWWTP(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.df = pd.read_excel('.\\outputs\\mc.xlsx')
        self.energy_max, self.energy_min, self.cost_max, self.cost_min, self.eutro_max, \
        self.eutro_min, self.ghg_max, self.ghg_min = ObjectiveFunction.min_max(self.df.iloc[:, :])

        # action parameters
        self.max_o2 = 5.0
        self.max_dosage = 200.0

        self.dim = 1
        observation_high = np.array([1000] * self.dim)
        observation_low = np.array([0] * self.dim)

        action_high = np.array([self.max_o2, self.max_dosage])
        action_low = np.array([0, 0])
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        self.observation_space = spaces.Box(low=observation_low, high=observation_high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, actions):
        self.do = np.array([actions[0]])
        self.dosage = np.array([actions[1]])
        # if actions == 0:
        #     self.do -= 0.2
        #     if self.do <= 0:
        #         self.do = np.array([0.0])
        # elif actions == 1:
        #     self.do += 0.2
        #     if self.do > 8:
        #         self.do = np.array([8.0])
        # elif actions == 2:
        #     self.dosage -= 20000.0
        #     if self.dosage <= 0:
        #         self.dosage = np.array([0.0])
        # elif actions == 3:
        #     self.dosage += 20000.0
        #     if self.dosage > 800000.0:
        #         self.dosage = np.array([800000.0])
        # else:
        #     pass

        actions = np.insert(self.dosage * 200000, 0, self.do * 5.0)
        # actions
        action_path_1 = ".\\outputs\\parallel\\temp_action1.txt"
        np.savetxt(action_path_1, actions)

        # today effluent
        state_path_1 = ".\\outputs\\parallel\\temp_state1.txt"
        today_effluent_1 = txt_read(state_path_1)
        clear_text(action_path_1)  # clear data in temp_action.txt
        clear_text(state_path_1)

        norm_energy_1, norm_cost_1, norm_eutro_1, norm_ghg_1, reward_1 = self.reward_calculate(today_effluent_1,
                                                                                               actions)
        state_1 = self.dosage

        return state_1, -reward_1, False, {}

    def reward_calculate(self, today_effluent, action):
        # unit conversion
        sludge_tss = today_effluent[8]
        sludge_flow = today_effluent[9]
        sludge_pro = today_effluent[10] / 1000
        onsite_energy = today_effluent[11]
        bio = today_effluent[12] / 24
        outflow = today_effluent[13]
        process_ghg = today_effluent[14] / 1000
        methane_offset = today_effluent[15] / 1000
        dosage = action[1] / 1000

        energy_consumption = ObjectiveFunction.energy_consumption(onsite_energy=onsite_energy, sludge_tss=sludge_tss,
                                                                  sludge_flow=sludge_flow, dosage=dosage)

        cost = ObjectiveFunction.cost(onsite_energy=onsite_energy, sludge_tss=sludge_tss, sludge_flow=sludge_flow,
                                      sludge=sludge_pro, dosage=dosage, bio=bio)

        eutro_po = ObjectiveFunction.eutrophication_potential(today_effluent[0], today_effluent[1], today_effluent[2],
                                                              today_effluent[3], today_effluent[4], today_effluent[5],
                                                              today_effluent[6], today_effluent[7], outflow)
        cod = today_effluent[0]
        bod = today_effluent[1]
        tn = today_effluent[2]
        tp = today_effluent[3]
        nh4 = today_effluent[4]
        ss = today_effluent[7]

        ghg = ObjectiveFunction.greenhouse_gas(process=process_ghg, energy=energy_consumption, sludge_flow=sludge_flow,
                                               sludge_tss=sludge_tss, bod=today_effluent[1], tn=today_effluent[2],
                                               dosage=dosage, methane=methane_offset, outflow=outflow)

        # normalization
        norm_energy = ObjectiveFunction.normalization(energy_consumption, self.energy_min, self.energy_max)
        norm_cost = ObjectiveFunction.normalization(cost, self.cost_min, self.cost_max)
        norm_eutro = ObjectiveFunction.normalization(eutro_po, self.eutro_min, self.eutro_max)
        norm_ghg = ObjectiveFunction.normalization(ghg, self.ghg_min, self.ghg_max)

        # reward
        reward = 0.25 * (norm_energy + norm_eutro + norm_cost + norm_ghg)

        if cod > 50 or nh4 > 5 or tn > 15 or bod > 10 or ss > 10 or tp > 0.5:
            reward += 1

        return norm_energy, norm_cost, norm_eutro, norm_ghg, reward

    def reset(self):
        state = self.np_random.uniform(low=[0] * self.dim, high=[1] * self.dim)

        return np.array(state)


