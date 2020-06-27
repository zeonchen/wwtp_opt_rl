import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import sys
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
        self.df = pd.read_excel('.\\model\\surrogate\\data\\data.xlsx')
        self.energy_max, self.energy_min, self.cost_max, self.cost_min, self.eutro_max, \
        self.eutro_min, self.ghg_max, self.ghg_min, self.variance_max, self.variance_min = ObjectiveFunction.min_max(
            self.df.iloc[:, 2:])

        # action parameters
        self.max_o2 = 8.0
        self.max_sludge = 200.0

        # state parameters
        observation_high = np.array([1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 0])
        observation_low = np.array([0, 0, 0, 0, 0, 0, 0, 0, -10])
        action_high = np.array([self.max_o2, self.max_sludge])
        action_low = np.array([0, 0])

        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        self.observation_space = spaces.Box(low=observation_low, high=observation_high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.df = pd.DataFrame([0] * 19).T

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, parallel=2):
        action_path = ".\\outputs\\parallel\\temp_action.txt"
        np.savetxt(action_path, action)

        # today effluent
        state_path = ".\\outputs\\parallel\\temp_state.txt"
        today_effluent = txt_read(state_path)
        clear_text(action_path)  # clear data in temp_action.txt
        clear_text(state_path)

        # unit conversion
        sludge_tss = today_effluent[8]
        sludge_flow = today_effluent[9]
        sludge_pro = today_effluent[10] / 1000
        onsite_energy = today_effluent[11]
        bio = today_effluent[12] / 24
        outflow = today_effluent[13]
        process_ghg = today_effluent[14] / 1000
        methane_offset = today_effluent[15] / 1000

        energy_consumption = ObjectiveFunction.energy_consumption(onsite_energy=onsite_energy, sludge_tss=sludge_tss,
                                                                  sludge_flow=sludge_flow, bio=bio)

        cost = ObjectiveFunction.cost(energy=energy_consumption, sludge_tss=sludge_tss, sludge_flow=sludge_flow,
                                      sludge=sludge_pro)

        eutro_po= ObjectiveFunction.eutrophication_potential(today_effluent[0], today_effluent[1], today_effluent[2],
                                                            today_effluent[3], today_effluent[4], today_effluent[5],
                                                            today_effluent[6], today_effluent[7], outflow)
        cod = today_effluent[0]
        bod = today_effluent[1]
        tn = today_effluent[2]
        tp = today_effluent[3]
        nh4 = today_effluent[4]
        no3 = today_effluent[5]
        no2 = today_effluent[6]
        ss = today_effluent[7]

        ghg = ObjectiveFunction.greenhouse_gas(process=process_ghg, energy=energy_consumption, sludge_flow=sludge_flow,
                                               sludge_tss=sludge_tss, bod=today_effluent[1], tn=today_effluent[2],
                                               methane=methane_offset, outflow=outflow)

        # normalization
        norm_energy = ObjectiveFunction.normalization(energy_consumption, self.energy_min, self.energy_max)
        norm_cost = ObjectiveFunction.normalization(cost, self.cost_min, self.cost_max)
        norm_eutro = ObjectiveFunction.normalization(eutro_po, self.eutro_min, self.eutro_max)
        norm_ghg = ObjectiveFunction.normalization(ghg, self.ghg_min, self.ghg_max)
        norm_variance = ObjectiveFunction.normalization(today_effluent[16], self.variance_min, self.variance_max)

        # reward
        reward = ObjectiveFunction.weighted_sum(norm_cost, norm_energy, norm_eutro, norm_ghg)
        # reward = norm_cost * 0.5 + norm_eutro * 0.5

        if cod > 60 or nh4 > 8 or tn > 20 or bod > 20 or ss > 20 or tp > 2:
            reward += 1

        reward += norm_variance

        state = np.concatenate((today_effluent[:8], [-reward]))

        output = np.concatenate([action,today_effluent])
        output = pd.DataFrame(output).T
        self.df = self.df.append(output)

        self.df.to_excel('D:\\rl_wwtp\\outputs\\tsne.xlsx')
        return state, -reward, False, {}

    def reset(self):
        state = self.np_random.uniform(low=[0, 0, 0, 0, 0, 0, 0, 0, -1], high=[0, 0, 0, 0, 0, 0, 0, 0, -1])

        return np.array(state)




