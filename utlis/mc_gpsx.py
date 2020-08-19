import sys
import os
import numpy as np
import pandas as pd
import time

state_save_path = "D:\\rl_wwtp\\outputs\\parallel\\temp_state2.txt"
action_save_path = "D:\\rl_wwtp\\outputs\\parallel\\temp_action2.txt"
epoch = 0


def start():
    """
    gpsx.getValue()
    gpsx.setValue()
    gpsx.setTstop()
    :return:
    """
    # aeration
    gpsx.setValue('setpsoaer', action[0])

    # srt
    gpsx.setValue('chem2dosager65', float(action[1]))

    # set time
    gpsx.setTstop(70)
    gpsx.setCint(0.04166)


def cint():
    cod = gpsx.getValue('cod59')
    bod = gpsx.getValue('bod59')
    tn = gpsx.getValue('tn59')
    tp = gpsx.getValue('tp59')
    nh4 = gpsx.getValue('snh59')
    no3 = gpsx.getValue('snoa59')
    no2 = gpsx.getValue('snoi59')
    ss = gpsx.getValue('x59')

    sludge_tss = gpsx.getValue('x47')
    sludge_flow = gpsx.getValue('q47')
    sludge = gpsx.getValue('xmf47')
    onsite_energy = gpsx.getValue('totalpowerkw')
    bio = gpsx.getValue('electricitySavedwdig')
    outflow = gpsx.getValue('q59')
    process_ghg = gpsx.getValue('scopeOneTotal')
    methane = gpsx.getValue('scopeOneOffset')

    state = [cod, bod, tn, tp, nh4, no3, no2, ss, sludge_tss, sludge_flow, sludge,
             onsite_energy, bio, outflow, process_ghg, methane]
    time = gpsx.getValue('t')

    if time >= 60:
        for i, variable in enumerate(state_variable):
            state_dict[variable].append(state[i])

    if time >= 79.8:
        for i, variable in enumerate(state_variable):
            state[i] = np.array(state_dict[variable]).mean()

        # state.append(max(variance))
        np.savetxt(state_save_path, state)


def eor():
    global finished
    finished = True


try:
    epoch = 0
    df = pd.DataFrame([0] * 18).T
    while True:
        print('Epoch {}'.format(epoch))
        sys.stdout.flush()
        gpsx.resetAllValues()
        state_dict = {}
        state_variable = ['cod', 'bod', 'tn', 'tp', 'nh4', 'no3', 'no2', 'ss', 'sludge_tss', 'sludge_flow', 'sludge',
                          'onsite_energy', 'bio', 'outflow', 'process_ghg', 'methane']
        for variable in state_variable:
            state_dict[variable] = []
        action = np.random.random(2) * [5, 200]
        # run simulation
        runSim()
        gpsx.resetAllValues()

        state = np.loadtxt(state_save_path)
        state = np.concatenate((action, state))
        output = pd.DataFrame(state).T
        df = df.append(output)
        df.to_excel('.\\outputs\\mc.xlsx')
        print('Epoch {} finished!'.format(epoch))
        sys.stdout.flush()
        epoch += 1

except Exception:
    pass
