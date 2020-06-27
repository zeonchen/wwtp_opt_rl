import sys
import os
import numpy as np
import pandas as pd

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
    gpsx.setValue('qconwas', float(action[1]))

    # set time
    gpsx.setTstop(200)
    gpsx.setCint(0.5)


def cint():
    # acquire state values
    cod = gpsx.getValue('codof2')
    bod = gpsx.getValue('bodof2')
    tn = gpsx.getValue('tnof2')
    tp = gpsx.getValue('tpof2')
    nh4 = gpsx.getValue('snhof2')
    no3 = gpsx.getValue('snoaof2')
    no2 = gpsx.getValue('snoiof2')
    ss = gpsx.getValue('xof2')

    sludge_tss = gpsx.getValue('xof3')
    sludge_flow = gpsx.getValue('qof3')
    sludge = gpsx.getValue('xmf54')
    onsite_energy = gpsx.getValue('totalpowerkw')
    bio = gpsx.getValue('electricitySavedof5')
    outflow = gpsx.getValue('qof2')
    process_ghg = gpsx.getValue('scopeOneTotal')
    methane = gpsx.getValue('scopeOneOffset')

    state = [cod, bod, tn, tp, nh4, no3, no2, ss, sludge_tss, sludge_flow, sludge,
             onsite_energy, bio, outflow, process_ghg, methane]
    time = gpsx.getValue('t')

    if time >= 100:
        for i, variable in enumerate(state_variable):
            state_dict[variable].append(state[i])

    if time == 200:
        for i, variable in enumerate(state_variable):
            state[i] = np.array(state_dict[variable]).mean()
            if i < 8:
                variance[i] = np.array(state_dict[variable]).std()

        state.append(max(variance))
        np.savetxt('.\\res\\mc.txt', state)

    # print('State saved!' + str(epoch))
    # sys.stdout.flush()


def eor():
    global finished
    finished = True


try:
    epoch = 0
    df = pd.DataFrame([0] * 19).T
    for _ in range(1001):
        print('Epoch {}'.format(epoch))
        sys.stdout.flush()

        variance = [0] * 8
        state_dict = {}
        state_variable = ['cod', 'bod', 'tn', 'tp', 'nh4', 'no3', 'no2', 'ss', 'sludge_tss', 'sludge_flow', 'sludge',
                          'onsite_energy', 'bio', 'outflow', 'process_ghg', 'methane']
        for variable in state_variable:
            state_dict[variable] = []

        # get actions
        action = np.random.uniform(0.0, 1.0, 2) * [8, 200.0]

        # run simulation
        runSim()
        gpsx.resetAllValues()

        # wait for state collection
        state = np.loadtxt('.\\res\\mc.txt')
        # print(state)
        # sys.stdout.flush()
        output = np.concatenate([action, state])
        output = pd.DataFrame(output).T
        df = df.append(output)
        epoch += 1

    df.to_excel('D:\\rl_wwtp\\outputs\\mc_cass1.xlsx')

except Exception:
    pass
