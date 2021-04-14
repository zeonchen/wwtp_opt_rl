import sys
import os
import numpy as np
import time

state_save_path = 'D:\\rl_wwtp\\outputs\\parallel\\temp_state1.txt'
action_save_path = 'D:\\rl_wwtp\\outputs\\parallel\\temp_action1.txt'
epoch = 0


def start():
    """
    gpsx.getValue()
    gpsx.setValue()
    gpsx.setTstop()
    :return:
    """
    # set time
    gpsx.setTstop(30)
    gpsx.setCint(0.041667*2)


def cint():
    cod = gpsx.getValue('cod59')
    bod = gpsx.getValue('bod59')
    tn = gpsx.getValue('tn59')
    tp = gpsx.getValue('tp59')
    nh4 = gpsx.getValue('snh59')
    no3 = gpsx.getValue('snoa59')
    no2 = gpsx.getValue('snoi59')
    ss = gpsx.getValue('x59')

    sludge_tss = gpsx.getValue('xwthick')
    sludge_flow = gpsx.getValue('qwthick')
    sludge = gpsx.getValue('xmf47')
    onsite_energy = gpsx.getValue('totalpowerkw')
    bio = gpsx.getValue('electricitySavedwdig')
    outflow = gpsx.getValue('q59')
    process_ghg = gpsx.getValue('scopeOneTotal')
    methane = gpsx.getValue('scopeOneOffset')

    aeration_power = gpsx.getValue('powerairkwaer')
    digester_power1 = gpsx.getValue('heatGenInBoilerwdig')
    digester_power2 = gpsx.getValue('heatGenFromCHPwdig')
    digester_power3 = gpsx.getValue('totalHeatGenwdig')
    digester_power4 = gpsx.getValue('heatDeficitwdig')
    digester_elec = gpsx.getValue('electricitySavedwdig')

    was_flow = gpsx.getValue('qwas')
    thick_flow = gpsx.getValue('qwthick')
    de_waste_flow = gpsx.getValue('qwaste')
    de_filtrate_flow = gpsx.getValue('qfiltrate')

    # inflow
    SF = gpsx.getValue('ssraw')
    SA = gpsx.getValue('sacraw')
    SI = gpsx.getValue('siraw')
    SNH4 = gpsx.getValue('snhraw')
    SPO = gpsx.getValue('spraw')
    XI = gpsx.getValue('xiraw')
    XS = gpsx.getValue('xsraw')
    XH = gpsx.getValue('xbhraw')
    TSS = gpsx.getValue('xiiraw')
    Q = gpsx.getValue('qconraw')
    T = gpsx.getValue('temp')

    inf = [SF, SA, SI, SNH4, SPO, XI, XS, XH, TSS, Q, T]
    state = [cod, bod, tn, tp, nh4, no3, no2, ss, sludge_tss, sludge_flow, sludge,
             onsite_energy, bio, outflow, process_ghg, methane, aeration_power, digester_power1,
             digester_power2, digester_power3, digester_power4, digester_elec, was_flow,
             thick_flow, de_waste_flow, de_filtrate_flow]
    # state = inf
    time = gpsx.getValue('t')

    if time > 20:
        np.savetxt(state_save_path, state)
        state_size = os.path.getsize(state_save_path)
        while state_size != 0:
            state_size = os.path.getsize(state_save_path)

        action_size = os.path.getsize(action_save_path)

        action = []
        # run the simulation and output
        while action_size == 0:
            action_size = os.path.getsize(action_save_path)

        with open(action_save_path, "r+") as f:  # read actions
            action_ = f.readlines()
            for i in action_:
                item = float(i.strip('\n'))
                action.append(item)

        gpsx.setValue('setpsoaer', action[0])
        gpsx.setValue('chem2dosager65', action[1])


def eor():
    global finished
    finished = True


try:
    epoch = 0
    while True:
    # for _ in range(1):
        print('Epoch {}'.format(epoch))
        sys.stdout.flush()

        gpsx.setValue('setpsoaer', 1.5)
        gpsx.setValue('chem2dosager65', 250000.0)

        # run simulation
        runSim()
        gpsx.resetAllValues()

        # wait for state collection
        state_size = os.path.getsize(state_save_path)

        epoch += 1

        while state_size != 0:
            state_size = os.path.getsize(state_save_path)

except Exception:
    pass
