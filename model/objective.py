import numpy as np


class ObjectiveFunction(object):
    """
    This is the class used to generate objective function, objective function consists of economic term
    and LCA term (energy consumption, eutrophication potential and green house for now). Each part is calculated
    in one day. The final score is obtained through weighted sum of two terms, and used as reward in reinforcement
    learning.
    """
    def __init__(self):
        self.cost_comp = []
        self.energy_comp = []
        self.ep_comp = []
        self.ghg_comp = []

    def cost(self, total_energy, sludge_tss, sludge_flow, dosage, sludge, bio_elec):
        """
        :param energy: kWh/d;
        :param sludge: kg/d;
        :return: scalar.
        """
        # 30为投加量
        dosage_for_sludge = sludge_tss * sludge_flow / 1000 * 30 * 0.4 / 1000  # kg FeCl3/d
        dosage_for_tp = dosage * 0.25  # kg PAC/d

        cost_tot = 0.8 * total_energy + 1.7 * dosage_for_sludge + 2.5 * dosage_for_tp + \
            0.005 * (dosage_for_sludge + dosage_for_tp + sludge) + 0.52 * sludge - bio_elec * 24 * 0.25 + \
            0.3 * 2000

        return cost_tot

    def energy_consumption(self, onsite_energy, aeration_power, recover_heat,
                           sludge_tss, sludge_flow, dosage, bio_elec):
        """
        :param bio: methane power, kw;
        :param sludge_tss: mean sludge TSS, mg/L (g/m3);
        :param sludge_flow: sludge flow, m3/d;
        :param onsite_energy: aeration and plump power, kw;

        :return: mean daily energy consumption, kWh/d.
        """
        dosage_for_sludge = sludge_tss * sludge_flow / 1000 * 30 * 0.4 / 1000  # kg FeCl3/d
        dosage_for_tp = dosage * 0.25  # kg PAC/d

        energy_tot = 24 * onsite_energy - 24 * recover_heat + 3.4 * dosage_for_sludge + 1.94 * dosage_for_tp - \
                     24 * bio_elec

        return energy_tot

    def eutrophication_potential(self, cod, tp, nh4, no3, no2, outflow):
        """
        This function is used to calculate unit eutrophication potential, kg PO4-eq/day.
        :param outflow: m3/d.
        :param tn: effluent total nitrogen.
        :param tp: effluent total phosphorus.
        :param cod: effluent cod.
        :param nh4: effluent ammonium.
        :param no3: effluent nitrate.
        :param no2: effluent nitrite.
        :return: daily eutrophication potential, a scalar, kg PO4-eq/d.
        """

        ep_tot = (3.07 * tp + 0.022 * cod + 0.33 * nh4 + 0.095 * no3 + 0.13 * no2) / 1000 * outflow

        return ep_tot

    def greenhouse_gas(self, process, energy, sludge_flow, sludge_tss, bod, tn, dosage,
                       outflow, sludge, bio_elec):
        """
        :param methane: methane offset, kg CO2-e/d;
        :param tn: total nitrogen, mg/L;
        :param sludge_tss: mean sludge TSS, mg/L (g/m3);
        :param outflow: m3/d;
        :param bod: bod, mg/L (g/m3);
        :param process: unknown unit kg CO2-e/d;
        :param energy: kWh/d.
        :param sludge_flow: m3/d.
        :return:
        """
        dosage_for_sludge = sludge_tss * sludge_flow / 1000 * 30 * 0.4 / 1000  # kg FeCl3/d
        dosage_for_tp = dosage * 0.25  # kg PAC/d

        ghg_tot = process + 1.17 * energy + 0.000192 * (dosage_for_sludge + dosage_for_tp + sludge) + \
            0.986 * dosage_for_sludge + 1.182 * dosage_for_tp + bod * outflow / 1000 * 0.25 * 0.035 * 25 + \
                  298 * 0.016 * tn * outflow / 1000 * 44 / 28 - bio_elec * 24 * 1.17

        return ghg_tot

    def min_max(self, data):
        cost_list = []
        energy_list = []
        eutro_list = []
        ghg_list = []

        for today_effluent in data.values:
            # unit conversion
            onsite_energy = today_effluent[24]
            aeration_power = today_effluent[29]
            recover_heat = today_effluent[31] / 1e8
            sludge_tss = today_effluent[21]
            sludge_flow = today_effluent[22]
            dosage = today_effluent[12] / 1000
            bio_elec = today_effluent[25] / 24
            sludge = today_effluent[23] / 1000
            outflow = today_effluent[26]

            process_ghg = today_effluent[27] / 1000
            methane_offset = today_effluent[28] / 1000

            energy_consumption = self.energy_consumption(onsite_energy=onsite_energy, aeration_power=aeration_power,
                                                            recover_heat=recover_heat,
                                                            sludge_tss=sludge_tss, sludge_flow=sludge_flow,
                                                            dosage=dosage, bio_elec=bio_elec)

            cost = self.cost(total_energy=energy_consumption, sludge_tss=sludge_tss, sludge_flow=sludge_flow,
                                dosage=dosage, sludge=sludge, bio_elec=bio_elec)

            cod = today_effluent[13]
            bod = today_effluent[14]
            tn = today_effluent[15]
            tp = today_effluent[16]
            nh4 = today_effluent[17]
            no3 = today_effluent[18]
            no2 = today_effluent[19]
            ss = today_effluent[20]

            eutro_po = self.eutrophication_potential(cod=cod, tp=tp, nh4=nh4, no3=no3, no2=no2, outflow=outflow)

            ghg = self.greenhouse_gas(process=process_ghg, energy=energy_consumption, sludge_flow=sludge_flow,
                                         sludge_tss=sludge_tss, bod=bod, tn=tn, dosage=dosage,
                                         outflow=outflow, sludge=sludge, bio_elec=bio_elec)

            energy_list.append(energy_consumption)
            cost_list.append(cost)
            eutro_list.append(eutro_po)
            ghg_list.append(ghg)

        energy_max = sorted(energy_list)[int(len(energy_list))-1]
        energy_min = min(energy_list)

        cost_max = sorted(cost_list)[int(len(cost_list))-1]
        cost_min = min(cost_list)

        eutro_max = sorted(eutro_list)[int(len(eutro_list))-1]
        eutro_min = min(eutro_list)

        ghg_max = sorted(ghg_list)[int(len(ghg_list))-1]
        ghg_min = min(ghg_list)

        return energy_max, energy_min, cost_max, cost_min, eutro_max, eutro_min, ghg_max, ghg_min

    @staticmethod
    def normalization(data, min, max):
        if data < min:
            return 0
        # elif data > max:
        #     return 1
        else:
            normalized_data = (data - min) / (max - min)

        return normalized_data


