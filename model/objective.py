class ObjectiveFunction(object):
    """
    This is the class used to generate objective function, objective function consists of economic term
    and LCA term (energy consumption, eutrophication potential and green house for now). Each part is calculated
    in one day. The final score is obtained through weighted sum of two terms, and used as reward in reinforcement
    learning.
    """
    @staticmethod
    def cost(onsite_energy, sludge_tss, sludge_flow, dosage, sludge, bio):
        """
        :param energy: kWh/d;
        :param sludge: kg/d;
        :return: scalar.
        """
        # 30为投加量
        cost_tot = 0.8 * onsite_energy + 0.6 * (sludge_tss * sludge_flow / 1000 * 30 * 0.4 / 1000 + sludge + dosage * 0.25) + \
                sludge_tss * sludge_flow / 1000 * 30 * 0.4 / 1000 * 1.7 + dosage * 2.5 * 0.25 - bio * 24 * 0.25

        return cost_tot

    @staticmethod
    def energy_consumption(onsite_energy, sludge_tss, sludge_flow, dosage):
        """
        :param bio: methane power, kw;
        :param sludge_tss: mean sludge TSS, mg/L (g/m3);
        :param sludge_flow: sludge flow, m3/d;
        :param onsite_energy: aeration and plump power, kw;

        :return: mean daily energy consumption, kWh/d.
        """
        energy_tot = 24 * onsite_energy + sludge_tss * sludge_flow / 1000 * 30 * 0.4 / 1000 * 3.4 + dosage * 0.25 * 1.94

        return energy_tot

    @staticmethod
    def eutrophication_potential(cod, bod, tn, tp, nh4, no3, no2, ss, outflow):
        """
        This function is used to calculate unit eutrophication potential, kg PO4-eq/m3.
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

    @staticmethod
    def greenhouse_gas(process, energy, sludge_flow, sludge_tss, bod, tn, dosage, methane, outflow):
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
        ghg_tot = process + 1.17 * energy + sludge_tss * sludge_flow / 1000 * 30 * 0.4 / 1000 * (0.986 + 0.000192 * 200) + \
            bod * outflow / 1000 * 0.25 * 0.035 * 25 + 298 * 0.016 * tn * outflow / 1000 * 44 / 28 + dosage * 0.25 * \
                  (1.182 + 0.000192 * 200)

        return ghg_tot


    @staticmethod
    def min_max(data):
        cost_list = []
        energy_list = []
        eutro_list = []
        ghg_list = []

        for today_effluent in data.values:
            sludge_tss = today_effluent[10]
            sludge_flow = today_effluent[11]
            sludge_pro = today_effluent[12] / 1000
            onsite_energy = today_effluent[13]
            bio = today_effluent[14] / 24
            outflow = today_effluent[15]
            process_ghg = today_effluent[16] / 1000
            methane_offset = today_effluent[17] / 1000
            dosage = today_effluent[1] / 1000

            energy_consumption = ObjectiveFunction.energy_consumption(onsite_energy=onsite_energy,
                                                                      sludge_tss=sludge_tss,
                                                                      sludge_flow=sludge_flow, dosage=dosage)

            cost = ObjectiveFunction.cost(onsite_energy=onsite_energy, sludge_tss=sludge_tss, sludge_flow=sludge_flow,
                                          sludge=sludge_pro, dosage=dosage, bio=bio)

            eutro_po = ObjectiveFunction.eutrophication_potential(today_effluent[2], today_effluent[3],
                                                                  today_effluent[4],
                                                                  today_effluent[5], today_effluent[6],
                                                                  today_effluent[7],
                                                                  today_effluent[8], today_effluent[9], outflow)
            ghg = ObjectiveFunction.greenhouse_gas(process=process_ghg, energy=energy_consumption,
                                                   sludge_flow=sludge_flow,
                                                   sludge_tss=sludge_tss, bod=today_effluent[1], tn=today_effluent[2],
                                                   dosage=dosage, methane=methane_offset, outflow=outflow)

            energy_list.append(energy_consumption)
            cost_list.append(cost)
            eutro_list.append(eutro_po)
            ghg_list.append(ghg)

        energy_max = sorted(energy_list)[int(len(energy_list))-1]
        energy_min = min(energy_list)

        cost_max = sorted(cost_list)[int(len(cost_list))-1]
        cost_min = min(cost_list)

        eutro_max = sorted(eutro_list)[int(0.5*len(cost_list))-1]
        eutro_min = min(eutro_list)

        ghg_max = sorted(ghg_list)[int(0.25*len(cost_list))-1]
        ghg_min = min(ghg_list)

        return energy_max, energy_min, cost_max, cost_min, eutro_max, eutro_min, ghg_max, ghg_min

    @staticmethod
    def normalization(data, min, max):
        normalized_data = abs((data - min) / (max - min))

        return normalized_data

    @staticmethod
    def weighted_sum(data_cost, data_energy, data_eutrophication, data_ghg,
                     w_cost=0.25, w_energy=0.25, w_eutrophication=0.25, w_ghg=0.25):
        reward = w_cost * data_cost + w_energy * data_energy + w_eutrophication * data_eutrophication + w_ghg * data_ghg

        return reward


