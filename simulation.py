#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2024/1/17 14:33
@Author  : Tengdi Zheng
@File    : simulation.py
"""

from alternatives import one_point_estimation
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from data_generation import *
from GWGPL import GWGPL
import os

np.random.seed(RANDOM_STATE)

''' Simulation Settings '''
datatype = 'continue'  # data type
p1, p2, p_noise = 5, 0, 5  # number of continuous covariates, discrete covariates and noise covariates
p, l, sample_average = p1 + p2 + p_noise, 10, 30  # number of covariates, number of locations, sample average
n = np.random.normal(sample_average, sample_average / 6, l).astype(int)  # generate samples
repeat_number, sigma = 100, 1/3  # number of repetitions and noise level
RHO, lt, sl = [0, 0.2, 0.5], ['dense', 'disperse', 'line'], ['low', 'high'] # correlation coefficients, location types, signal types
''''''


''' Simulation beginning '''
for location_type in lt:
    if location_type == 'dense':
        g_coords = np.array([
            [0.44201802, 0.56235160],
            [0.40734233, 0.57933696],
            [0.60023565, 0.565987510],
            [0.53526954, 0.523568520],
            [0.43021354, 0.415462460],
            [0.45239755, 0.512314650],
            [0.55423135, 0.454313250],
            [0.50232135, 0.422531560],
            [0.46542356, 0.542351540],
            [0.58456213, 0.421536970]])

        # l is 20
        # g_coords = np.array([
        #     [0.44201802, 0.56235160],
        #     [0.40734233, 0.57933696],
        #     [0.60023565, 0.565987510],
        #     [0.53526954, 0.523568520],
        #     [0.43021354, 0.415462460],
        #     [0.45239755, 0.512314650],
        #     [0.55423135, 0.454313250],
        #     [0.50232135, 0.422531560],
        #     [0.46542356, 0.542351540],
        #     [0.58456213, 0.421536970],
        #     [0.40562389, 0.479236465],
        #     [0.42256963, 0.539462346],
        #     [0.47765852, 0.462351321],
        #     [0.50946365, 0.569423132],
        #     [0.52247896, 0.462513645],
        #     [0.55264352, 0.413254612],
        #     [0.57512346, 0.567432152],
        #     [0.60321987, 0.459621321],
        #     [0.57896546, 0.512646879],
        #     [0.43621546, 0.454986431]])
    elif location_type == 'disperse':
        g_coords = np.array([
            [0.02375981, 0.08825333],
            [0.03665332, 0.60964299],
            [0.32523272, 0.88500807],
            [0.08494575, 0.25065348],
            [0.74082562, 0.93311949],
            [0.90742148, 0.66599465],
            [0.93718995, 0.83749397],
            [0.92127660, 0.34535785],
            [0.73589767, 0.06708488],
            [0.52188345, 0.04101815]])

        # l is 20
        # g_coords = np.array([
        #     [0.02375981, 0.08825333],
        #     [0.03665332, 0.60964299],
        #     [0.32523272, 0.88500807],
        #     [0.08494575, 0.25065348],
        #     [0.74082562, 0.93311949],
        #     [0.90742148, 0.66599465],
        #     [0.93718995, 0.83749397],
        #     [0.92127660, 0.34535785],
        #     [0.73589767, 0.06708488],
        #     [0.52188345, 0.04101815],
        #     [0.16583648, 0.80654134],
        #     [0.52646797, 0.96136464],
        #     [0.73649714, 0.82334648],
        #     [0.96167314, 0.52479646],
        #     [0.81324963, 0.20264679],
        #     [0.31479649, 0.12437373],
        #     [0.19367497, 0.16971343],
        #     [0.04364866, 0.48316879],
        #     [0.16319466, 0.77167964],
        #     [0.90649876, 0.18464796]])
    elif location_type == 'line':
        g_coords = np.array([
             [0.80480382, 0.87863867],
             [0.21890399, 0.23217517],
             [0.31830656, 0.3364209 ],
             [0.96341663, 1.02237454],
             [0.69402776, 0.71589313],
             [0.52767575, 0.58523621],
             [0.59098828, 0.64109221],
             [0.05787263, 0.09492604],
             [0.00228143, 0.04646661],
             [0.90860972, 0.97848101]])

        # l is 20
        # g_coords = np.array([
        #     [0.80480382, 0.87863867],
        #     [0.21890399, 0.23217517],
        #     [0.31830656, 0.3364209],
        #     [0.96341663, 0.99237454],
        #     [0.69402776, 0.71589313],
        #     [0.52767575, 0.58523621],
        #     [0.59098828, 0.64109221],
        #     [0.05787263, 0.09492604],
        #     [0.00228143, 0.04646661],
        #     [0.90860972, 0.97848101],
        #     [0.15974645, 0.13647979],
        #     [0.19464316, 0.23647946],
        #     [0.39134674, 0.42446797],
        #     [0.43647949, 0.40326997],
        #     [0.61347645, 0.59346446],
        #     [0.79136454, 0.77146789],
        #     [0.84316479, 0.84664644],
        #     [0.45316479, 0.45914646],
        #     [0.13649799, 0.17646799],
        #     [0.79346974, 0.81469794]])
    for signal_level in sl:
        for rho in RHO:
            # save the coefficient estimation results for each simulation and the overall estimation under different settings
            output_dir = 'SIM/' + 'data_type=' + datatype + ' location_type=' + location_type + ' signal_level=' \
                         + signal_level + ' p1, p2, p_noise=' + str(p1) + ',' + str(p2) + ',' + str(
                p_noise) + ' rho, sigma=' \
                         + str(rho) + ',' + str(sigma) + ',n=' + str(
                sample_average) + ',l=' + str(l)
            if output_dir[-1] != "/":
                output_dir += "/"
            if not os.path.isdir(output_dir):
                print("Directory doesn't exist, creating it")
                os.mkdir(output_dir)

            ''' Initializes the related settings '''
            repeat_data, beta, threshold = generate_repeat_data(repeat_number, datatype, rho, sigma, signal_level, p1, p2, p_noise, l, g_coords, n)
            loss = np.zeros([5, repeat_number])  # used to store the error of each method in repeated experiments
            coef_loss = np.zeros([5, repeat_number])  # used to store the coefficient estimation errors of each method in repeated experiments
            coef_selection_n2, coef_selection_n4= np.zeros([3, repeat_number]), np.zeros([3, repeat_number])  # used to store variable selection
            b_min, b_max = bw_selection(g_coords)
            bw = np.linspace(b_min + 0.01, b_max * 2, 10) # give the bandwidth range
            lamda = np.linspace(0.1, 15, 30) # give the lambda range

            ''' Start repeating the experiment '''
            for repeat in range(repeat_number):
                try:
                    loss_par_lamda = np.zeros([5, len(lamda)])
                    loss_par_bw = np.zeros([5, len(bw)])

                    data_generate = repeat_data[repeat]
                    split1 = StratifiedShuffleSplit(n_splits=1, test_size=0.33, random_state=RANDOM_STATE)
                    split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=RANDOM_STATE)
                    for train_index, test_index in split1.split(data_generate, data_generate["loc_x"]): # divided by location
                        train_o, test_o = data_generate.iloc[train_index, :], data_generate.iloc[test_index, :]
                        X_train_o, y_train_o = train_o.iloc[:, 0:p + 2], train_o.iloc[:, p + 2]
                        X_test_o, y_test_o = test_o.iloc[:, 0:p + 2], test_o.iloc[:, p + 2]
                    for train_index, test_index in split2.split(train_o, train_o["loc_x"]): # divided by location
                        train_c, test_c = train_o.iloc[train_index, :], train_o.iloc[test_index, :]
                        X_train_x, y_train_x = train_c.iloc[:, 0:p + 2], train_c.iloc[:, p + 2]
                        X_test_c, y_test_c = test_c.iloc[:, 0:p + 2], test_c.iloc[:, p + 2]
                    X_train_x, X_test_c, X_test_o, y_train_x, y_test_c, y_test_o = np.array(X_train_x), np.array(X_test_c), \
                        np.array(X_test_o), np.array(y_train_x), np.array(y_test_c), np.array(y_test_o)

                    for i in range(len(lamda)):
                        lamd, gwr_bw = lamda[i], b_max / 2
                        print("lamd=", lamd, "bw=", gwr_bw, ",calcaulating...")
                        Aerfa = Kernel(g_coords, gwr_bw, 'bi-square')
                        lsq_loss, weight_loss, lasso_loss, weight_lasso_loss, sim_loss = [], [], [], [], []
                        simultaneous_model = GWGPL(lamd=lamd, tol=1e-3, max_iter=1000, threshold=threshold)
                        simultaneous_model.fit(X_train_x, y_train_x, Aerfa, g_coords, p, l)
                        for loc in range(len(g_coords)):
                            aerfa = Aerfa.iloc[:, loc]
                            XX_test, yy_test = add_intercept(X_test_c[list((X_test_c[:, 0] == g_coords[loc][0])&(X_test_c[:, 1] == g_coords[loc][1])), :][:, 2:]), \
                                y_test_c[list((X_test_c[:, 0] == g_coords[loc][0]) & (X_test_c[:, 1] == g_coords[loc][1]))]
                            if len(XX_test) == 0:  # if there is no data for the location, no prediction is made
                                continue
                            #########################OLS###################################
                            lsq_model = one_point_estimation(add_weight=False, lamd=0, tol=1e-10, max_iter=1000, threshold=threshold)
                            lsq_model.fit(X_train_x, y_train_x, g_coords, loc, aerfa)
                            y_lsq = lsq_model.predict(XX_test)
                            lsq_loss.append(metrics.mean_squared_error(yy_test, y_lsq) ** 0.5)
                            ##########################GWR##################################
                            weight_model = one_point_estimation(add_weight=True,  lamd=0, tol=1e-10, max_iter=1000, threshold=threshold)
                            weight_model.fit(X_train_x, y_train_x, g_coords, loc, aerfa)
                            y_weight = weight_model.predict(XX_test)
                            weight_loss.append(metrics.mean_squared_error(yy_test, y_weight) ** 0.5)
                            #########################Lasso##################################
                            lasso_model = one_point_estimation(add_weight=False, lamd=lamd, tol=1e-10, max_iter=1000, threshold=threshold)
                            lasso_model.fit(X_train_x, y_train_x, g_coords, loc, aerfa)
                            y_lsq_lasso = lasso_model.predict(XX_test)
                            lasso_loss.append(metrics.mean_squared_error(yy_test, np.array(y_lsq_lasso)) ** 0.5)
                            ##########################GWL###################################
                            weight_lasso_model = one_point_estimation(add_weight=True, lamd=lamd, tol=1e-10, max_iter=1000, threshold=threshold)
                            weight_lasso_model.fit(X_train_x, y_train_x, g_coords, loc, aerfa)
                            y_weight_lasso = weight_lasso_model.predict(XX_test)
                            weight_lasso_loss.append(metrics.mean_squared_error(yy_test, y_weight_lasso) ** 0.5)
                            #########################GWGPL#################################
                            y_sim = simultaneous_model.predict(XX_test, loc, p)
                            sim_loss.append(metrics.mean_squared_error(yy_test, np.array(y_sim)) ** 0.5)
                        loss_par_lamda[0, i] = np.average(lsq_loss) if len(lsq_loss) != 0 else 0
                        loss_par_lamda[1, i] = np.average(weight_loss) if len(weight_loss) != 0 else 0
                        loss_par_lamda[2, i] = np.average(lasso_loss) if len(lasso_loss) != 0 else 0
                        loss_par_lamda[3, i] = np.average(weight_lasso_loss) if len(weight_lasso_loss) != 0 else 0
                        loss_par_lamda[4, i] = np.average(sim_loss) if len(sim_loss) != 0 else 0

                    for i in range(len(bw)):
                        lamd1, lamd2 = lamda[np.argmin(loss_par_lamda[3, :])], lamda[np.argmin(loss_par_lamda[4, :])]
                        gwr_bw = bw[i]
                        print("lamd1=", lamd1, "lamd2=", lamd2, "bw=", gwr_bw, ",calcaulating...")
                        Aerfa = Kernel(g_coords, gwr_bw, 'bi-square')

                        weight_loss, weight_lasso_loss, sim_loss = [], [], []
                        simultaneous_model = GWGPL(lamd=lamd2, tol=1e-10, max_iter=1000, threshold=threshold)
                        simultaneous_model.fit(X_train_x, y_train_x, Aerfa, g_coords, p, l)
                        for loc in range(len(g_coords)):
                            aerfa = Aerfa.iloc[:, loc]
                            XX_test, yy_test = add_intercept(X_test_c[list((X_test_c[:, 0] == g_coords[loc][0]) & (X_test_c[:, 1] == g_coords[loc][1])), :][:, 2:]), \
                                               y_test_c[list((X_test_c[:, 0] == g_coords[loc][0]) & (X_test_c[:, 1] == g_coords[loc][1]))]
                            if len(XX_test) == 0:  # if there is no data for the location, no prediction is made
                                continue
                            ##########################GWR##################################
                            weight_model = one_point_estimation(add_weight=True,lamd=0, tol=1e-10, max_iter=1000, threshold=threshold)
                            weight_model.fit(X_train_x, y_train_x, g_coords, loc, aerfa)
                            y_weight = weight_model.predict(XX_test)
                            weight_loss.append(metrics.mean_squared_error(yy_test, y_weight) ** 0.5)
                            ############################GWL##################################
                            weight_lasso_model = one_point_estimation(add_weight=True, lamd=lamd1, tol=1e-10, max_iter=1000, threshold=threshold)
                            weight_lasso_model.fit(X_train_x, y_train_x, g_coords, loc, aerfa)
                            y_weight_lasso = weight_lasso_model.predict(XX_test)
                            weight_lasso_loss.append(metrics.mean_squared_error(yy_test, y_weight_lasso) ** 0.5)
                            ###########################GWGPL################################
                            y_sim = simultaneous_model.predict(XX_test, loc, p)
                            sim_loss.append(metrics.mean_squared_error(yy_test, np.array(y_sim)) ** 0.5)
                        loss_par_bw[1, i] = np.average(weight_loss) if len(weight_loss) != 0 else 0
                        loss_par_bw[3, i] = np.average(weight_lasso_loss) if len(weight_lasso_loss) != 0 else 0
                        loss_par_bw[4, i] = np.average(sim_loss) if len(sim_loss) != 0 else 0


                    print("The best parameters selected in this simulation：")
                    print("GWR_model (The best bw):", bw[np.argmin(loss_par_bw[1, :])])
                    print("Lasso_model (The best lamda):", lamda[np.argmin(loss_par_lamda[2, :])])
                    print("GWL_model (The best lamda):", lamda[np.argmin(loss_par_lamda[3, :])], "(The best bw):", bw[np.argmin(loss_par_bw[3, :])])
                    print("GWGPL_model (The best lamda):", lamda[np.argmin(loss_par_lamda[4, :])], "(The best bw):", bw[np.argmin(loss_par_bw[4, :])])



                    ''' Testing set'''
                    lsq_coef_loss, weight_coef_loss, lasso_coef_loss, weight_lasso_coef_loss, sim_coef_loss = [], [], [], [], []
                    lsq_coef, weight_coef, lasso_coef, weight_lasso_coef = [], [], [], []
                    lasso_tpr, weight_lasso_tpr, sim_tpr = [], [], []
                    lasso_fpr, weight_lasso_fpr, sim_fpr = [], [], []
                    lsq_intercept, weight_intercept, lasso_intercept, weight_lasso_intercept = [], [], [], []
                    lasso_fp, weight_lasso_fp, sim_fp = 0, 0, 0
                    lasso_tp, weight_lasso_tp, sim_tp = 0, 0, 0
                    ########################GWGPL#############################
                    gwr_bw = bw[np.argmin(loss_par_bw[4, :])]
                    Aerfa = Kernel(g_coords, gwr_bw, 'bi-square')
                    simultaneous_model = GWGPL(lamd=lamda[np.argmin(loss_par_lamda[4, :])], tol=1e-10, max_iter=1000, threshold=threshold)
                    simultaneous_model.fit(X_train_x, y_train_x, Aerfa, g_coords, p, l)
                    for lst in simultaneous_model.coef_.reshape((l, p))[:, p1 + p2:].T.tolist():
                        for a in lst:
                            if abs(a) > 1e-6:
                                sim_fp += 1
                    for lst in simultaneous_model.coef_.reshape((l, p))[:, :p1 + p2].T.tolist():
                        for a in lst:
                            if abs(a) > 1e-6:
                                sim_tp += 1
                    for loc in range(len(g_coords)):
                        XX_test, yy_test = add_intercept(X_test_o[list((X_test_o[:, 0] == g_coords[loc][0]) & (X_test_o[:, 1] == g_coords[loc][1])), :][:,2:]), \
                                                         y_test_o[list((X_test_o[:, 0] == g_coords[loc][0]) & (X_test_o[:, 1] == g_coords[loc][1]))]
                        if len(XX_test) == 0:
                            continue

                    ########################GWL###############################
                    gwr_bw = bw[np.argmin(loss_par_bw[3, :])]
                    Aerfa = Kernel(g_coords, gwr_bw, 'bi-square')
                    for loc in range(len(g_coords)):
                        aerfa = Aerfa.iloc[:, loc]
                        XX_test, yy_test = add_intercept(X_test_o[list((X_test_o[:, 0] == g_coords[loc][0]) & (X_test_o[:, 1] == g_coords[loc][1])), :][:, 2:]), \
                                                         y_test_o[list((X_test_o[:, 0] == g_coords[loc][0]) & (X_test_o[:, 1] == g_coords[loc][1]))]
                        if len(XX_test) == 0:
                            continue

                        weight_lasso_model = one_point_estimation(add_weight=True, lamd=lamda[np.argmin(loss_par_lamda[3, :])], tol=1e-10, max_iter=1000, threshold=threshold)
                        weight_lasso_model.fit(X_train_x, y_train_x, g_coords, loc, aerfa)
                        weight_lasso_coef.append(weight_lasso_model.coef_.tolist()[0])
                        weight_lasso_intercept.append(weight_lasso_model.intercept_)
                        for a in weight_lasso_model.coef_[:, p1 + p2:].tolist()[0]:
                            if abs(a) > 1e-6:
                                weight_lasso_fp += 1
                        for a in weight_lasso_model.coef_[:, :p1 + p2].tolist()[0]:
                            if abs(a) > 1e-6:
                                weight_lasso_tp += 1

                    ########################OLS+GWR+Lasso####################################
                    gwr_bw = bw[np.argmin(loss_par_bw[1, :])]
                    Aerfa = Kernel(g_coords, gwr_bw, 'bi-square')
                    for loc in range(len(g_coords)):
                        aerfa = Aerfa.iloc[:, loc]
                        XX_test, yy_test = add_intercept(X_test_o[list((X_test_o[:, 0] == g_coords[loc][0]) & (X_test_o[:, 1] == g_coords[loc][1])), :][:,2:]), \
                                                         y_test_o[list((X_test_o[:, 0] == g_coords[loc][0]) & (X_test_o[:, 1] == g_coords[loc][1]))]
                        if len(XX_test) == 0:
                            continue
                        #########################OLS###################################
                        lsq_model = one_point_estimation(add_weight=False, lamd=0, tol=1e-10, max_iter=1000, threshold=threshold)
                        lsq_model.fit(X_train_x, y_train_x, g_coords, loc, aerfa)
                        lsq_coef.append(lsq_model.coef_.tolist()[0])
                        lsq_intercept.append(lsq_model.intercept_)
                        ##########################GWR##################################
                        weight_model = one_point_estimation(add_weight=True, lamd=0, tol=1e-10, max_iter=1000, threshold=threshold)
                        weight_model.fit(X_train_x, y_train_x, g_coords, loc, aerfa)
                        weight_coef.append(weight_model.coef_.tolist()[0])
                        weight_intercept.append(weight_model.intercept_)
                        ########################Lasso##################################
                        lasso_model = one_point_estimation(add_weight=False,  lamd=lamda[np.argmin(loss_par_lamda[2, :])], tol=1e-10, max_iter=1000, threshold=threshold)
                        lasso_model.fit(X_train_x, y_train_x, g_coords, loc, aerfa)
                        lasso_coef.append(lasso_model.coef_.tolist()[0])
                        lasso_intercept.append(lasso_model.intercept_)
                        for a in lasso_model.coef_[:, p1 + p2:].tolist()[0]:
                            if abs(a) > 1e-6:
                                lasso_fp += 1
                        for a in lasso_model.coef_[:, :p1 + p2].tolist()[0]:
                            if abs(a) > 1e-6:
                                lasso_tp += 1

                    print('***********Coefficient estimation error***********')
                    lsq_coef_loss.append(np.sum(np.absolute(np.array(lsq_coef)[:, :p1+p2] - beta[:, :p1+p2])) + np.sum(
                        np.absolute(np.array(lsq_coef)[:, p1+p2:])) + np.sum(
                        np.absolute(np.array(lsq_intercept) - beta[:, p1+p2])))
                    weight_coef_loss.append(np.sum(np.absolute(np.array(weight_coef)[:, :p1+p2] - beta[:, :p1+p2])) + np.sum(
                        np.absolute(np.array(weight_coef)[:, p1+p2:])) + np.sum(
                        np.absolute(np.array(weight_intercept) - beta[:, p1+p2])))
                    lasso_coef_loss.append(np.sum(np.absolute(np.array(lasso_coef)[:, :p1+p2] - beta[:, :p1+p2])) + np.sum(
                        np.absolute(np.array(lasso_coef)[:, p1+p2:])) + np.sum(
                        np.absolute(np.array(lasso_intercept) - beta[:, p1+p2])))
                    weight_lasso_coef_loss.append(
                        np.sum(np.absolute(np.array(weight_lasso_coef)[:, :p1+p2] - beta[:, :p1+p2])) + np.sum(
                            np.absolute(np.array(weight_lasso_coef)[:, p1+p2:])) + np.sum(
                            np.absolute(np.array(weight_lasso_intercept) - beta[:, p1+p2])))
                    sim_coef_loss.append(np.sum(
                        np.absolute(
                            np.array(simultaneous_model.coef_.reshape((l, p)))[:, :p1+p2] - beta[:, :p1+p2])) + np.sum(
                        np.absolute(np.array(simultaneous_model.coef_.reshape((l, p)))[:, p1+p2:])) +
                            np.sum(np.absolute(np.array(simultaneous_model.intercept_) - beta[:, p1+p2].reshape(-1, 1))))


                except:
                    continue

                print("Simulation", repeat + 1, "OLScl, GWRcl, Lassocl, GWLcl, GWGPLcl", end='')
                print(np.average(lsq_coef_loss), np.average(weight_coef_loss), np.average(lasso_coef_loss),
                      np.average(weight_lasso_coef_loss),
                      np.average(sim_coef_loss))
                coef_loss[0, repeat] = np.average(lsq_coef_loss)
                coef_loss[1, repeat] = np.average(weight_coef_loss)
                coef_loss[2, repeat] = np.average(lasso_coef_loss)
                coef_loss[3, repeat] = np.average(weight_lasso_coef_loss)
                coef_loss[4, repeat] = np.average(sim_coef_loss)
                coef_selection_n2[0, repeat] = lasso_fp
                coef_selection_n4[0, repeat] = lasso_tp
                coef_selection_n2[1, repeat] = weight_lasso_fp
                coef_selection_n4[1, repeat] = weight_lasso_tp
                coef_selection_n2[2, repeat] = sim_fp
                coef_selection_n4[2, repeat] = sim_tp

                # save the coefficient estimates for each simulation for all methods
                output_dir2 = output_dir + 'simulation' + str(repeat + 1)
                if output_dir2[-1] != "/":
                    output_dir2 += "/"
                if not os.path.isdir(output_dir2):
                    print("Directory doesn't exist, creating it")
                    os.mkdir(output_dir2)

                pd.DataFrame(np.concatenate((np.array(lsq_coef),np.array(lsq_intercept)), axis=1)).to_excel(output_dir2 +'OLS_coef' + ".xlsx")
                pd.DataFrame(np.concatenate((np.array(weight_coef), np.array(weight_intercept)), axis=1)).to_excel(
                    output_dir2 +'GWR_coef' + ".xlsx")
                pd.DataFrame(np.concatenate((np.array(lasso_coef), np.array(lasso_intercept).reshape(-1,1)), axis=1)).to_excel(
                    output_dir2 +'Lasso_coef' + ".xlsx")
                pd.DataFrame(np.concatenate((np.array(weight_lasso_coef), np.array(weight_lasso_intercept).reshape(-1,1)), axis=1)).to_excel(
                    output_dir2 +'GWL_coef' + ".xlsx")
                pd.DataFrame(np.concatenate((np.array(simultaneous_model.coef_.reshape((l, p)))
                                             , np.array(simultaneous_model.intercept_)), axis=1)).to_excel(output_dir2 +'GWGPL_coef' + ".xlsx")


            coef_average = coef_loss.mean(axis=1)
            coef_std = coef_loss.std(axis=1)
            selection_n2 = coef_selection_n2 / (p_noise * l)
            selection_n2_average = selection_n2.mean(axis=1)
            selection_n2_std = selection_n2.std(axis=1)
            selection_n4 = coef_selection_n4 / ((p1 + p2) * l)
            selection_n4_average = selection_n4.mean(axis=1)
            selection_n4_std = selection_n4.std(axis=1)

            output_dir = 'SIM/'
            if output_dir[-1] != "/":
                output_dir += "/"
            if not os.path.isdir(output_dir):
                print("Directory doesn't exist, creating it")
                os.mkdir(output_dir)

            result = {}
            result['OLS_coef_average_loss(var)'] = [str(round(coef_average[0], 3)) + ' (' + str(round(coef_std[0], 3)) + ')']
            result['GWR_coef_average_loss(var)'] = [str(round(coef_average[1], 3)) + ' (' + str(round(coef_std[1], 3)) + ')']
            result['Lasso_coef_average_loss(var)'] = [str(round(coef_average[2], 3)) + ' (' + str(round(coef_std[2], 3)) + ')']
            result['GWL_coef_average_loss(var)'] = [str(round(coef_average[3], 3)) + ' (' + str(round(coef_std[3], 3)) + ')']
            result['GWGPL_coef_average_loss(var)'] = [str(round(coef_average[4], 3)) + ' (' + str(round(coef_std[4], 3)) + ')']
            result['Lasso_tpr(var)'] = [str(round(selection_n4_average[0], 3)) + ' (' + str(round(selection_n4_std[0], 3)) + ')']
            result['GWL_tpr(var)'] = [str(round(selection_n4_average[1], 3)) + ' (' + str(round(selection_n4_std[1], 3)) + ')']
            result['GWGPL_tpr(var)'] = [str(round(selection_n4_average[2], 3)) + ' (' + str(round(selection_n4_std[2], 3)) + ')']
            result['Lasso_fpr(var)'] = [str(round(selection_n2_average[0], 3)) + ' (' + str(round(selection_n2_std[0], 3)) + ')']
            result['GWL_fpr(var)'] = [str(round(selection_n2_average[1], 3)) + ' (' + str(round(selection_n2_std[1], 3)) + ')']
            result['GWGPL_fpr(var)'] = [str(round(selection_n2_average[2], 3)) + ' (' + str(round(selection_n2_std[2], 3)) + ')']
            pd.DataFrame(result).to_excel(
                output_dir + "Result, " + 'data_type=' + datatype + ' location_type=' + location_type + ' signal_level='
                + signal_level + ' p1, p2, p_noise=' + str(p1) +','+ str(p2) +','+ str(p_noise) + ' rho, sigma='
                + str(rho) +','+ str(sigma) + ',n=' + str(sample_average) + ',l=' + str(l) + ".xlsx")