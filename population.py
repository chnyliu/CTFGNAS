from individual import Individual
import scipy.stats as stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from network import Network
from model_manager import ModelManager
from utils import *
import random
from genotypes import LayerAggregator_type, Activation_type, Correlation_coefficients, NodeAggregator_type
import time
from fine_tune import run_fine_tune


class Evolution_Trainer(object):
    def __init__(self, args, log):
        self.args = args
        self.log = log
        self.population = []
        self.offspring = []
        self.gen_num = 0
        self.eva_indi_arc = []  # evaluated individuals -> Selection
        self.eva_genoBin_sim_loss_test_arc = []  # [genoBin_sim, val_loss, test_acc] of evaluated individuals
        self.eva_genoDec_loss_arc = []  # [genoDec, val loss] of evaluated individuals -> Surrogate Model
        self.eva_min = [1e+10, 1e+10, 1e+10]
        self._initialize_population()
        self.pre_acc_arc = []  # accuracy of predictor

    def _initialize_population(self):
        self.log.info('{}: args = {}'.format(time.strftime('%Y%m%d-%H%M%S'), self.args))
        self.model_manager = ModelManager(self.args, self.log)
        self.log.info('{}: Begin to initial the population.'.format(time.strftime('%Y%m%d-%H%M%S')))
        bad_times = 0
        while len(self.population) < self.args.init_size:
            self.args.indi_no = len(self.population) + 1
            individual = Individual(self.args)
            temp = [x[0] for x in self.eva_genoBin_sim_loss_test_arc]
            if individual.genoBin_sim in temp and bad_times < self.args.init_size:
                bad_times += 1
            else:
                gnn = self.form_gnn_info(individual.genoBin_sim)
                individual.val_loss, individual.test_acc = self.model_manager.train(gnn)
                if individual.val_loss < self.eva_min[0]:
                    self.eva_min = [individual.val_loss, individual.test_acc, individual.genoBin_sim]
                self.population.append(individual)
                self.eva_indi_arc.append(individual)
                self.eva_genoBin_sim_loss_test_arc.append([individual.genoBin_sim, individual.val_loss, individual.test_acc])
                self.eva_genoDec_loss_arc.append([individual.genoDec, individual.val_loss])
        self.log.info('\t bad times = {}'.format(bad_times))
        self.log.info('\t Best individual: val loss:{}, acc:{}, genotype:{}'.format(self.eva_min[0], self.eva_min[1], self.eva_min[2]))

    def form_gnn_info(self, genotype):
        return Network(self.args, self.log, genotype)

    def train(self):
        for gen in range(self.args.max_gen):
            self.gen_num = gen + 1
            self.log.info('{}: Generate the {}-gen population.'.format(time.strftime('%Y%m%d-%H%M%S'), self.gen_num))
            self._crossover_mutation_and_repairment()
            self._predict()
            self._selection()
        middle_time = time.strftime('%Y%m%d-%H%M%S')
        top_loss, top_acc, top_geno = self.derive()
        return middle_time, top_loss, top_acc, top_geno, self.pre_acc_arc

    def _predict(self):
        inputs = np.array([x[0] for x in self.eva_genoDec_loss_arc])
        targets = np.array([x[1] for x in self.eva_genoDec_loss_arc])

        mlp_predictor = MLPRegressor(random_state=1, max_iter=5000)
        mlp_predictor.fit(inputs, targets)
        mlp_prediction = mlp_predictor.predict(inputs)
        mlp_rho, _ = stats.spearmanr(mlp_prediction, targets)

        gbdt_predictor = GradientBoostingRegressor()
        gbdt_predictor.fit(inputs, targets)
        gbdt_prediction = gbdt_predictor.predict(inputs)
        gbdt_rho, _ = stats.spearmanr(gbdt_prediction, targets)

        rf_predictor = RandomForestRegressor()
        rf_predictor.fit(inputs, targets)
        rf_prediction = rf_predictor.predict(inputs)
        rf_rho, _ = stats.spearmanr(rf_prediction, targets)

        if mlp_rho >= gbdt_rho and mlp_rho >= rf_rho:
            predictor = mlp_predictor
            prediction = mlp_prediction
            self.log.info("\t Select {}-predictor, Spearman's Rho = {:.4f}".format('MLP', mlp_rho))
        elif gbdt_rho >= mlp_rho and gbdt_rho >= rf_rho:
            predictor = gbdt_predictor
            prediction = gbdt_prediction
            self.log.info("\t Select {}-predictor, Spearman's Rho = {:.4f}".format('GBDT', gbdt_rho))
        else:
            predictor = rf_predictor
            prediction = rf_prediction
            self.log.info("\t Select {}-predictor, Spearman's Rho = {:.4f}".format('RF', rf_rho))
        temp = [x[0] for x in self.eva_genoBin_sim_loss_test_arc]
        for individual in self.offspring:
            if individual.genoBin_sim in temp:
                index = temp.index(individual.genoBin_sim)
                individual.val_loss = self.eva_genoBin_sim_loss_test_arc[index][1]
            else:
                individual.val_loss = predictor.predict(np.array([individual.genoDec])).item()
        selected_index = self._select_index()
        self.args.indi_no = 1
        for index in selected_index:
            indi = self.offspring[index]
            temp = [x[0] for x in self.eva_genoBin_sim_loss_test_arc]
            if indi.genoBin_sim in temp:
                indi_index = temp.index(indi.genoBin_sim)
                indi.val_loss = self.eva_genoBin_sim_loss_test_arc[indi_index][1]
                self.log.info('\t No.{:02d} has exist, val_loss: {:.4f}.'.format(self.args.indi_no, indi.val_loss))
            else:
                prediction = np.concatenate((prediction, [indi.val_loss]))
                gnn = self.form_gnn_info(indi.genoBin_sim)
                indi.val_loss, indi.test_acc = self.model_manager.train(gnn)
                if indi.val_loss < self.eva_min[0]:
                    self.eva_min = [indi.val_loss, indi.test_acc, indi.genoBin_sim]
                targets = np.concatenate((targets, [indi.val_loss]))

                self.eva_indi_arc.append(indi)
                self.eva_genoBin_sim_loss_test_arc.append([indi.genoBin_sim, indi.val_loss, indi.test_acc])
                self.eva_genoDec_loss_arc.append([indi.genoDec, indi.val_loss])
            self.args.indi_no += 1
        rmse = np.sqrt(((prediction - targets) ** 2).mean())
        rho, _ = stats.spearmanr(prediction, targets)
        tau, _ = stats.kendalltau(prediction, targets)
        self.pre_acc_arc.append(rho)
        self.log.info("\t RMSE = {:.4f}, Spearman's Rho = {:.4f}, Kendall's Tau = {:.4f}".format(rmse, rho, tau))
        self.log.info('\t Best individual: val loss:{}, acc:{}, genotype:{}'.format(self.eva_min[0], self.eva_min[1], self.eva_min[2]))

    def _crossover_mutation_and_repairment(self):
        self.offspring = []
        while len(self.offspring) < self.args.pop_size:
            self.args.indi_no = len(self.offspring) + 1
            index1, individual1 = choose_one_parent(self.population)
            index2, individual2 = choose_one_parent(self.population)
            while index2 == index1:
                index2, individual2 = choose_one_parent(self.population)
            parent1 = copy.deepcopy(individual1.genoBin)
            parent2 = copy.deepcopy(individual2.genoBin)
            if random.random() < self.args.pc:
                position = random.randint(1, self.args.max_len - 1)
                child = [parent1[:position] + parent2[position:self.args.max_len],
                         parent2[:position] + parent1[position:self.args.max_len]]
                for genoBin in child:
                    layerStatus = [1]
                    for index in range(position):
                        if isinstance(genoBin[index][0], list):
                            layerStatus.append(1)
                        else:
                            layerStatus.append(0)
                    for index in range(position, self.args.max_len):
                        # The non-empty layer guarantees that it continues to be non-empty
                        if isinstance(genoBin[index][0], list):
                            # If the front connection is broken, reset it.
                            genoBin[index][0] = binArrayRepair(genoBin[index][0], layerStatus)
                            while sum(genoBin[index][0]) == 0:
                                genoBin[index][0] = binArrayRepair(np.random.randint(0, 2, index + 1), layerStatus)
                            if sum(genoBin[index][0]) == 1:
                                genoBin[index][1] = 1
                            else:
                                if genoBin[index][1] == 1:
                                    genoBin[index][1] = random.randint(2, len(LayerAggregator_type) + 1)
                            layerStatus.append(1)
                        else:
                            layerStatus.append(0)
                    # Make sure there are layers behind it to connect it
                    for index in range(position):
                        if isinstance(genoBin[index][0], list):
                            temp = 0
                            while temp == 0 and index < self.args.max_len - 1:
                                for j in range(index + 1, self.args.max_len):
                                    if isinstance(genoBin[j][0], list):
                                        temp += genoBin[j][0][index + 1]
                                if temp == 0:
                                    for j in range(index + 1, self.args.max_len):
                                        if isinstance(genoBin[j][0], list):
                                            genoBin[j][0][index + 1] = random.randint(0, 1)
                                            temp += genoBin[j][0][index + 1]
                                            if sum(genoBin[j][0]) > 1 and genoBin[j][1] == 1:
                                                genoBin[j][1] = random.randint(2, len(LayerAggregator_type) + 1)
                                        else:
                                            if j == self.args.max_len - 1 and temp == 0:
                                                genoBin[j][0] = [0 for _ in range(self.args.max_len)]
                                                genoBin[j][0][index + 1] = 1
                                                genoBin[j][1] = 1
                                                genoBin[j][2] = random.randint(1, len(Activation_type))
            else:
                child = [parent1, parent2]
            for genoBin in child:
                genoBin = self._mutation_and_repairment(genoBin)
                individual = Individual(self.args, genotype=genoBin)
                if len(self.offspring) < self.args.pop_size:
                    self.offspring.append(individual)
                else:
                    break

    def _mutation_and_repairment(self, genoBin0):
        genoBin = copy.deepcopy(genoBin0)
        layerStatus = [1]
        for i in range(self.args.max_len):  # the i-th layer
            if random.random() < self.args.pm:
                if isinstance(genoBin[i][0], list):  # non-empty layer
                    state = True
                    while state is True:   # Connection, Fusion, Activation, Coefficients, NodeAgg
                        mutation_type = random.randint(0, self.args.geno_len - 1)
                        if mutation_type == 0 and i < self.args.max_len - 1:
                            layerAggBinList = binArrayRepair(np.random.randint(0, 2, i + 1), layerStatus)
                            while layerAggBinList == genoBin[i][0]:
                                layerAggBinList = binArrayRepair(np.random.randint(0, 2, i + 1), layerStatus)
                            genoBin[i][0] = layerAggBinList
                            if sum(genoBin[i][0]) == 0 and i < self.args.max_len - 1:
                                genoBin[i] = [0, 0, 0, 0, 0]
                                state = False
                                layerStatus_temp = copy.deepcopy(layerStatus)
                                layerStatus_temp.append(0)
                                for index in range(i + 1, self.args.max_len):
                                    # The non-empty layer guarantees that it continues to be non-empty
                                    if isinstance(genoBin[index][0], list):
                                        genoBin[index][0] = binArrayRepair(genoBin[index][0], layerStatus_temp)
                                        while sum(genoBin[index][0]) == 0:
                                            genoBin[index][0] = binArrayRepair(np.random.randint(0, 2, index + 1),
                                                                               layerStatus_temp)
                                        if sum(genoBin[index][0]) == 1:
                                            genoBin[index][1] = 1
                                        else:
                                            if genoBin[index][1] == 1:
                                                genoBin[index][1] = random.randint(2, len(LayerAggregator_type) + 1)
                                        layerStatus_temp.append(1)
                                    else:
                                        layerStatus_temp.append(0)
                                # Make sure there are layers behind it to connect it
                                for index in range(i):
                                    if isinstance(genoBin[index][0], list):
                                        temp = 0
                                        while temp == 0 and index < self.args.max_len - 1:
                                            for j in range(index + 1, self.args.max_len):
                                                if isinstance(genoBin[j][0], list):
                                                    temp += genoBin[j][0][index + 1]
                                            if temp == 0:
                                                for j in range(index + 1, self.args.max_len):
                                                    if isinstance(genoBin[j][0], list):
                                                        genoBin[j][0][index + 1] = random.randint(0, 1)
                                                        temp += genoBin[j][0][index + 1]
                                                        if sum(genoBin[j][0]) > 1 and genoBin[j][1] == 1:
                                                            genoBin[j][1] = random.randint(2, len(LayerAggregator_type) + 1)
                                                    else:
                                                        if j == self.args.max_len - 1 and temp == 0:
                                                            genoBin[j][0] = [0 for _ in range(self.args.max_len)]
                                                            genoBin[j][0][i + 1] = 1
                                                            genoBin[j][1] = 1
                                                            genoBin[j][2] = random.randint(1, len(Activation_type))
                            elif sum(genoBin[i][0]) == 1:
                                genoBin[i][1] = 1
                                state = False
                            else:
                                if genoBin[i][1] == 1:
                                    genoBin[i][1] = random.randint(2, len(LayerAggregator_type) + 1)
                                    state = False
                        elif mutation_type == 1:
                            if genoBin[i][1] > 1:
                                genoBin[i][1] = different_random_num(2, len(LayerAggregator_type) + 1, genoBin[i][1])
                                state = False
                            elif genoBin[i][1] == 1:
                                state = True
                            else:
                                raise
                        elif mutation_type == 2:
                            genoBin[i][2] = different_random_num(1, len(Activation_type), genoBin[i][2])
                            state = False
                        elif mutation_type == 3:
                            if i < self.args.max_len - 1:
                                genoBin[i][3] = different_random_num(1, len(Correlation_coefficients), genoBin[i][3])
                                state = False
                        elif mutation_type == 4:
                            if i < self.args.max_len - 1:
                                genoBin[i][4] = different_random_num(1, len(NodeAggregator_type), genoBin[i][4])
                                state = False
                # empty layer
                else:
                    while genoBin[i][0] == 0 or sum(genoBin[i][0]) == 0:
                        # Connections
                        layerAggBinList = binArrayRepair(np.random.randint(0, 2, i + 1), layerStatus)
                        # Fusion
                        if sum(layerAggBinList) == 0:
                            continue
                        elif sum(layerAggBinList) == 1:
                            layerAggType = 1
                        else:
                            layerAggType = random.randint(2, len(LayerAggregator_type) + 1)
                        # Activation
                        layerAct = random.randint(1, len(Activation_type))
                        # Coefficients
                        nodeCC = random.randint(1, len(Correlation_coefficients))
                        # NodeAgg
                        nodeAgg = random.randint(1, len(NodeAggregator_type))

                        if i < self.args.max_len - 1:
                            genoBin[i] = [layerAggBinList, layerAggType, layerAct, nodeCC, nodeAgg]
                        else:
                            genoBin[i] = [layerAggBinList, layerAggType, layerAct]
                        temp = 0
                        while temp == 0 and i < self.args.max_len - 1:
                            for j in range(i + 1, self.args.max_len):
                                if isinstance(genoBin[j][0], list):
                                    genoBin[j][0][i + 1] = random.randint(0, 1)
                                    temp += genoBin[j][0][i + 1]
                                    if sum(genoBin[j][0]) > 1 and genoBin[j][1] == 1:
                                        genoBin[j][1] = random.randint(2, len(LayerAggregator_type) + 1)
                                else:
                                    if j == self.args.max_len - 1 and temp == 0:
                                        genoBin[j][0] = [0 for _ in range(self.args.max_len)]
                                        genoBin[j][0][i + 1] = 1
                                        genoBin[j][1] = 1
                                        genoBin[j][2] = random.randint(1, len(Activation_type))
            if isinstance(genoBin[i][0], list):
                layerStatus.append(1)
            else:
                layerStatus.append(0)
        return genoBin

    def _select_index(self):
        random_range = []
        selected_length = int(self.args.sn * 2)
        while len(random_range) < selected_length:
            num = random.randint(0, self.args.pop_size - 1)
            if num not in random_range:
                random_range.append(num)
        selected_index = []
        for i in range(0, selected_length, 2):
            if self.offspring[random_range[i]].val_loss <= self.offspring[random_range[i + 1]].val_loss:
                selected_index.append(random_range[i])
            else:
                selected_index.append(random_range[i + 1])
        return selected_index

    def _selection(self):
        populations = self.population + self.offspring
        random_range = []
        selected_length = int(self.args.pop_size * 2)
        while len(random_range) < selected_length:
            num = random.randint(0, selected_length - 1)
            if num not in random_range:
                random_range.append(num)
        selected_index = []
        for i in range(0, selected_length, 2):
            if populations[random_range[i]].val_loss <= populations[random_range[i + 1]].val_loss:
                selected_index.append(random_range[i])
            else:
                selected_index.append(random_range[i + 1])
        self.population, pop_objs = [], []
        for i in selected_index:
            self.population.append(populations[i])
            pop_objs.append(populations[i].val_loss)
        self.offspring = []

    def derive(self):
        fir_loss, sec_loss = 1e+7, 1e+8
        fir_index, sec_index = -1, -1
        for index, individual in enumerate(self.eva_indi_arc):
            if individual.val_loss < fir_loss:
                fir_loss, fir_index = individual.val_loss, index
        for index, individual in enumerate(self.population):
            if individual.val_loss < sec_loss:
                sec_loss, sec_index = individual.val_loss, index
        
        top_loss = self.eva_indi_arc[fir_index].val_loss
        top_acc = self.eva_indi_arc[fir_index].test_acc
        top_geno = self.eva_indi_arc[fir_index].genoBin_sim
        
        if self.population[sec_index].genoBin_sim != self.eva_indi_arc[fir_index].genoBin_sim:
            self.args.tune_genotype = self.population[sec_index].genoBin_sim
            self.args, top_loss, top_acc, top_geno = run_fine_tune(self.args, self.log, self.population[sec_index], top_loss, top_acc, top_geno)
        self.args.tune_genotype = self.eva_indi_arc[fir_index].genoBin_sim
        self.args, top_loss, top_acc, top_geno = run_fine_tune(self.args, self.log, self.eva_indi_arc[fir_index], top_loss, top_acc, top_geno)
        
        self.log.info('{}: [DERIVE] Best Results: val loss:{}, test acc:{}, genotype:{}'.format(time.strftime('%Y%m%d-%H%M%S'), top_loss, top_acc, top_geno))
        return top_loss, top_acc, top_geno
