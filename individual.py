from genotypes import LayerAggregator_type, Correlation_coefficients, NodeAggregator_type, Activation_type
from utils import *


class Individual(object):
    def __init__(self, args, genotype=None):
        self.args = args
        self.genoBin = genotype
        self.genoBin_sim = None
        self.genoDec = None      # Simpled Vector
        self.val_loss = None
        self.test_acc = None
        self._initialize()

    def _initialize(self):
        if self.genoBin is None:
            genoBin = []
            layerStatus = [1]
            for i in range(self.args.max_len):
                # Connections
                layerAggBinList = binArrayRepair(np.random.randint(0, 2, i+1), layerStatus)
                # Fusion
                if sum(layerAggBinList) == 0:
                    genoBin.append([0, 0, 0, 0, 0])
                    layerStatus.append(0)
                    continue
                elif sum(layerAggBinList) == 1:
                    layerAggType = 1
                else:
                    layerAggType = random.randint(2, len(LayerAggregator_type)+1)
                # Activation
                layerAct = random.randint(1, len(Activation_type))
                # Coefficients
                nodeCC = random.randint(1, len(Correlation_coefficients))
                # NodeAgg
                nodeAgg = random.randint(1, len(NodeAggregator_type))

                genoBin.append([layerAggBinList, layerAggType, layerAct, nodeCC, nodeAgg])
                layerStatus.append(1)
            self.genoBin, self.genoBin_sim, self.genoDec = genoRepair(genoBin, self.args.max_len)
        else:
            layerStatus = [1]
            for index in range(self.args.max_len):
                if isinstance(self.genoBin[index][0], list):
                    self.genoBin[index][0] = binArrayRepair(self.genoBin[index][0], layerStatus)
                    if sum(self.genoBin[index][0]) == 0:
                        self.genoBin[index] = [0, 0, 0, 0, 0]
                        layerStatus.append(0)
                        continue
                    elif sum(self.genoBin[index][0]) == 1:
                        self.genoBin[index][1] = 1
                    else:
                        if self.genoBin[index][1] <= 1:
                            self.genoBin[index][1] = random.randint(2, len(LayerAggregator_type) + 1)
                    layerStatus.append(1)
                else:
                    layerStatus.append(0)
            self.genoBin, self.genoBin_sim, self.genoDec = genoRepair(self.genoBin, self.args.max_len)
