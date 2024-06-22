from __future__ import division
import time
import math
import random
from copy import deepcopy
import numpy as np
from models import ValueNet
import torch
model_path = './saved_models/supervised_best_op_0615_k4_1.pt'

# predictionNet = ValueNet(856+3, 24)
predictionNet = ValueNet(1616, 16)
predictionNet.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
predictionNet.eval()


def getReward(state):
    inputState = torch.tensor(state.board + state.predicatesEncode, dtype=torch.float32)
    with torch.no_grad():
        predictionRuntime = predictionNet(inputState)  # 预测完整连接顺序得到的标签
    prediction = predictionRuntime.detach().cpu().numpy()
    maxindex = np.argmax(prediction)
    reward = (4 - maxindex/4) / 4.0
    return reward


def randomPolicy(state):
    while not state.isTerminal():
        try:
            temp = state.getPossibleActions()
            action = random.choice(temp)
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        state = state.takeAction(action)  # 迭代or递归？
    # reward = state.getReward()
    reward = getReward(state)
    # print(reward)
    return reward

# 保存当前节点状态
class treeNode():
    def __init__(self, state, parent):
        self.state = state
        self.isTerminal = state.isTerminal()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}


class mcts():
    def __init__(self, iterationLimit=None, explorationConstant=1 / math.sqrt(2),
                 rolloutPolicy=randomPolicy):
        if iterationLimit == None:
            raise ValueError("Must have either a time limit or an iteration limit")
        # number of iterations of the search
        if iterationLimit < 1:
            raise ValueError("Iteration limit must be greater than one")  # 极限要大于1
        self.searchLimit = iterationLimit
        self.explorationConstant = explorationConstant  #
        self.rollout = rolloutPolicy  # rollout方法将在选择的节点上随机执行一种Action

    # 对应模拟
    def search(self, initialState):
        self.root = treeNode(initialState, None)  # 第一个是state(进来的是planstate类型的数据),第二个是parent
        for i in range(self.searchLimit):  # 运行searchLimit次的executeRound
            self.executeRound()

        bestChild = self.getBestChild(self.root, 0)
        # print(self.getAction(self.root, bestChild))  (26, 6)  (9, 16)...
        return self.getAction(self.root, bestChild)  # 返回的是

    # 一次模拟流程
    def executeRound(self):
        node = self.selectNode(self.root)
        newState = deepcopy(node.state)
        reward = self.rollout(newState)
        self.backpropogate(node, reward)

    # 节点选择
    # 该节点若有子节点,则使用getBestChild方法获得UCT值最大的节点
    # 若无子节点,则使用expand方法扩展子节点
    def selectNode(self, node):
        while not node.isTerminal:
            if node.isFullyExpanded:
                node = self.getBestChild(node, self.explorationConstant)
            else:
                return self.expand(node)
        return node

    def expand(self, node):
        actions = node.state.getPossibleActions()
        for action in actions:
            if action not in node.children:
                newNode = treeNode(node.state.takeAction(action), node)
                node.children[action] = newNode
                # print(action)
                # print(newNode.state)
                if len(actions) == len(node.children):
                    node.isFullyExpanded = True
                # if newNode.isTerminal:
                #     print(newNode)
                return newNode

        raise Exception("Should never reach here")

    def backpropogate(self, node, reward):
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

    # 在n次executeRound执行完后,选择子节点中最优的
    def getBestChild(self, node, explorationValue):
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
            # print(child.totalReward)
            # print(node.numVisits)
            # print(child.numVisits)
            nodeValue = child.totalReward / child.numVisits + explorationValue * math.sqrt(
                2 * math.log(node.numVisits) / child.numVisits)
            # print(nodeValue)
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        return random.choice(bestNodes)

    # 从子节点中获取其动作(下到哪里)
    def getAction(self, root, bestChild):
        # print(root.children)
        for action, node in root.children.items():
            if node is bestChild:
                # print(action)
                # print(node)
                return action
