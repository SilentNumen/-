import math
import random

import numpy as np
import my_visualize

marks = ['AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AJ', 'AK', 'AL', 'AM', 'AN', 'AO',
         'AP', 'AQ', 'AR', 'AS', 'AT', 'AU', 'AV', 'AW', 'AX', 'AY', 'AZ', 'BA', 'BB', 'BC', 'BD', 'BE', 'BF', 'BG',
         'BH', 'BI', 'BJ', 'BK', 'BL', 'BM', 'BN', 'BO', 'BP', 'BQ', 'BR', 'BS', 'BT', 'BU', 'BV', 'BW', 'BX', 'BY',
         'BZ', 'CA', 'CB', 'CC', 'CD', 'CE', 'CF', 'CG', 'CH', 'CI', 'CJ', 'CK', 'CL', 'CM', 'CN', 'CO', 'CP', 'CQ',
         'CR', 'CS', 'CT', 'CU', 'CV', 'CW', 'CX', 'CY', 'CZ', 'DA', 'DB', 'DC', 'DD', 'DE', 'DF', 'DG', 'DH', 'DI',
         'DJ', 'DK', 'DL', 'DM', 'DN', 'DO', 'DP', 'DQ', 'DR', 'DS', 'DT', 'DU', 'DV', 'DW', 'DX', 'DY', 'DZ', 'EA',
         'EB', 'EC', 'ED', 'EE', 'EF', 'EG', 'EH', 'EI', 'EJ', 'EK', 'EL', 'EM', 'EN', 'EO', 'EP', 'EQ', 'ER', 'ES',
         'ET', 'EU', 'EV', 'EW', 'EX', 'EY', 'EZ', 'FA', 'FB', 'FC', 'FD', 'FE', 'FF', 'FG', 'FH', 'FI', 'FJ', 'FK',
         'FL', 'FM', 'FN', 'FO', 'FP', 'FQ', 'FR', 'FS', 'FT', 'FU', 'FV', 'FW', 'FX', 'FY', 'FZ', 'GA', 'GB', 'GC',
         'GD', 'GE', 'GF', 'GG', 'GH', 'GI', 'GJ', 'GK', 'GL', 'GM', 'GN', 'GO', 'GP', 'GQ', 'GR', 'GS', 'GT', 'GU',
         'GV', 'GW', 'GX', 'GY', 'GZ', 'HA', 'HB', 'HC', 'HD', 'HE', 'HF', 'HG', 'HH', 'HI', 'HJ', 'HK', 'HL', 'HM',
         'HN', 'HO', 'HP', 'HQ', 'HR', 'HS', 'HT', 'HU', 'HV', 'HW', 'HX', 'HY', 'HZ', 'IA', 'IB', 'IC', 'ID', 'IE',
         'IF', 'IG', 'IH', 'II', 'IJ', 'IK', 'IL', 'IM', 'IN', 'IO', 'IP', 'IQ', 'IR', 'IS', 'IT', 'IU', 'IV', 'IW',
         'IX', 'IY', 'IZ', 'JA', 'JB', 'JC', 'JD', 'JE', 'JF', 'JG', 'JH', 'JI', 'JJ', 'JK', 'JL', 'JM', 'JN', 'JO',
         'JP', 'JQ', 'JR', 'JS', 'JT', 'JU', 'JV', 'JW', 'JX', 'JY', 'JZ', 'KA', 'KB', 'KC', 'KD', 'KE', 'KF', 'KG',
         'KH', 'KI', 'KJ', 'KK', 'KL', 'KM', 'KN', 'KO', 'KP', 'KQ', 'KR', 'KS', 'KT', 'KU', 'KV', 'KW', 'KX', 'KY',
         'KZ', 'LA', 'LB', 'LC', 'LD', 'LE', 'LF', 'LG', 'LH', 'LI', 'LJ', 'LK', 'LL', 'LM', 'LN', 'LO', 'LP', 'LQ',
         'LR', 'LS', 'LT', 'LU', 'LV', 'LW', 'LX', 'LY', 'LZ', 'MA', 'MB', 'MC', 'MD', 'ME', 'MF', 'MG', 'MH', 'MI',
         'MJ', 'MK', 'ML', 'MM', 'MN', 'MO', 'MP', 'MQ', 'MR', 'MS', 'MT', 'MU', 'MV', 'MW', 'MX', 'MY', 'MZ', 'NA',
         'NB', 'NC', 'ND', 'NE', 'NF', 'NG', 'NH', 'NI', 'NJ', 'NK', 'NL', 'NM', 'NN', 'NO', 'NP', 'NQ', 'NR', 'NS',
         'NT', 'NU', 'NV', 'NW', 'NX', 'NY', 'NZ', 'OA', 'OB', 'OC', 'OD', 'OE', 'OF', 'OG', 'OH', 'OI', 'OJ', 'OK',
         'OL', 'OM', 'ON', 'OO', 'OP', 'OQ', 'OR', 'OS', 'OT', 'OU', 'OV', 'OW', 'OX', 'OY', 'OZ', 'PA', 'PB', 'PC',
         'PD', 'PE', 'PF', 'PG', 'PH', 'PI', 'PJ', 'PK', 'PL', 'PM', 'PN', 'PO', 'PP', 'PQ', 'PR', 'PS', 'PT', 'PU',
         'PV', 'PW', 'PX', 'PY', 'PZ', 'QA', 'QB', 'QC', 'QD', 'QE', 'QF', 'QG', 'QH', 'QI', 'QJ', 'QK', 'QL', 'QM',
         'QN', 'QO', 'QP', 'QQ', 'QR', 'QS', 'QT', 'QU', 'QV', 'QW', 'QX', 'QY', 'QZ', 'RA', 'RB', 'RC', 'RD', 'RE',
         'RF', 'RG', 'RH', 'RI', 'RJ', 'RK', 'RL', 'RM', 'RN', 'RO', 'RP', 'RQ', 'RR', 'RS', 'RT', 'RU', 'RV', 'RW',
         'RX', 'RY', 'RZ', 'SA', 'SB', 'SC', 'SD', 'SE', 'SF', 'SG', 'SH', 'SI', 'SJ', 'SK', 'SL', 'SM', 'SN', 'SO',
         'SP', 'SQ', 'SR', 'SS', 'ST', 'SU', 'SV', 'SW', 'SX', 'SY', 'SZ', 'TA', 'TB', 'TC', 'TD', 'TE', 'TF', 'TG',
         'TH', 'TI', 'TJ', 'TK', 'TL', 'TM', 'TN', 'TO', 'TP', 'TQ', 'TR', 'TS', 'TT', 'TU', 'TV', 'TW', 'TX', 'TY',
         'TZ', 'UA', 'UB', 'UC', 'UD', 'UE', 'UF', 'UG', 'UH', 'UI', 'UJ', 'UK', 'UL', 'UM', 'UN', 'UO', 'UP', 'UQ',
         'UR', 'US', 'UT', 'UU', 'UV', 'UW', 'UX', 'UY', 'UZ', 'VA', 'VB', 'VC', 'VD', 'VE', 'VF', 'VG', 'VH', 'VI',
         'VJ', 'VK', 'VL', 'VM', 'VN', 'VO', 'VP', 'VQ', 'VR', 'VS', 'VT', 'VU', 'VV', 'VW', 'VX', 'VY', 'VZ', 'WA',
         'WB', 'WC', 'WD', 'WE', 'WF', 'WG', 'WH', 'WI', 'WJ', 'WK', 'WL', 'WM', 'WN', 'WO', 'WP', 'WQ', 'WR', 'WS',
         'WT', 'WU', 'WV', 'WW', 'WX', 'WY', 'WZ', 'XA', 'XB', 'XC', 'XD', 'XE', 'XF', 'XG', 'XH', 'XI', 'XJ', 'XK',
         'XL', 'XM', 'XN', 'XO', 'XP', 'XQ', 'XR', 'XS', 'XT', 'XU', 'XV', 'XW', 'XX', 'XY', 'XZ', 'YA', 'YB', 'YC',
         'YD', 'YE', 'YF', 'YG', 'YH', 'YI', 'YJ', 'YK', 'YL', 'YM', 'YN', 'YO', 'YP', 'YQ', 'YR', 'YS', 'YT', 'YU',
         'YV', 'YW', 'YX', 'YY', 'YZ', 'ZA', 'ZB', 'ZC', 'ZD', 'ZE', 'ZF', 'ZG', 'ZH', 'ZI', 'ZJ', 'ZK', 'ZL', 'ZM',
         'ZN', 'ZO', 'ZP', 'ZQ', 'ZR', 'ZS', 'ZT', 'ZU', 'ZV', 'ZW', 'ZX', 'ZY', 'ZZ']

marks.sort(reverse=True)


class ClusterNode(object):
    def __init__(self, vec, left=None, right=None, id=None, count=1):
        """
        :param id: 用来标记哪些节点是计算过的
        :param vec: 坐标，保存两个数据聚类后形成新的中心
        :param count: 这个节点的叶子节点个数
         :param left: 左节点
         :param right:  右节点
        """
        self.mark = marks.pop()
        self.id = id
        self.center_vec = vec
        self.child_vecs = [vec]
        self.count = count
        self.left = left
        self.right = right


def euler_distance(point1, point2):
    """
    计算两点之间的欧式距离
    """
    eulerdistance = round(math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2)), 2)
    return eulerdistance  # 开方求距离


class Hierarchical(object):
    def __init__(self):
        self.fin_nodes = None
        self.labels = None  # 属于哪个簇，簇的标签

    def show_tree(self):
        if self.reserve == 1:
            my_visualize.show_BTree(self.fin_nodes[0])
        else:
            print("Hierarchical.cluster()函数的参数reverse置1")

    def cluster(self, x, type='average', reserve=1):
        # 初始化
        self.reserve = reserve
        currentclustid = -1
        nodes = [ClusterNode(vec=v, id=i) for i, v in enumerate(x)]  # i为下标，v为坐标元素
        point_num, future_num = np.shape(x)  # 特征的维度 point_num行元素个数 future_num列元素个数
        self.labels = [-1] * point_num  # 为初始的的簇初始化分类标签

        while len(nodes) > self.reserve:
            closest_part = self.clustering(nodes, type)
            # 合并两个聚类
            part1, part2 = closest_part
            node1, node2 = nodes[part1], nodes[part2]
            new_vec = [
                (node1.center_vec[i] * node1.count + node2.center_vec[i] * node2.count) / (node1.count + node2.count)
                for i in range(future_num)]  # 求新簇中心点的横(纵)坐标（平均值）

            # 生成二叉树
            new_node = ClusterNode(vec=new_vec,
                                   left=node1,
                                   right=node2,
                                   id=currentclustid,
                                   count=node1.count + node2.count
                                   )
            new_node.child_vecs = node1.child_vecs + node2.child_vecs
            currentclustid -= 1
            del nodes[part2], nodes[part1]
            nodes.append(new_node)

        self.fin_nodes = nodes
        self.calc_label()

    def calc_label(self):
        """
        调取聚类的结果
        """
        for i, node in enumerate(self.fin_nodes):
            # 将节点的所有叶子节点都分类
            self.leaf_traversal(node, i)

    def leaf_traversal(self, node: ClusterNode, label):
        """
        递归遍历叶子节点,并为他们分类
        """
        if node.left is None and node.right is None:
            self.labels[node.id] = label
        if node.left:
            self.leaf_traversal(node.left, label)
        if node.right:
            self.leaf_traversal(node.right, label)

    @staticmethod
    def clustering(nodes, type='average'):

        closest_part = None  # 表示最相似的两个簇的下标
        distances = np.zeros((len(nodes), len(nodes)))  # 由簇的个数初始化相应的距离矩阵

        dist = math.inf  # 初始设本次(average/single)聚类的最近的两簇的距离设为无限远
        if type == "average":  # average距离聚类
            for i in range(len(nodes) - 1):  # 簇A
                for j in range(i + 1, len(nodes)):  # 簇B
                    average_linkage_distance = 0  # 初始赋值簇间平均距离0
                    for k in range(nodes[i].count):  # 簇A元素
                        for l in range(nodes[j].count):  # 簇B元素
                            d = euler_distance(nodes[i].child_vecs[k], nodes[j].child_vecs[l])
                            average_linkage_distance += d
                    average_linkage_distance = average_linkage_distance / (nodes[i].count * nodes[j].count)  # 求两簇间平均距离
                    distances[i, j] = average_linkage_distance  # 更新各簇间的距离矩阵
                    if average_linkage_distance < dist:  # 如果该距离小于本次聚类的的最近两簇的平均距离（目前），则更新该距离以及相应的两簇下标
                        dist = average_linkage_distance
                        closest_part = (i, j)
            # print('closest_part:{}'.format(closest_part))
            # print('distances:\n{}'.format(distances))
            return closest_part

        if type == "single":  # single距离聚类
            for i in range(len(nodes) - 1):  # 簇A
                for j in range(i + 1, len(nodes)):  # 簇B
                    single_linkage_distance = math.inf
                    for k in range(nodes[i].count):  # 簇A元素
                        for l in range(nodes[j].count):  # 簇B元素
                            d = euler_distance(nodes[i].child_vecs[k], nodes[j].child_vecs[l])
                            if d <= single_linkage_distance:  # 如果该距离小于本次聚类的的最近两簇的距离（目前），则更新该距离以及相应的两簇下标
                                single_linkage_distance = d
                                distances[i, j] = single_linkage_distance  # 更新各簇间的距离矩阵
                            if single_linkage_distance < dist:  # 如果该距离小于本次聚类的的最近两簇的距离（目前），则更新该距离以及相应的两簇下标
                                dist = single_linkage_distance
                                closest_part = (i, j)
            print('closest_part:{}'.format(closest_part))
            print('distances:\n{}'.format(distances))
            return closest_part

        dist = 0  # 初始设本次(complete)聚类的最远的两簇的距离设为0
        if type == "complete":  # complete距离聚类
            for i in range(len(nodes) - 1):  # 簇A
                for j in range(i + 1, len(nodes)):  # 簇B
                    complete_linkage_distance = 0
                    for k in range(nodes[i].count):  # 簇A元素
                        for l in range(nodes[j].count):  # 簇B元素
                            d = euler_distance(nodes[i].child_vecs[k], nodes[j].child_vecs[l])
                            if d >= complete_linkage_distance:  # 两簇最长距离（笛卡尔积）
                                complete_linkage_distance = d
                                distances[i, j] = complete_linkage_distance  # 更新各簇间的距离矩阵
                            if complete_linkage_distance >= dist:  # 如果该距离大于本次聚类的的最远两簇的距离（目前），则更新该距离以及相应的两簇下标
                                dist = complete_linkage_distance
                                closest_part = (i, j)

            # print('closest_part:{}'.format(closest_part))
            # print('distances:\n{}'.format(distances))
            return closest_part


if __name__ == '__main__':
    # data = [[0, 0], [1, 0], [2, 0], [3, 0]]
    data = [[round(random.random() * 10, 2), 0] for i in range(20)]  # [x,y]x为横坐标，y为纵坐标
    print(data)
    test_reserve = 1  # int(input("请输入您最终想聚类成的簇数："))
    test_type = 'single'  # input("请输入您采取的簇对象间的距离衡量方式（single/complete/average）：")
    test = Hierarchical()
    test.cluster(data, test_type, test_reserve)
    print("{}聚类最终保留为{}簇，他们的聚类状况为：{}".format(test_type, test_reserve, np.array(test.labels)))
    test.show_tree()
