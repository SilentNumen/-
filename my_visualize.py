import time

from matplotlib import pyplot as plt

d_hor = 4  # 节点水平距离
d_vec = 8  # 节点垂直距离


def get_left_width(node):
    '''获得根左边宽度'''
    return get_width(node.left)


def get_right_width(node):
    '''获得根右边宽度'''
    return get_width(node.right)


def get_width(node):
    '''获得树的宽度'''
    if node == None:
        return 0
    return get_width(node.left) + 1 + get_width(node.right)


def get_height(node):
    '''获得二叉树的高度'''
    if node == None:
        return 0
    return max(get_height(node.left), get_height(node.right)) + 1


def get_w_h(rootnode):
    '''返回树的宽度和高度'''
    w = get_width(rootnode)
    h = get_height(rootnode)
    return w, h


def draw_a_node(x, y, mark, child_vecs):
    '''画一个节点'''
    plt.text(x, y, '{}'.format(mark), ha='center', va='bottom', fontsize=50)
    # plt.text(x, y, '{}:{}'.format(mark, child_vecs), ha='center', va='bottom', fontsize=30)


def draw_a_edge(x1, y1, x2, y2):
    '''画一条边'''
    x = (x1, x2)
    y = (y1, y2)
    plt.plot(x, y, 'k-')


def create_win(rootnode):
    '''创建窗口'''
    WEIGHT, HEIGHT = get_w_h(rootnode)
    WEIGHT = (WEIGHT + 1) * d_hor
    HEIGHT = (HEIGHT + 1) * d_vec
    fig = plt.figure(figsize=(100, 20))

    plt.xlim(0, WEIGHT)
    plt.ylim(0, HEIGHT)

    x = (get_left_width(rootnode) + 1) * d_hor  # x, y 是第一个要绘制的节点坐标，由其左子树宽度决定
    y = HEIGHT - d_vec
    return fig, x, y


def print_tree_by_inorder(rootnode, x, y):
    '''通过中序遍历打印二叉树'''
    if rootnode == None:
        return
    draw_a_node(x, y, rootnode.mark, rootnode.child_vecs)
    lx = rx = 0
    ly = ry = y - d_vec
    if rootnode.left != None:
        lx = x - d_hor * (get_right_width(rootnode.left) + 1)  # x-左子树的右边宽度
        draw_a_edge(x, y, lx, ly)
    if rootnode.right != None:
        rx = x + d_hor * (get_left_width(rootnode.right) + 1)  # x-右子树的左边宽度
        draw_a_edge(x, y, rx, ry)
    # 递归打印
    print_tree_by_inorder(rootnode.left, lx, ly)
    print_tree_by_inorder(rootnode.right, rx, ry)


def show_BTree(rootnodenode):
    '''可视化二叉树'''
    _, x, y = create_win(rootnodenode)
    print_tree_by_inorder(rootnodenode, x, y)
    plt.savefig("{}.png".format(time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))))
    plt.show()
