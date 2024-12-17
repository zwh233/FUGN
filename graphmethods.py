import numpy as np
from skimage.util import view_as_blocks
import torch
import cv2

# def calculate_similarity(block1, block2):
#     # 使用欧氏距离的倒数作为相似度度量
#     distance = np.linalg.norm(block1 - block2)
#     similarity = 1 / (1 + distance)  # 加1防止除以0
#     return similarity


# def calculate_similarity(block1, block2):
#     # 归一化图像块的值到0-255范围内
#     block1_norm = cv2.normalize(block1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#     block2_norm = cv2.normalize(block2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#
#     # 使用SIFT特征描述符计算相似度
#     sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
#     kp1, des1 = sift.detectAndCompute(block1_norm, None)
#     kp2, des2 = sift.detectAndCompute(block2_norm, None)
#
#     if des1 is None or des2 is None:
#         # 如果SIFT算法提取不到有效特征，则尝试其他特征提取算法
#         orb = cv2.ORB_create()
#         kp1, des1 = orb.detectAndCompute(block1, None)
#         kp2, des2 = orb.detectAndCompute(block2, None)
#
#     bf = cv2.BFMatcher()
#     matches = bf.knnMatch(des1, des2, k=2)
#
#     good_matches = []
#     for m, n in matches:
#         if m.distance < 0.75 * n.distance:
#             good_matches.append(m)
#
#     similarity = len(good_matches) / min(len(kp1), len(kp2))
#
#     return similarity


def calculate_similarity(block1, block2):
    # 将图像块转换为灰度图像
    block1_gray = cv2.cvtColor(block1, cv2.COLOR_BGR2GRAY)
    block2_gray = cv2.cvtColor(block2, cv2.COLOR_BGR2GRAY)

    # 定义Sobel算子
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # 对图像块进行Sobel边缘检测
    block1_sobel_x = cv2.filter2D(block1_gray, -1, sobel_x)
    block1_sobel_y = cv2.filter2D(block1_gray, -1, sobel_y)

    block2_sobel_x = cv2.filter2D(block2_gray, -1, sobel_x)
    block2_sobel_y = cv2.filter2D(block2_gray, -1, sobel_y)

    # 计算相似度（可以根据实际需求设计相似度计算方法）
    similarity = np.sum(np.abs(block1_sobel_x - block2_sobel_x) + np.abs(block1_sobel_y - block2_sobel_y))

    return similarity

def build_adjacency_matrix(image, block_size, k):

    # 确保图像尺寸可以被block_size整除
    assert image.shape[0] % block_size == 0 and image.shape[
        1] % block_size == 0, "Image dimensions must be divisible by block_size."

    # 将图像分割为blocks
    blocks = view_as_blocks(image, block_shape=(block_size, block_size, 3))
    num_blocks_y, num_blocks_x = blocks.shape[0], blocks.shape[1]
    blocks = blocks.reshape(-1, block_size, block_size, 3)  # 展平blocks数组

    # 初始化邻接矩阵
    num_nodes = num_blocks_y * num_blocks_x
    adjacency_matrix = np.zeros((num_nodes, num_nodes))

    # 计算所有blocks之间的相似度
    for i in range(num_nodes):
        for j in range(max(0, i - 2), min(num_nodes, i + 3)):  # 限制在二跳邻居内
            if i != j:
                sim = calculate_similarity(blocks[i], blocks[j])
                adjacency_matrix[i, j] = sim
                adjacency_matrix[j, i] = sim  # 无向图

    # 为每个节点选择k个最相似的邻居
    for i in range(num_nodes):
        neighbors = np.argsort(-adjacency_matrix[i])[:k + 1]  # 获取最大的k个相似度，+1因为包含自身
        adjacency_matrix[i] = 0  # 重置行
        adjacency_matrix[i, neighbors] = 1  # 设置最高的k个邻居为1

    return adjacency_matrix


def build_adjacency_matrices(dataloader, block_size, k=5):
    """
    根据输入的dataloader，为每个batch的图像构建邻接矩阵并按对角线拼接。
    :param dataloader: 输入的 DataLoader，加载的图像大小应为 batchsize*3*256*256
    :param block_size: 方块的大小
    :param k: 选择的邻居数量
    :return: 拼接后的邻接矩阵
    """
    # 初始化一个空的列表来存储每个batch的邻接矩阵
    # batch_adj_matrices = []
    all_batches_adj_matrices = []

    # for _, batch_images in dataloader:
    for batch_images, _  in dataloader:
        # 确保输入图像大小符合要求
        assert batch_images.shape[2] == 256 and batch_images.shape[3] == 256, "Image dimensions must be 256x256."

        batch_adj_matrices = []
        # 处理每个图像
        for image in batch_images:
            # 转换图像格式以适应build_adjacency_matrix函数
            image_np = image.permute(1, 2, 0).numpy()  # 转换为256*256*3
            adj_matrix = build_adjacency_matrix(image_np, block_size, k)
            batch_adj_matrices.append(adj_matrix)

        # 按对角线拼接所有邻接矩阵
        full_adj_matrix = np.block([[adj if i == j else np.zeros_like(adj)
                                     for j, adj in enumerate(batch_adj_matrices)]
                                    for i, adj in enumerate(batch_adj_matrices)])
        all_batches_adj_matrices.append(full_adj_matrix)

    return all_batches_adj_matrices

def dense_to_sparse(adj_matrix):
    """
    Convert a dense adjacency matrix to a sparse edge index tensor.

    Parameters:
    adj_matrix (np.ndarray): A two-dimensional NumPy array representing the adjacency matrix.

    Returns:
    edge_index (torch.Tensor): A tensor with two rows, where each column represents an edge in the graph.
    """
    # Ensure the adjacency matrix is square
    assert adj_matrix.shape[0] == adj_matrix.shape[1], "Adjacency matrix must be square."

    # Find the indices of all non-zero elements using np.where
    src, dst = np.where(adj_matrix != 0)

    # Concatenate the src and dst arrays into a single numpy array
    edge_index_np = np.stack((src, dst), axis=0)

    # Convert the numpy array to a torch tensor
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)

    return edge_index
