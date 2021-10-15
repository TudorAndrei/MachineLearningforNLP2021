import numpy as np

vocab = ["table", "chair", "lecture", "hotho", "assignment"]

u_dict = {"chair": [0.8, 0.6, 0.7],
          "table": [0.74, 0.7, 0.6],
          "lecture": [0.1, 0.5, 0.2],
          "assignment": [0.2, 0.4, 0.18]}
edge_weight = {(0, 1): 0.9,
               (2, 0): 0.4,
               (2, 3): 0.7,
               (3, 4): 0.5,
               (4, 2): 0.6}

emb_len = len(list(u_dict.values())[0])


def generate_matices(u_dict, vocab, emb_len, edge_weight):
    w = []
    a = None
    s = None
    u = []
    for node in vocab:
        if node in u_dict:
            u.append(u_dict[node])
            w.append(u_dict[node])
        else:
            w.append([0]*emb_len)

    u = np.array(u)
    w = np.array(w)
    a = np.eye(len(vocab))
    missing_index = []
    for v in vocab:
        if v not in u_dict:
            missing_index.append(vocab.index(v))

    for m_i in missing_index:
        a[m_i, m_i] = 0

    s = np.eye(len(vocab))
    for src, dst in edge_weight:
        s[src, dst] = edge_weight[(src, dst)]
        s[dst, src] = edge_weight[(src, dst)]

    return u, w, a, s


def update_w(w0, s, a):
    w = s @ w0 + w0
    ia = np.linalg.inv(np.eye(a.shape[0]) + a)
    w = (w.T @ ia).T
    w = w/w.sum(axis=1, keepdims=True)

    return w


def loss(w, edge_weight):
    loss = 0
    for i in range(w.shape[0]):
        # print(f"{i=}")
        l = 0
        for src, dst in edge_weight:
            if i == src:
                # print(f" {src=} {dst=}\n")
                ew = edge_weight[(src, dst)]
                aux = w[i, :] - w[dst, :]
                aux = np.abs(aux)
                aux = np.sum(aux)
                aux = aux ** 2
                l += ew * aux
        loss += l
    return loss*2


u, w0, a, s = generate_matices(u_dict, vocab, emb_len, edge_weight)
w1 = update_w(w0, s, a)
print(loss(w0, edge_weight))
print(loss(w1, edge_weight))
