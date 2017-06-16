import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import matplotlib.animation as animation


class Qlearning(object):
    """
    Q学習クラス
    """

    def __init__(self,
                 s_num,
                 a_num,
                 lr=0.1,
                 discount=0.9
                 ):
        self.Q = np.random.rand(s_num, a_num)
        self.lr = lr            # 学習係数
        self.discount = discount
        pass

    def action(self, state):
        a = np.argmax(self.Q[state, :])
        return a

    def update(self, r, s, a, next_s):
        next_exp = r + self.discount * np.max(self.Q[next_s, :])
        self.Q[s, a] = (1 - self.lr) * self.Q[s, a] + self.lr * next_exp


class Player(object):
    """
    プレイヤー
    """

    def __init__(self, name=None):
        self.name = name
        self.labyrinth = None
        pass

    def enter(self, labyrinth):
        self.position = 0
        self.past_pos = 0
        self.labyrinth = labyrinth
        pass

    def operation(self, opr):
        self.opr = opr
        pass

    def action(self):
        dirc = self.opr.action(self.position)
        self.move(dirc)
        return dirc

    def move(self, direction):
        """
        direction (0 ~ 3) 方向へ動く
        """
        if self.labyrinth is None:
            raise("Not in labyrinth.")

        if self.labyrinth.movable(self.position, direction):
            self.past_pos = self.position

            if direction == 0:
                self.position = self.position + 1
            elif direction == 1:
                self.position = self.position + self.labyrinth.size[1]
            elif direction == 2:
                self.position = self.position - 1
            elif direction == 3:
                self.position = self.position - self.labyrinth.size[1]
            else:
                raise("予期しない方向への移動です")

            return self.position
        else:
            self.past_pos = self.position

    def search(self):
        """
        探索を行う
        """
        dirc = self.action()
        reward = self.labyrinth.rewards[self.position]
        self.opr.update(reward, self.past_pos, dirc, self.position)

    def restart(self):
        if self.labyrinth is None:
            raise("Not in Labyrinth")
        self.past_pos = self.labyrinth.start
        self.position = self.labyrinth.start
        pass


class Labyrinth(object):
    """
    Q学習で使うマップ
    """

    def __init__(self,
                 size=(3, 3),
                 reward=1.0):
        self.size = size
        self.N = np.prod(self.size)
        self.rewards = np.zeros(self.N, dtype=float)
        # goal
        self.start = 0
        self.goal = self.N - 1
        self.rewards[self.goal] = reward
        pass

    def movable(self, position, dir):
        if position // self.size[1] == 0 and dir == 3:
            return False
        elif position // self.size[1] == self.size[0] - 1 and dir == 1:
            return False
        elif position % self.size[1] == self.size[1] - 1 and dir == 0:
            return False
        elif position % self.size[1] == 0 and dir == 2:
            return False
        else:
            return True


def test(seed=0):
    np.random.seed(seed)

    p = Player(name="pochi")
    lab = Labyrinth(size=(4, 4), reward=1)
    p.operation(
        Qlearning(s_num=lab.N, a_num=4)
    )

    def update(num, player, lab, rec, txts):
        # はじめだけは更新なし表示のみ
        if num == 0:
            rec.set_xy((player.position % lab.size[1],
                        player.position // lab.size[1]))
            for i, txt in enumerate(txts):
                txt.set_text("{:.2f}".format(p.opr.Q[i // 4, i % 4]))
            return rec, txts

        # ゴールに到着していなければ探索
        if player.position != lab.goal:
            player.search()
            # ゴールに到着したら色変更
            if player.position == lab.goal:
                rec.set_color("red")
        else:
            # ゴールに到着していたら、スタートへジャンプ
            player.restart()
            rec.set_color("black")
        # プレイヤー位置を変更
        rec.set_xy((player.position % lab.size[1],
                    player.position // lab.size[1]))

        # Q値の表示
        for i, txt in enumerate(txts):
            txt.set_text("{:.2f}".format(p.opr.Q[i // 4, i % 4]))

        return rec, txts

    p.enter(lab)

    fig = plt.figure(figsize=(6, 6), frameon=False)
    ax = fig.add_axes([0, 0, 1, 1],
                      xlim=(0, lab.size[0]),
                      ylim=(0, lab.size[1]),
                      xticks=[], yticks=[],
                      aspect='equal',
                      )
    ax.invert_yaxis()
    rec = ax.add_patch(plt.Rectangle(
        xy=(0, 0), width=1, height=1, fill=False,
        lw=5, facecolor='black', edgecolor='black'))
    # rec = ax.add_patch(plt.Rectangle(
    #     xy=(0, 0), width=1, height=1, fill=True,
    #     lw=5, facecolor='black', edgecolor='black'))

    txts = []

    for pos in range(p.opr.Q.shape[0]):
        x = pos % lab.size[1] + 0.4
        y = pos // lab.size[0] + 0.5
        txts.append(ax.text(x + 0.4, y, "", fontsize=9))
        txts.append(ax.text(x, y + 0.4, "", fontsize=9))
        txts.append(ax.text(x - 0.4, y, "", fontsize=9))
        txts.append(ax.text(x, y - 0.4, "", fontsize=9))

    ax.text(0.3, 0.53, "START", fontsize=13)
    ax.text(lab.size[1] - 0.67, lab.size[0] -
            0.47, "GOAL", fontsize=13, color="red")

    print("start.")
    ani = animation.FuncAnimation(fig,
                                  update,
                                  300,
                                  fargs=(p, lab, rec, txts),
                                  interval=30,
                                  repeat=False,
                                  )
    # ani.save("labylinth.mp4")
    # ani.save("labylinth_withQ.mp4")
    plt.show()
