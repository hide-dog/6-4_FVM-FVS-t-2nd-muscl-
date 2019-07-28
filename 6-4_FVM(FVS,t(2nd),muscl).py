import numpy
import os
import matplotlib.pyplot as pyplot

# --------------------------
# -- initial value        --
# --------------------------
nstep = 300                  # 時間ステップ数
nx0 = 100                    # 空間ステップ数
dt = 0.001                   # 時間刻み幅
dx = 0.01                    # 空間刻み幅

lpml = 1                     # 仮想境界セル数
nx = nx0+2*lpml              # 総空間セル数

# -- muscl --
k_muscl = 1/3                  # muscl精度
b_muscl = (3-k_muscl)/(1-k_muscl)

# -- constant --
gamma = 1.4                    # 比熱比

# -- output --
dir_name = "FVS_muscl"         # 出力フォルダ名
out_name_front = "time"        # 出力ファイル名（前半）
out_name_back = "d-3"          # 出力ファイル名（後半）
# --------------------------
# -- function             --
# --------------------------


def setup():  # 初期値入力
    global x, bol, bor, qf, Qc
    u = [0.0]*nx               # 速度
    rho = [0.0]*nx             # 密度
    p = [0.0]*nx               # 圧力
    e = [0.0]*nx               # エネルギー
    x = [0.0]*nx               # 位置

    """
    e=p/(r-1)+rho*u^2/2
    """
    for i in range(nx):
        u[i] = 0.0

        if i <= nx*0.5:
            rho[i] = 1.0
            p[i] = 1.0

        else:
            rho[i] = 0.125
            p[i] = 0.1

        e[i] = p[i]/(gamma-1)+rho[i]*(u[i]**2)/2
        x[i] = i*dx-dx/2

    bol = [0.0]*3             # 左端仮想セル
    bor = [0.0]*3             # 右端仮想セル
    for j in range(3):
        if j == 0:
            bol[j] = rho[0]
            bor[j] = rho[nx-1]
        elif j == 1:
            bol[j] = u[0]*rho[0]
            bor[j] = u[nx-1]*rho[nx-1]
        elif j == 2:
            bol[j] = e[0]
            bor[j] = e[nx-1]

    qf = [[0.0] * 3 for i in [1] * nx]  # 基本量
    Qc = [[0.0] * 3 for i in [1] * nx]  # 保存量
    for i in range(nx):
        for j in range(3):
            if j == 0:
                qf[i][j] = u[i]
                Qc[i][j] = rho[i]
            elif j == 1:
                qf[i][j] = rho[i]
                Qc[i][j] = u[i]*rho[i]
            elif j == 2:
                qf[i][j] = p[i]
                Qc[i][j] = e[i]


def cal_Q():
    global Qc

    cal_Res()                         # 右辺Rの計算

    Qc = numpy.array(Qc)                # Qcのarray化
    lo_R = numpy.array(Res)             # 右辺Rのarray化

    Qc_np = numpy.array(Qc[:])          # Qc^nの保存

    for i in range(1, nx-1):           # Qc^(n+1)の計算
        Qc[i] = Qc[i]-dt/dx*lo_R[i]
    cal_Res()                         # R^(n+1)の計算
    lo_R = numpy.array(Res)             # R^(n+1)のarray化
    for i in range(1, nx-1):           # nとn+1の平均の値を用いて,二次精度のn+1の計算
        Qc[i] = 1/2*(Qc_np[i]+Qc[i]-dt/dx*lo_R[i])

    Qc.tolist()

    bound()


def bound():  # 境界の計算
    global Qc
    for i in range(3):
        Qc[0][i] = 2*bol[i]-Qc[1][i]  # 左端境界の計算
        Qc[nx-1][i] = Qc[nx-2][i]     # 右端の計算


def cal_Res():  # 境界フラックスの計算
    global Res

    Res = numpy.array([[0.0] * 3 for i in [1] * nx])
    fvs()                             # FVS法によるフラックスの作成

    for i in range(1, nx-1):
        Res[i] = Fplus[i]-Fplus[i-1]

    Res.tolist()


def fvs():  # FVS法によるフラックスの計算(セル1と2の境界をFplus[1]に格納)
    global Fplus

    Fplus = [[0.0] * 3 for i in [1] * (nx+1)]
    muscl()

    for i in range(0, nx-1):
        # i+1/2セルにおけるR,R^-1,Λ,|Λ|
        R, R_inv, Gam, Gam_abs = A_pm(QcL[i], qfL[i])
        Ap = numpy.dot((numpy.dot(R, Gam+Gam_abs)),
                       R_inv)               # 固有値が正のものを計算

        # i+1/2セルにおけるR,R^-1,Λ,|Λ|
        R, R_inv, Gam, Gam_abs = A_pm(QcR[i], qfR[i])
        Am = numpy.dot((numpy.dot(R, Gam-Gam_abs)),
                       R_inv)               # 固有値が負のものを計算

        Fplus[i] = 0.5*(numpy.dot(Ap, QcL[i]) +
                        numpy.dot(Am, QcR[i]))   # フラックスを計算


def A_pm(lQc, lqf):  # ヤコビアン行列の固有値もろもろ計算
    H = (lQc[2]+lqf[2])/lQc[0]  # エンタルピー
    u = lqf[0]
    c = numpy.sqrt((gamma-1)*(H-0.5*u**2))
    b_para = (gamma-1)/c**2
    a_para = 0.5*b_para*u**2

    R = numpy.array([[1.0, 1.0, 1.0, ], [u-c, u, u+c],
                     [H-u*c, 0.5*u**2, H+u*c]])
    R_inv = numpy.array([[0.5*(a_para+u/c), 0.5*(-b_para*u-1/c), 0.5*b_para], [
                        1-a_para, b_para*u, -b_para], [0.5*(a_para-u/c), 0.5*(-b_para*u+1/c), 0.5*b_para]])
    Gam = numpy.array([[(u-c), 0.0, 0.0], [0.0, u, 0.0], [0.0, 0.0, (u+c)]])
    Gam_abs = numpy.array(
        [[abs(u-c), 0.0, 0.0], [0.0, abs(u), 0.0], [0.0, 0.0, abs(u+c)]])

    return R, R_inv, Gam, Gam_abs


def muscl():
    global qf, qfL, qfR, QcL, QcR
    # 1と2の間を1に収納

    qf = Qctoqf(Qc)                            # 基本量の算出

    qfL = [[0.0] * 3 for i in [1] * (nx+1)]    # セル境界左右の基本量のリスト
    qfR = [[0.0] * 3 for i in [1] * (nx+1)]

    for i in range(1, nx-2):
        for j in range(3):
            dplus_j = qf[i+1][j]-qf[i][j]
            dminus_j = qf[i][j]-qf[i-1][j]
            dplus_jp = qf[i+2][j]-qf[i+1][j]
            dminus_jp = qf[i+1][j]-qf[i][j]

            # muscl法による左右の値を計算
            qfL[i][j] = qf[i][j]+1/4*((1-k_muscl)*minmod(dminus_j, dplus_j, b_muscl)+(
                1+k_muscl)*minmod(dplus_j, dminus_j, b_muscl))
            qfR[i][j] = qf[i+1][j]-1/4*((1-k_muscl)*minmod(dplus_jp, dminus_jp, b_muscl)+(
                1+k_muscl)*minmod(dminus_jp, dplus_jp, b_muscl))

    # 境界条件の内側
    for j in range(3):
        dplus_jp = qf[2][j]-qf[1][j]
        dminus_jp = qf[1][j]-qf[0][j]
        qfR[0][j] = qf[1][j]-1/4*((1-k_muscl)*minmod(dplus_jp, dminus_jp,
                                                     b_muscl)+(1+k_muscl)*minmod(dminus_jp, dplus_jp, b_muscl))

        dplus_j = qf[nx-1][j]-qf[nx-2][j]
        dminus_j = qf[nx-2][j]-qf[nx-3][j]
        qfL[nx-2][j] = qf[nx-2][j]+1/4*((1-k_muscl)*minmod(
            dminus_j, dplus_j, b_muscl)+(1+k_muscl)*minmod(dplus_j, dminus_j, b_muscl))

    # 基本量から保存量へ
    QcL = qftoQc(qfL)
    QcR = qftoQc(qfR)

    # 境界条件の外側用(境界は風上)
    qfL[0] = qf[0][:]
    QcL[0] = Qc[0][:]
    qfR[nx-2] = qf[nx-1][:]
    QcR[nx-2] = Qc[nx-1][:]


def minmod(x, y, b):  # 流束制限関数minmod
    ans = numpy.sign(x)*max(0, min(abs(x), numpy.sign(x)*y*b))

    return ans


def qftoQc(qf):  # 基本量から保存量変換
    lo_Qc = [[0.0] * 3 for i in [1] * nx]
    for i in range(nx):
        for j in range(3):
            if j == 0:
                lo_Qc[i][j] = qf[i][1]
            elif j == 1:
                lo_Qc[i][j] = qf[i][1]*qf[i][0]
            elif j == 2:
                lo_Qc[i][j] = (qf[i][2]/(gamma-1)+1.0 /
                               2.0*qf[i][1]*(qf[i][0]**2))

    return lo_Qc


def Qctoqf(Qc):  # 保存量から基本量変換
    lo_qf = [[0.0] * 3 for i in [1] * nx]
    for i in range(nx):
        for j in range(3):
            if j == 0:
                lo_qf[i][j] = Qc[i][1]/Qc[i][0]
            elif j == 1:
                lo_qf[i][j] = Qc[i][0]
            elif j == 2:
                lo_qf[i][j] = (gamma-1)*(Qc[i][2]-1.0/2.0*Qc[i]
                                         [0]*((Qc[i][1]/Qc[i][0])**2))
    return lo_qf


def output_q(f_name):  # テキスト形式で出力
    outlist = ["x[m] u[m/s] rho[kg/m3] p[Pa]"]  # 出力するものの名前
    for i in range(len(qf)):
        outlist.append(str(x[i])+" "+str(qf[i][0])+" " +
                       str(qf[i][1])+" "+str(qf[i][2]))
    outlist = '\n'.join(outlist)

    with open(dir_name+"/"+f_name, 'wt') as f:
        f.write(outlist)


def cre_dir():  # フォルダ作成
    try:
        os.mkdir(dir_name)
    except:
        pass


# --------------------------
# -- preparetion          --
# --------------------------
cre_dir()
setup()
# --------------------------
# -- main                 --
# --------------------------

for k in range(nstep):

    print(k)
    cal_Q()
    qf = Qctoqf(Qc)

    output_q(out_name_front+'{:0=4}'.format(int(k*dt*1000))+out_name_back)
