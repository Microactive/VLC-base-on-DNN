import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms
import torch.nn.functional as f
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import mlab
from matplotlib import rcParams
import statsmodels.api as sm
import tkinter as tk
from tkinter import messagebox  # import this to fix messagebox error
import pickle
from tkinter import filedialog
import time  # 用于计算训练时间


global t1, t2, t3, t4
global file_path_TrL
global file_path_TrD
global file_path_TeL
global file_path_TeD



def file_adress_TrD(t1):
    global file_path_TrD
    root = tk.Tk()
    root.withdraw()
    file_path_TrD = filedialog.askopenfilename()
    t1.delete(1.0, tk.END)  # 清空整个框内的数据
    t1.insert('end', str(file_path_TrD))

def file_adress_TrL(t2):
    global file_path_TrL
    root = tk.Tk()
    root.withdraw()
    file_path_TrL = filedialog.askopenfilename()
    t2.delete(1.0, tk.END)
    t2.insert('end', str(file_path_TrL))

def file_adress_TeD(t3):
    global file_path_TeD
    root = tk.Tk()
    root.withdraw()
    file_path_TeD = filedialog.askopenfilename()
    t3.delete(1.0, tk.END)
    t3.insert('end', str(file_path_TeD))


def file_adress_TeL(t4):
    global file_path_TeL
    root = tk.Tk()
    root.withdraw()
    file_path_TeL = filedialog.askopenfilename()
    t4.delete(1.0, tk.END)
    t4.insert('end', str(file_path_TeL))


def save(if_train_new):
    # 处理数据集
    global file_path_TrL
    global file_path_TrD
    global file_path_TeL
    global file_path_TeD
    global t

    if if_train_new:
        TrainLabel = torch.from_numpy(np.loadtxt(str(file_path_TrL), delimiter=',', dtype=np.float32))
        TrainData = torch.from_numpy(np.loadtxt(str(file_path_TrD), delimiter=',', dtype=np.float32))
        TestLabel = np.loadtxt(str(file_path_TeL), delimiter=',', dtype=np.float32)
        TestData = torch.from_numpy(np.loadtxt(str(file_path_TeD), delimiter=',', dtype=np.float32))
        trainlabel = Variable(TrainLabel)
        traindata = Variable(TrainData)
        testdata = Variable(TestData)
    else:
        TrainLabel = torch.from_numpy(np.loadtxt('./dataset/TrainLabel_fromrealworld.csv', delimiter=',', dtype=np.float32))
        TrainData = torch.from_numpy(np.loadtxt('./dataset/TrainData_fromrealworld.csv', delimiter=',', dtype=np.float32))
        TestLabel = np.loadtxt('./dataset/TestLabel_fromrealworld.csv', delimiter=',', dtype=np.float32)
        TestData = torch.from_numpy(np.loadtxt('./dataset/TestData_fromrealworld.csv', delimiter=',', dtype=np.float32))
        trainlabel = Variable(TrainLabel)
        traindata = Variable(TrainData)
        testdata = Variable(TestData)


    # 定义神经网络
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.hidden1 = torch.nn.Linear(4, 64)
            self.hidden2 = torch.nn.Linear(64, 32)
            self.hidden3 = torch.nn.Linear(32, 20)
            self.hidden4 = torch.nn.Linear(20, 10)
            self.predict = torch.nn.Linear(10, 3)

        def forward(self, x):
            x = f.relu(self.hidden1(x))
            x = f.relu(self.hidden2(x))
            x = f.relu(self.hidden3(x))
            x = f.relu(self.hidden4(x))
            y_pred = self.predict(x)
            return y_pred

        # 备用的神经网络
        # def __init__(self):
        #     super(Net, self).__init__()
        #     self.hidden1 = torch.nn.Linear(4, 32)
        #     self.hidden2 = torch.nn.Linear(32, 20)
        #     self.hidden3 = torch.nn.Linear(20, 10)
        #     self.predict = torch.nn.Linear(10, 3)
        #
        # def forward(self, x):
        #     x = f.relu(self.hidden1(x))
        #     x = f.relu(self.hidden2(x))
        #     x = f.relu(self.hidden3(x))
        #     y_pred = self.predict(x)
        #     return y_pred


    net = Net()

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.99))  # 训练的工具
    loss_func = torch.nn.MSELoss()

    # 训练神经网络
    step = 0
    start_time = time.time()  # 开始计时
    while True:

        train_pred = net(traindata)

        loss = loss_func(train_pred, trainlabel)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1
        if step % 10 == 0:
            test_pred = net(testdata)
            test_pred_data = test_pred.data.numpy()
            accuracy = np.mean(np.sqrt(np.sum((test_pred_data - TestLabel) ** 2, 1)))
            print('step:', step, ' 定位误差:', accuracy)
            num = 0
            loss_list = []  # 记录定位误差数据，用于绘制误差分布直方图和cdf
            if accuracy < 0.026:
                for point_pred in test_pred_data:
                    point_loss = np.mean(np.sqrt((point_pred - TestLabel[num, :]) ** 2))
                    loss_list.append(point_loss)
                    num += 1
                torch.save(net.state_dict(), 'net_2cm_parameters.pkl')
                print("训练结束")
                print("成功保存神经网络")
                end_time = time.time()  # 结束计时
                print('训练用时：%.3f' % ((end_time - start_time) / 60), 'min')
                break

    # 绘制空间三维散点图
    # TestLabel的x,y,z轴坐标
    x1 = TestLabel[:, 0]
    y1 = TestLabel[:, 1]
    z1 = TestLabel[:, 2]

    # test_pred_data的x,y,z轴坐标
    x2 = test_pred_data[:, 0]
    y2 = test_pred_data[:, 1]
    z2 = test_pred_data[:, 2]

    # 绘图
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_title('Indoor visible light location')
    ax.scatter(x1, y1, z1, c='r', label='real')
    ax.scatter(x2, y2, z2, c='g', label='prediction')
    ax.legend(loc='best')

    # 添加坐标轴
    ax.set_zlabel('z/m', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('y/m', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('x/m', fontdict={'size': 15, 'color': 'red'})

    plt.show()

    # 绘制误差分布直方图
    # arr: 需要计算直方图的一维数组
    # bins: 直方图的柱数
    # normed: 是否将得到的直方图向量归一化。默认为0
    arr = loss_list  # loss_list 为定位误差集
    plt.hist(arr, bins=30, normed=0, edgecolor='None', facecolor='red')
    plt.title("Error distribution histogram")
    plt.xlabel('Positioning error')
    plt.ylabel('Number')
    plt.show()

    # 绘制CDF
    sample = loss_list  # loss_list的类型为列表，表示定位误差集.
    ecdf = sm.distributions.ECDF(sample)
    x = np.linspace(min(sample), max(sample))
    y = ecdf(x)
    plt.title('CDF')
    plt.xlabel('Positioning error')
    plt.ylabel('Probability')
    plt.step(x, y)
    # 标出概率为90%的那个点
    for a, b in zip(x, y):
        if round(b, 2) == 0.900:
            plt.text(round(a, 2), round(b, 3), (round(a, 2), round(b, 3)), ha='center', va='bottom', fontsize=10,
                     color='red')
    plt.show()

# 定义提取神经网络状态参数的函数
def restore_net_parameters():

    # 处理数据集
    global file_path_TeL
    global file_path_TeD
    TestLabel = np.loadtxt(str(file_path_TeL), delimiter=',', dtype=np.float32)
    TestData = torch.from_numpy(np.loadtxt(str(file_path_TeD), delimiter=',', dtype=np.float32))
    testdata = Variable(TestData)

    class Net2(torch.nn.Module):
        def __init__(self):
            super(Net2, self).__init__()
            self.hidden1 = torch.nn.Linear(4, 256)
            self.hidden2 = torch.nn.Linear(256, 128)
            self.hidden3 = torch.nn.Linear(128, 64)
            self.hidden4 = torch.nn.Linear(64, 32)
            self.hidden5 = torch.nn.Linear(32, 20)
            self.hidden6 = torch.nn.Linear(20, 10)
            self.predict = torch.nn.Linear(10, 3)

        def forward(self, x):
            x = f.relu(self.hidden1(x))
            x = f.relu(self.hidden2(x))
            x = f.relu(self.hidden3(x))
            x = f.relu(self.hidden4(x))
            x = f.relu(self.hidden5(x))
            x = f.relu(self.hidden6(x))
            y_pred = self.predict(x)
            return y_pred
    net2 = Net2()
    net2.load_state_dict(torch.load('net_parameters.pkl'))  # 提取net1的状态参数，将状态参数给net3

    # 使用已经保存好的神经网络进行测试
    test_pred = net2(testdata)
    test_pred_data = test_pred.data.numpy()
    num = 0
    loss_list = []  # 记录定位误差数据，用于绘制误差分布直方图和cdf

    for point_pred in test_pred_data:
        point_loss = np.mean(np.sqrt((point_pred - TestLabel[num, :]) ** 2))
        loss_list.append(point_loss)
        num += 1


    # 绘制空间三维散点图
    # TestLabel的x,y,z轴坐标
    x1 = TestLabel[:, 0]
    y1 = TestLabel[:, 1]
    z1 = TestLabel[:, 2]

    # test_pred_data的x,y,z轴坐标
    x2 = test_pred_data[:, 0]
    y2 = test_pred_data[:, 1]
    z2 = test_pred_data[:, 2]

    # 绘图
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_title('Indoor visible light location')
    ax.scatter(x1, y1, z1, c='r', label='real')
    ax.scatter(x2, y2, z2, c='g', label='prediction')
    ax.legend(loc='best')

    # 添加坐标轴
    ax.set_zlabel('z/m', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('y/m', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('x/m', fontdict={'size': 15, 'color': 'red'})

    plt.show()

    # 绘制误差分布直方图
    # arr: 需要计算直方图的一维数组
    # bins: 直方图的柱数
    # normed: 是否将得到的直方图向量归一化。默认为0
    arr = loss_list  # loss_list 为定位误差集
    plt.hist(arr, bins=30, normed=0, edgecolor='None', facecolor='red')
    plt.title("Error distribution histogram")
    plt.xlabel('Positioning error')
    plt.ylabel('Number')
    plt.show()

    # 绘制CDF
    sample = loss_list  # loss_list的类型为列表，表示定位误差集.
    ecdf = sm.distributions.ECDF(sample)
    x = np.linspace(min(sample), max(sample))
    y = ecdf(x)
    plt.title('CDF')
    plt.xlabel('Positioning error')
    plt.ylabel('Probability')
    plt.step(x, y)
    # 标出概率为90%的那个点
    for a, b in zip(x, y):
        if round(b, 2) == 0.900:
            plt.text(round(a, 2), round(b, 3), (round(a, 2), round(b, 3)), ha='center', va='bottom', fontsize=10, color='red')
    plt.show()

# 创建主窗口
window = tk.Tk()
window.title('欢迎使用基于深度神经网络的室内可见光定位系统')
window.geometry('450x300')


# 添加背景图
canvas = tk.Canvas(window, height=200, width=500)  # 创建画布
image_file = tk.PhotoImage(file='RC.png')  # 加载图片文件
image = canvas.create_image(180, 20, anchor='nw', image=image_file)  # 将图片置于画布上
canvas.pack()  # 放置画布


# 用户信息
tk.Label(window, text='User name:').place(x=120, y=150)  # 创建一个`label`名为`User name: `置于坐标（120,150）
tk.Label(window, text='Password:').place(x=120, y=190)
# 账号密码输入框
var_usr_name = tk.StringVar()  # 定义变量,用于输入用户名
var_usr_name.set('R&C')  # 输入用户名窗口默认显示'write your name'
entry_usr_name = tk.Entry(window, textvariable=var_usr_name)  # 创建输入框
entry_usr_name.place(x=200, y=150)
var_usr_pwd = tk.StringVar()
entry_usr_pwd = tk.Entry(window, textvariable=var_usr_pwd, show='*')  # `show`这个参数将输入的密码变为`***`的形式
entry_usr_pwd.place(x=200, y=190)


# 登录功能
def usr_login():

    global file_path_TrD
    global file_path_TrL
    global file_path_TeD
    global file_path_TeL

    # 这两行代码就是获取用户输入的`usr_name`和`usr_pwd`
    usr_name = var_usr_name.get()
    usr_pwd = var_usr_pwd.get()

    # 这里设置异常捕获，当我们第一次访问用户信息文件时是不存在的，所以这里设置异常捕获
    try:
        # 'rb'是打开文件为以二进制格式“读”，文件必须存在，否则会报错
        with open('usrs_info.pickle', 'rb') as usr_file:
            usrs_info = pickle.load(usr_file)
    except FileNotFoundError:
# 这里就是我们在没有读取到`usr_file`的时候，程序会创建一个`usr_file`这个文件，并将管理员
# 的用户和密码写入，即用户名为`admin`密码为`admin`。
        with open('usrs_info.pickle', 'wb') as usr_file:
            usrs_info = {'admin': 'admin'}
            pickle.dump(usrs_info, usr_file)
    if usr_name in usrs_info:
        if usr_pwd == usrs_info[usr_name]:
            # 登录成功后的功能，可以改为其它功能
            if_run = tk.messagebox.askyesno('Welcome',
                                                'Do you want to train and test new data?')
            if if_run:
                window_sign_up = tk.Toplevel(window)
                window_sign_up.geometry('530x270')
                window_sign_up.title('choose new data to train and test')

                # Train Data框
                tk.Label(window_sign_up, text='Train Data: ').place(x=10, y=10)  # 将`User name:`放置在坐标（10,10）。
                t1 = tk.Text(window_sign_up, width=50, height=1)
                t1.place(x=100, y=10)
                open_TrD = tk.Button(window_sign_up, text='open', command=lambda: file_adress_TrD(t1)).place(x=440, y=7)

                # Train Label框
                tk.Label(window_sign_up, text='Train Label: ').place(x=10, y=60)  # 将`User name:`放置在坐标（10,10）。
                t2 = tk.Text(window_sign_up, width=50, height=1)
                t2.place(x=100, y=60)
                open_TrL = tk.Button(window_sign_up, text='open', command=lambda: file_adress_TrL(t2)).place(x=440, y=57)

                # Test Data框
                tk.Label(window_sign_up, text='Test Data: ').place(x=10, y=110)  # 将`User name:`放置在坐标（10,10）。
                t3 = tk.Text(window_sign_up, width=50, height=1)
                t3.place(x=100, y=110)
                open_TeD = tk.Button(window_sign_up, text='open', command=lambda: file_adress_TeD(t3)).place(x=440, y=107)

                # Test Label框
                tk.Label(window_sign_up, text='Test Label: ').place(x=10, y=160)  # 将`User name:`放置在坐标（10,10）。
                t4 = tk.Text(window_sign_up, width=50, height=1)
                t4.place(x=100, y=160)
                open_TeL = tk.Button(window_sign_up, text='open', command=lambda: file_adress_TeL(t4)).place(x=440,y=157)


                # train and test按钮
                train_and_test = tk.Button(window_sign_up, text='train and test', command=lambda: save(True)).place(x=50, y=207)

                # only test按钮
                only_test = tk.Button(window_sign_up, text='only test', command=lambda: restore_net_parameters()).place(x=400, y=207)

            else:
                save(False)



        else:
            tk.messagebox.showerror(message='Error, your password is wrong, try again')
    else:  # 如果发现用户名不存在
        is_sign_up = tk.messagebox.askyesno('Welcome',
                           'You have not sign up yet. Sign up today?')
        # 提示需不需要注册新用户
        if is_sign_up:
            usr_sign_up()


# 注册功能
def usr_sign_up():
    def sign_to_Mofan_Python():
        # 以下三行就是获取我们注册时所输入的信息
        np = new_pwd.get()
        npf = new_pwd_confirm.get()
        nn = new_name.get()

        # 这里是打开我们记录数据的文件，将注册信息读出
        with open('usrs_info.pickle', 'rb') as usr_file:
            exist_usr_info = pickle.load(usr_file)
            # 这里就是判断，如果两次密码输入不一致，则提示`'Error', 'Password and confirm password must be the same!'`
            if np != npf:
                tk.messagebox.showerror('Error', 'Password and confirm password must be the same!')
            # 如果用户名已经在我们的数据文件中，则提示`'Error', 'The user has already signed up!'`
            elif nn in exist_usr_info:
                tk.messagebox.showerror('Error', 'The user has already signed up!')
            else:
                exist_usr_info[nn] = np
                with open('usrs_info.pickle', 'wb') as usr_file:
                    pickle.dump(exist_usr_info, usr_file)
                tk.messagebox.showinfo('Welcome', 'You have successfully signed up!')
                window_sign_up.destroy()


    # 这里就是在主体窗口的window上创建一个Sign up window窗口
    window_sign_up = tk.Toplevel(window)
    window_sign_up.geometry('350x200')
    window_sign_up.title('注册个人账号')

    # 用户名
    new_name = tk.StringVar()
    new_name.set('R&C')
    tk.Label(window_sign_up, text='User name: ').place(x=10, y= 10)  # 将`User name:`放置在坐标（10,10）。
    entry_new_name = tk.Entry(window_sign_up, textvariable=new_name)  # 设置输入姓名框
    entry_new_name.place(x=150, y=10)  # `entry`放置在坐标（150,10）

    # 初次密码
    new_pwd = tk.StringVar()
    tk.Label(window_sign_up, text='Password: ').place(x=10, y=50)
    entry_usr_pwd = tk.Entry(window_sign_up, textvariable=new_pwd, show='*')
    entry_usr_pwd.place(x=150, y=50)

    # 确认密码
    new_pwd_confirm = tk.StringVar()
    tk.Label(window_sign_up, text='Confirm password: ').place(x=10, y=90)
    entry_usr_pwd_confirm = tk.Entry(window_sign_up, textvariable=new_pwd_confirm, show='*')
    entry_usr_pwd_confirm.place(x=150, y=90)

    # 下面的 sign_to_Mofan_Python
    btn_comfirm_sign_up = tk.Button(window_sign_up, text='Sign up', command=sign_to_Mofan_Python)
    btn_comfirm_sign_up.place(x=150, y=130)


# 登录和注册按钮
btn_login = tk.Button(window, text='log in', command=usr_login)  # 定义一个`button`按钮，名为`Login`,触发命令为`usr_login'
btn_login.place(x=270, y=230)
btn_sign_up = tk.Button(window, text='Sign up', command=usr_sign_up)
btn_sign_up.place(x=120, y=230)

window.mainloop()