# -*- coding: utf-8 -*-
import tkinter as tk
import tkinter.messagebox
import pickle
import torch
from train import LSTM_Net
from data_get import get_data
from test import senti_dict
from torchvision import transforms


# 窗口
window = tk.Tk()
window.title('欢迎进入情感分类系统')
window.geometry('530x300')
# 画布放置图片
canvas = tk.Canvas(window, height=300, width=550)
imagefile = tk.PhotoImage(file="C:\\Users\\llhy\\Desktop\\practical_workshop\\senti_sys\\photos\\2.gif")
image = canvas.create_image(0, 0, anchor='nw', image=imagefile)
canvas.pack(side='top')
# 标签 用户名密码
tk.Label(window, text='用户名:').place(x=150, y=150)
tk.Label(window, text='密码:').place(x=150, y=190)
# 用户名输入框
var_usr_name = tk.StringVar()
entry_usr_name = tk.Entry(window, textvariable=var_usr_name)
entry_usr_name.place(x=200, y=150)
# 密码输入框
var_usr_pwd = tk.StringVar()
entry_usr_pwd = tk.Entry(window, textvariable=var_usr_pwd, show='*')
entry_usr_pwd.place(x=200, y=190)


# 登录函数
def usr_log_in():
    def multiple_class():
        def senti_class():
            text1.delete('1.0', 'end')
            data = train.get()
            batch_size = 100
            device = torch.device("cpu")
            output_size = 6
            hidden_dim = 200
            n_layers = 1
            embedding_dim = 200
            test0 = [data]
            res = pickle.load(open('./train/dict', 'rb'))

            test_data = get_data(test0, 1, res)
            tran = transforms.ToTensor()
            test_data = tran(test_data)
            test_data = torch.squeeze(test_data, 1)
            # print(test_data)

            model = LSTM_Net(output_size, hidden_dim, n_layers, embedding_dim, batch_size, device, res['len']).to(device)
            model.load_state_dict(torch.load('./result/GRU.pth'), strict=False)
            h = model.init_hidden(1)
            ans = model(test_data, h)
            diction = senti_dict("./dataset/train/train.xlsx")
            ans = diction[int(ans[0].argmax(dim=1))]
            text1.insert('0.0', ans)

        window_sign_up.destroy()
        window_1 = tk.Toplevel(window)
        window_1.geometry('350x400')
        window_1.title('情感分析系统')

        train = tk.StringVar()
        tk.Label(window_1, text='需要进行情感分类的内容：').place(x=10, y=10)
        tk.Entry(window_1, textvariable=train).place(x=20, y=40, height=100, width=300)

        tk.Label(window_1, text='情感分析结果：').place(x=70, y=250)
        text1 = tk.Text(window_1, height=1, width=8)
        text1.place(x=155, y=250)

        bt_1 = tk.Button(window_1, text='开始分析情感', command=senti_class)
        bt_1.place(x=130, y=300)

    def binary_class():
        def senti_class():
            text1.delete('1.0', 'end')
            data = train.get()
            batch_size = 100
            device = torch.device("cpu")
            output_size = 2
            hidden_dim = 200
            n_layers = 1
            embedding_dim = 200
            test0 = [data]
            res = pickle.load(open('./train/dict_', 'rb'))

            test_data = get_data(test0, 1, res)
            tran = transforms.ToTensor()
            test_data = tran(test_data)
            test_data = torch.squeeze(test_data, 1)

            model = LSTM_Net(output_size, hidden_dim, n_layers, embedding_dim, batch_size, device, res['len']).to(device)
            model.load_state_dict(torch.load('./result/LSTM_b.pth'), strict=False)
            h = model.init_hidden(1)
            ans = model(test_data, h)
            diction = senti_dict("./dataset/train/b_train.xlsx")
            ans = diction[int(ans[0].argmax(dim=1))]
            if ans:
                ans = 'positive'
            else:
                ans = 'negative'
            text1.insert('0.0', ans)

        window_sign_up.destroy()
        window_1 = tk.Toplevel(window)
        window_1.geometry('350x400')
        window_1.title('情感分析系统')

        train = tk.StringVar()
        tk.Label(window_1, text='需要进行情感分类的内容：').place(x=10, y=10)
        tk.Entry(window_1, textvariable=train).place(x=20, y=40, height=100, width=300)

        tk.Label(window_1, text='情感分析结果：').place(x=70, y=250)
        text1 = tk.Text(window_1, height=1, width=8)
        text1.place(x=155, y=250)

        bt_1 = tk.Button(window_1, text='开始分析情感', command=senti_class)
        bt_1.place(x=130, y=300)

    # 输入框获取用户名密码
    usr_name = var_usr_name.get()
    usr_pwd = var_usr_pwd.get()
    # 从本地字典获取用户信息，如果没有则新建本地数据库
    try:
        with open('usr_info.pickle', 'rb') as usr_file:
            usrs_info = pickle.load(usr_file)
    except FileNotFoundError:
        with open('usr_info.pickle', 'wb') as usr_file:
            usrs_info = {'admin': 'admin'}
            pickle.dump(usrs_info, usr_file)
    # 判断用户名和密码是否匹配
    if usr_name in usrs_info:
        if usr_pwd == usrs_info[usr_name]:
            tk.messagebox.showinfo(title='welcome',
                                   message='欢迎您：' + usr_name)
            window_sign_up = tk.Toplevel(window)
            window_sign_up.geometry('300x150')
            window_sign_up.title('选择模式')
            # 用户名变量及标签、输入框
            tk.Label(window_sign_up, text='请选择中文情感分类的方式').place(x=40, y=15)
            mode_1 = tk.Button(window_sign_up, text='多分类', command=multiple_class)
            mode_1.place(x=70, y=50)
            mode_2 = tk.Button(window_sign_up, text='二分类', command=binary_class)
            mode_2.place(x=160, y=50)

        else:
            tk.messagebox.showerror(message='密码错误')
    # 用户名密码不能为空
    elif usr_name == '' or usr_pwd == '':
        tk.messagebox.showerror(message='用户名或密码为空')
    # 不在数据库中弹出是否注册的框
    else:
        is_signup = tk.messagebox.askyesno('欢迎', '您还没有注册，是否现在注册')
        if is_signup:
            usr_sign_up()


# 注册函数
def usr_sign_up():
    # 确认注册时的相应函数
    def signtowcg():
        # 获取输入框内的内容
        nn = new_name.get()
        np = new_pwd.get()
        npf = new_pwd_confirm.get()

        # 本地加载已有用户信息,如果没有则已有用户信息为空
        try:
            with open('usr_info.pickle', 'rb') as usr_file:
                exist_usr_info = pickle.load(usr_file)
        except FileNotFoundError:
            exist_usr_info = {}

            # 检查用户名存在、密码为空、密码前后不一致
        if nn in exist_usr_info:
            tk.messagebox.showerror('错误', '用户名已存在')
        elif np == '' or nn == '':
            tk.messagebox.showerror('错误', '用户名或密码为空')
        elif np != npf:
            tk.messagebox.showerror('错误', '密码前后不一致')
        # 注册信息没有问题则将用户名密码写入数据库
        else:
            exist_usr_info[nn] = np
            with open('usr_info.pickle', 'wb') as usr_file:
                pickle.dump(exist_usr_info, usr_file)
            tk.messagebox.showinfo('欢迎', '注册成功')
            # 注册成功关闭注册框
            window_sign_up.destroy()

    # 新建注册界面
    window_sign_up = tk.Toplevel(window)
    window_sign_up.geometry('350x200')
    window_sign_up.title('注册')
    # 用户名变量及标签、输入框
    new_name = tk.StringVar()
    tk.Label(window_sign_up, text='用户名：').place(x=10, y=10)
    tk.Entry(window_sign_up, textvariable=new_name).place(x=150, y=10)
    # 密码变量及标签、输入框
    new_pwd = tk.StringVar()
    tk.Label(window_sign_up, text='请输入密码：').place(x=10, y=50)
    tk.Entry(window_sign_up, textvariable=new_pwd, show='*').place(x=150, y=50)
    # 重复密码变量及标签、输入框
    new_pwd_confirm = tk.StringVar()
    tk.Label(window_sign_up, text='请再次输入密码：').place(x=10, y=90)
    tk.Entry(window_sign_up, textvariable=new_pwd_confirm, show='*').place(x=150, y=90)
    # 确认注册按钮及位置
    bt_confirm_sign_up = tk.Button(window_sign_up, text='确认注册',
                                   command=signtowcg)
    bt_confirm_sign_up.place(x=150, y=130)


# 退出的函数
def usr_sign_quit():
    window.destroy()


# 登录 注册按钮
bt_login = tk.Button(window, text='登录', command=usr_log_in)
bt_login.place(x=150, y=230)
bt_logup = tk.Button(window, text='注册', command=usr_sign_up)
bt_logup.place(x=230, y=230)
bt_logquit = tk.Button(window, text='退出', command=usr_sign_quit)
bt_logquit.place(x=310, y=230)
# 主循环
window.mainloop()