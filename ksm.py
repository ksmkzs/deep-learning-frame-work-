import weakref, contextlib, os, subprocess, math, MeCab, collections

from PIL import Image

import pandas as pd

import matplotlib.pyplot as plt

try:

    import cupy as np

    import cupyx

    gpu_enable = True

except ImportError:

    gpu_enable = False

    import numpy as np

#----------------------------------------------------------------------------------------

# Config

#----------------------------------------------------------------------------------------

class Config:

    enable_backcrop = True

    train = True









@contextlib.contextmanager

def using_config(name, value):

# with文で切り替えをしたいコンフィグの名前をname、オンオフをvalueで引数とする

    old_value = getattr(Config, name)

    setattr(Config, name, value)

    try:

        yield

    finally:

        setattr(Config, name, old_value)

# with中はコンフィグをvalueに切り替えて抜け出すときに元のvalueに戻す

def test_mode():

    return using_config("train", False)



def no_grad():

    return using_config("enable_backcrop", False)

# 逆伝播の無効化を簡略化



#----------------------------------------------------------------------------------------

# Variable

#----------------------------------------------------------------------------------------

# 入力はndarrayインスタンスを想定 self.dataにデータを格納し、self.creatorに出力元の関数を設定

# backwardをするとcreatorの関数を呼び、一つ前のVar変数を呼びその勾配を求める。そのVar変数のcreator関数を求めて勾配を求めることを繰り返す



class Variable:

    __array_priority__ = 200

    # 演算子を利用可能にするためのおまじない

    def __init__(self, data, name=None):

    # ndarrayのみを想定 

        if data is not None:

            if not isinstance(data, np.ndarray):

                raise TypeError(f"{type(data)} is not supported")

                # Variableの入力データがndarrayでない時にエラーを出力

        self.data = data

        # self.dataにデータを格納

        self.grad = None

        # 勾配を定義

        self.creator = None

        # creatorを定義

        self.generation = 0

        # 世代(計算グラフを作るときに使う)を定義

        self.name = name

        # 変数に名前を設定



    def set_creator(self, func):

    # 引数はFunction関数を想定 出力元の関数を定義

        self.creator = func

        # creator関数を設定

        self.generation = func.generation + 1

        #世代を関数の一個後に定義

    

    def backward(self, retain_grad=False, create_graph=False):

        if self.grad is None:

            self.grad = Variable(np.ones_like(self.data))

            # 最初のgy=1を出力データの形状に合わせて設定する Var変数として保存する

        funcs = []

        seen_set = set()

        # funcs内に関数を格納し、関数に入力されたVariable変数の勾配を求めるとfuncから消去する

        def add_func(f):

            if f not in seen_set:

                funcs.append(f)

                seen_set.add(f)

                funcs.sort(key= lambda x: x.generation)

        add_func(self.creator)

        # 関数をfuncsに追加して世代順に並び替える



        while funcs:

            f = funcs.pop()

            # funcs.popでは一番世代の遅い関数が呼び出される

            gys = [output().grad for output in f.outputs]

            # 関数から出力されたVar変数の勾配をリストとしてまとめる 弱参照なので()をつける

            with using_config("enable_backcrop", create_graph):

            # 逆伝播のさらに逆伝播はいらないためデフォルトでオフにしておく　gradもVariableなので逆伝播に関連する変数が保存されてしまう

                gxs = f.backward(*gys)

                # 関数に入力されたVar変数の勾配を求め、リストとしてまとめる

                if not isinstance(gxs, tuple):

                    gxs = (gxs,)

                for x , gx in zip(f.inputs, gxs):

                    if x.grad is None:

                        x.grad = gx

                    else:

                        x.grad = x.grad + gx

                # 関数に入力されたVar変数のself.gradに勾配を入れる

                # 同じxを使いまわした時(例：add(x,x))に勾配が初期化されないように加算する



                    if x.creator is not None:

                        add_func(x.creator)

                    # funcsに関数を追加する。その際に世代順に並び替えを行う

                

                if not retain_grad:

                    for y in f.outputs:

                        y().grad = None

                # retain_gradがFalseのときWやbなどの末端の勾配(creatorが存在しないvar変数)以外の勾配を削除する

    

    def cleargrad(self):

        self.grad = None



    def to_cpu(self):

        if self.data is not None:

            self.data = as_numpy(self.data)



    def unchain(self):

        self.creator = None

    

    def unchain_backward(self):

        if self.creator is not None:

            funcs = [self.creator]

            while funcs:

                f = funcs.pop()

                for x in f.inputs:

                    if x.creator is not None:

                        funcs.append(x.creator)

                        x.unchain()

    # 主にRNNで用いる関数二つ backwardでつながりを断ち切ることで逆伝播を効率よく行う



    #--------------------------------------------------

    # オーバーロード

    #--------------------------------------------------

    @property

    def shape(self):

        return self.data.shape

    # x.shapeとしたときにdataのshapeを表示できる



    @property

    def ndim(self):

        return self.data.ndim



    @property

    def size(self):

        return self.data.size



    @property

    def dtype(self):

        return self.data.dtype



    def __len__(self):

        return len(self.data)



    def __repr__(self):

        if self.data is None:

            return "Variable(None)"

        p = str(self.data).replace("\n", "\n" + " " * 9)

        return "Variable(" + p + ")"

    #print関数が呼ばれたときに綺麗に表示されるようにする



    def reshape(self, *shape):

        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):

            shape = shape[0]

        return reshape(self, shape)

    # var変数で　x.reshape(a)のように使えるようにする



    def trasnpose(self):

        return transpose(self)

    

    @property

    def T(self):

        return transpose(self)

    # x.Tのような転値の使い方が可能に



class Parameter(Variable):

    pass

#parameterクラスの変数をvarと同じ設定で定義、varとは区別できる







#----------------------------------------------------------------------------------------

# Function

#----------------------------------------------------------------------------------------

# __call__ではVariable変数を引数としforwardを行いVariable変数を出力する

# self.inputs、self.outputsはリスト



class Function:

    def __call__(self, *inputs):

    # inputは任意の数のVariableを想定 

        inputs = [as_variable(x) for x in inputs]

        # 計算に含まれる数にndarrayがあったときにVariableに変換する

        xs = [x.data for x in inputs]

        # Variableからデータをとりだす

        # 入力データが一つの場合の x = input.data をinputが複数の場合に拡張

        ys = self.forward(*xs)

        # Functionを呼び出したクラスのforwardを用いることで出力結果を出す 引数には可変長引数を指定

        if not isinstance(ys, tuple):

            ys = (ys,)

        # ysはリストでなければならないためリストでないときにタプルに変更する

        ys = [as_array(y) for y in ys]

        # yをndarrayインスタンスに揃える

        # 入力データが一つの時の y = as_array(y) を拡張

        outputs = [Variable(y) for y in ys]

        # 出力結果をVariable変数にする

        if Config.enable_backcrop:

        # 勾配を求める必要がない時はoutputsやinputsを求める必要がない

            self.generation = max([x.generation for x in inputs])

            # 世代を変数のうち一番遅い世代と同じに設定

            for output in outputs:

                output.set_creator(self) 

            # 出力先のcreatorを自分に設定

            self.inputs = inputs

            # 入力を保存

            self.outputs = [weakref.ref(output) for output in outputs]

            # 出力した変数を覚える 弱参照にすることで必要がなくなった瞬間にメモリからデータを削除しメモリの無駄遣いを無くす

        return outputs if len(outputs) > 1 else outputs[0]

        # 出力はVariable変数 基本リストだが一つしか持たない場合はその要素を返す

    

    def forward(self, x):

        raise NotImplementedError()

        # Functionクラスのforwardは想定されていない

    

    def backward(self, gy):

        raise NotImplementedError()

        # Functionクラスのbackwardは想定されていない





#----------------------------------------------------------------------------------------

# Layer

#----------------------------------------------------------------------------------------

#layerは基底クラスとして設定、主にパラメータを保存出力する役目を持つ



class Layer:

    def __init__(self):

        self._params = set()

        #_params内にLayerが持つパラメータを保持



    def __setattr__(self, name, value):

        if isinstance(value, (Parameter, Layer)):

            self._params.add(name)

        super().__setattr__(name, value)

    # layer.l1の様にlayer内のインスタンス変数を設定した時にそれがparameterだった場合_paramsに名前を保管できる

    # layer.__dict__[name]でname内のvalueを取り出せる



    def __call__(self, *inputs):

        outputs = self.forward(*inputs)

        if not isinstance(outputs, tuple):

            outputs = (outputs,)

        self.inputs = [weakref.ref(x) for x in inputs]

        self.outputs = [weakref.ref(y) for y in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    # 呼ばれたときにoutputsとinputを保管　他はfunctionの__call__と同じ



    def forward(self, inputs):

        raise NotImplementedError()

    

    def params(self):

        for name in self._params:

            obj = self.__dict__[name]

            if isinstance(obj, Layer):

                yield from obj.params()

            else:

                yield obj

    # paramsが呼ばれたとき保存されているparamsのvalueを全て出力する。paramsにLayerが保存されている場合そのLayer内のparamsを全て出力



    def cleargrads(self):

        for param in self.params():

            param.cleargrad()

    #全てのパラメータの勾配を初期化させる



    def to_cpu(self):

        for param in self.params():

            param.to_cpu()



    def _flatten_params(self, params_dict, parent_key=""):

        for name in self._params:

            obj = self.__dict__[name]

            key = parent_key + "/" + name if parent_key else name

            if isinstance(obj, Layer):

                obj._flatten_params(params_dict, parent_key=key)

            else:

                params_dict[key] = obj

    # 重みを保存するための関数　L1のW1はL1/W1というような名前でparams_dictに保存される



    def save_weights(self, path):

        self.to_cpu()



        params_dict = {}

        self._flatten_params(params_dict)

        array_dict = {key: param.data for key, param in params_dict.items() if param is not None}

        try:

            np.savez_compressed(path, **array_dict)

        except(Exception, KeyboardInterrupt) as e:

            if os.path.exists(path):

                os.remove(path)

            raise

    # pathに重みを保存する。ディクショナリ型で保存される



    def load_weights(self, path):

        npz = np.load(path)

        params_dict = {}

        self._flatten_params(params_dict)

        for key, param in params_dict.items():

            param.data = npz[key]

    # Layer内の特定の重みとpath内の重みを_flatten_paramsで結びつけ、for文内で各パラメータにロードした値を入力する



    











#----------------------------------------------------------------------------------------

# Function commons

#----------------------------------------------------------------------------------------

# 実際に処理をする関数

# 入力はx=数値(可変長)、出力も数値を想定

# Function.__call__内のself.forwardによって出力が実行される

# 逆伝播の入力はgy=数値のみ 出力はVariable変数を想定

class Square(Function):

    def forward(self, x):

        return x ** 2



    def backward(self, gy):

        x = self.inputs[0]

        gx = 2 * x * gy

        return gx



def square(x):

    return Square()(x)

# x^2



class Exp(Function):

    def forward(self, x):

        return np.exp(x)

    

    def backward(self, gy):

        x = self.inputs[0]

        gx = np.exp(x) * gy

        return gx



def exp(x):

    return Exp()(x)

# e^x



class Add(Function):

    def forward(self, x0, x1):

        self.x0_shape, self.x1_shape = x0.shape, x1.shape

        y = x0 + x1

        return y

    

    def backward(self, gy):

        gx0, gx1 = gy, gy

        if self.x0_shape != self.x1_shape:

            gx0 = sum_to(gx0, self.x0_shape)

            gx1 = sum_to(gx1, self.x1_shape)

        # x0とx1の形状が違う時自動でブロードキャストが行われるためbackwardでsum_toをする必要がある

        return gy, gy



def add(x0, x1):

    x1 = as_array(x1)

    # x1がfloatやintの時にndarryに変換→その後Functionの__call__でVariableに変換

    return Add()(x0, x1)

# x0 + x1



class Mul(Function):

    def forward(self, x0, x1):

        y = x0 * x1

        return y

    

    def backward(self, gy):

        x0,x1 = self.inputs

        return x1 * gy, x0 * gy



def mul(x0, x1):

    x1 = as_array(x1)

    # x1がfloatやintの時にndarryに変換→その後Functionの__call__でVariableに変換

    return Mul()(x0, x1)

# x0 * x1



class Neg(Function):

    def forward(self,x):

        return -x



    def backward(self, gy):

        return -gy



def neg(x):

    return Neg()(x)

# -x



class Sub(Function):

    def forward(self, x0, x1):

        y = x0 - x1

        return y

    def backward(self, gy):

        return gy, -gy



def sub(x0, x1):

    x1 = as_array(x1)

    return Sub()(x0, x1)

# x0 - x1



def rsub(x0, x1):

    x1 = as_array(x1)

    return Sub()(x1, x0)

# x1 - x0



class Div(Function):

    def forward(self, x0, x1):

        return x0 / x1

    

    def backward(self, gy):

        x0,x1 = self.inputs

        gx0 = gy / x1 

        gx1 = - gy * x0 / (x1**2)

        return gx0, gx1



def div(x0, x1):

    x1 = as_array(x1)

    as_variable(x1)

    return Div()(x0, x1)

# x0 / x1



def rdiv(x0, x1):

    x1 = as_array(x1)

    as_variable(x1)

    return Div()(x1, x0)

# x1 / x0



class Pow(Function):

    def __init__(self, c):

        self.c = c



    def forward(self, x):

        return x ** self.c

    

    def backward(self, gy):

        x = self.inputs[0]

        c = self.c

        gx = gy * c * (x ** (c-1))

        return gx



def pow(x, c):

    return Pow(c)(x)

# x^c



class Log(Function):

    def forward(self, x):

        y = np.log(x)

        return y



    def backward(self, gy):

        x, = self.inputs

        gx = gy / x

        return gx





def log(x):

    return Log()(x)

# log(x)



class Sin(Function):

    def forward(self, x):

        y = np.sin(x)

        return y

    

    def backward(self, gy):

        x = self.inputs[0]

        gx = cos(x) * gy

        return gx



def sin(x):

    return Sin()(x)

#sin(x)



class Cos(Function):

    def forward(self, x):

        y = np.cos(x)

        return y



    def backward(self, gy):

        x = self.inputs[0]

        gx = gy * -sin(x)

        return gx



def cos(x):

    return Cos()(x)



class Tanh(Function):

    def forward(self, x):

        y = np.tanh(x)

        return y



    def backward(self, gy):

        y = self.outputs[0]()  # weakref

        gx = gy * (1 - y * y)

        return gx



def tanh(x):

    return Tanh()(x)



class Reshape(Function):

    def __init__(self, shape):

        self.shape = shape



    def forward(self, x):

        self.x_shape = x.shape

        y = x.reshape(self.shape)

        return y

    

    def backward(self, gy):

        return reshape(gy, self.x_shape)



def reshape(x, shape):

    return Reshape(shape)(x)

# xの形状をshapeに変換する　逆伝播はgyの形状をxの形状にするだけでいい



class Transpose(Function):

    def __init__(self, axes=None):

        self.axes = axes

        # axesは転置の方向を指定する時に使われる



    def forward(self, x):

        y = x.transpose(self.axes)

        return y



    def backward(self, gy):

        if self.axes is None:

            return transpose(gy)



        axes_len = len(self.axes)

        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))

        return transpose(gy, inv_axes)

        # 転置の方向を指定した場合元のデータに戻す

def transpose(x, axes=None):

    return Transpose(axes)(x)

# 転値を行う関数



class Sum(Function):

    def __init__(self, axis, keepdims):

        self.axis = axis

        self.keepdims = keepdims



    def forward(self, x):

        self.x_shape = x.shape

        y = x.sum(axis=self.axis, keepdims=self.keepdims)

        return y

    

    def backward(self, gy):

        gy = reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)

        gx = broadcast_to(gy, self.x_shape)

        return gx



def sum(x, axis=None, keepdims=False):

    return Sum(axis, keepdims)(x)

# 行列の値を合計する関数　backwardでは元のデータの形状にgyがブロードキャストされる

# axisで指定した方向に沿って和を求められる keepdimsは次元数を保つ





class Broadcast_to(Function):

    def __init__(self, shape):

        self.shape = shape

    def forward(self, x):

        self.x_shape = x.shape

        y = np.broadcast_to(x, self.shape)

        return y

    

    def backward(self, gy):

        gx = sum_to(gy, self.x_shape)

        return gx



def broadcast_to(x, shape):

    return Broadcast_to(shape)(x)

# xをshapeの形状に広げる関数 backwardでは元の形状に合計される



class Sumto(Function):

    def __init__(self, shape):

        self.shape = shape

    

    def forward(self, x):

        self.x_shape = x.shape

        y = def_sum_to(x, self.shape)

        return y



    def backward(self, gy):

        gx = broadcast_to(gy, self.x_shape)

        return gx



def sum_to(x, shape):

    if x.shape ==shape:

        return as_variable(x)

    return Sumto(shape)(x)

# 指定された形状になるように要素を足し合わせる関数



class Matmul(Function):

    def forward(self, x, W):

        y = x.dot(W)

        return y

    

    def backward(self, gy):

        x, W = self.inputs

        gx = matmul(gy, W.T)

        gW = matmul(x.T, gy)

        return gx, gW



def matmul(x, W):

    return Matmul()(x, W)

#行列の積を求める関数



class Clip(Function):

    def __init__(self, x_min, x_max):

        self.x_min = x_min

        self.x_max = x_max



    def forward(self, x):

        y = np.clip(x, self.x_min, self.x_max)

        return y



    def backward(self, gy):

        x, = self.inputs

        mask = (x.data >= self.x_min) * (x.data <= self.x_max)

        gx = gy * mask

        return gx





def clip(x, x_min, x_max):

    return Clip(x_min, x_max)(x)

#clip ※要検討



class Sigmoid(Function):

    def forward(self, x):

        y = 1 / (1 + np.exp(-x))

        return y



    def backward(self, gy):

        y = self.outputs[0]()

        gx = gy * y * (1 - y)

        return gx



def sigmoid(x):

    return Sigmoid()(x)

#sigmoid(x)



class Relu(Function):

    def forward(self, x):

        y = np.maximum(0.0, x)

        return y



    def backward(self, gy):

        x = self.inputs[0]

        mask = x.data > 0

        gx = gy * mask

        return gx

        # 入力がTrue(x)ならgyを、False(0)なら0を返す



def relu(x):

    return Relu()(x)





class FLinear(Function):

    def forward(self, x, W, b=None):

        y = x.dot(W)

        if b is not None:

            y = y + b

        return y

    

    def backward(self, gy):

        x, W, b = self.inputs

        gb = None if b.data is None else sum_to(gy, b.shape)

        #bはブロードキャストされているため勾配はsum_toで求められる

        gx = matmul(gy, W.T)

        gW = matmul(x.T, gy)

        return gx, gW, gb



def linear(x, W, b):

    return FLinear()(x, W, b)

# 全結合をする関数

            

class Getitem(Function):

    def __init__(self, slices):

        self.slices = slices

    

    def forward(self, x):

        y = x[self.slices]

        return y

    

    def backward(self, gy):

        x = self.inputs[0]

        f = Getitemgrad(self.slices, x.shape)

        return f(gy)



def get_item(x, slices):

    return Getitem(slices)(x)

# sliceを行う関数 x[2:4]などが可能になる



class Getitemgrad(Function):

    def __init__(self, slices, in_shape):

        self.slices = slices

        self.in_shape = in_shape

    

    def forward(self, gy):

        gx = np.zeros(self.in_shape)

        if gpu_enable:
            cupyx.scatter_add(gx, self.slices, gy)
        else:
            np.add.at(gx, self.slices, gy)

        return gx

    

    def backward(self, ggx):



        return get_item(ggx, self.slices)



# getitem関数のbackwardで用いる関数、スライスの対象にのみgyを与える









class Softmax(Function):

    def __init__(self, axis=1):

        self.axis = axis



    def forward(self, x):

        y = x - x.max(axis=self.axis, keepdims=True)

        # e(x)はオーバーフローを起こすので最大値Cを引くことでオーバーフロー対策を行う

        y = np.exp(y)

        y /= y.sum(axis=self.axis, keepdims=True)

        return y



    def backward(self, gy):

        y = self.outputs[0]()

        gx = y * gy

        sumdx = gx.sum(axis=self.axis, keepdims=True)

        gx -= y * sumdx

        return gx





def softmax(x, axis=1):

    return Softmax(axis)(x)

#softmax(x)



class SoftmaxCrossEntropy(Function):

    def forward(self, x, t):

    # tはone_hotではなく正解ラベルの数値が与えられていることを想定

        N = x.shape[0]

        # バッチ数

        log_z = logsumexp(x, axis=1)

        # C + log Σexp(x-C)

        log_p = x - log_z

        # x - {C + log Σexp(x-C)}

        # = log{exp(x-C)/Σexp(x-C)}

        # = log(softmax(x))

        log_p = log_p[np.arange(N), t.ravel()]

        # 教師データに対応するモデルの出力が取り出され、要素がN個の一次元配列にまとめられる

        y = -log_p.sum() / np.float32(N)

        # 全てのtk * logpkを足し合わせバッチ数で割る

        return y



    def backward(self, gy):

        x, t = self.inputs

        N, CLS_NUM = x.shape

        # CLS_NUMは選択肢の数

        gy *= 1 / N

        y = softmax(x)

        t_onehot = np.eye(CLS_NUM, dtype=t.dtype)[t.data]

        # 単位行列を作成し、t.dataの列をt_onehotに出力することでラベルをone-hot表現に変換する

        y = (y - t_onehot) * gy

        return y

# 交差エントロピー誤差を求める 



def softmax_cross_entropy(x, t):

    return SoftmaxCrossEntropy()(x, t)





class Sigmoidwithloss(Function):

    def forward(self, x, t):

        y = 1 / (1 + np.exp(-x))

        L = -(t * np.log(1.0 - y) + (1.0 - t) * np.log(y))

        y = np.sum(L) / len(x)

        return y

    

    def backward(self, gy):

        x, t = self.inputs

        y = sigmoid(x)

        N = len(x)

        gy *= 1 / N

        y = (y - t) * gy

        return y



    

def sigmoidwithloss(x, t):

    t = as_array(t)

    return Sigmoidwithloss()(x, t)



def sigmoid_cross_entropy(x, t):

    t = as_array(t)

    x, t = as_variable(x), as_variable(t)

    N = len(x)

    p = sigmoid(x)

    p = clip(p, 1e-15, 1.0)

    tlog_p = t * log(p) + (1 - t) * log(1 - p)

    y = -1 * sum(tlog_p) / N

    return y











def mean_squared_error_simple(x0, x1):

    x0, x1 = as_variable(x0), as_variable(x1)

    diff = x0 - x1

    y = sum(diff ** 2) / len(diff)

    return y





class MeanSquaredError(Function):

    def forward(self, x0, x1):

        diff = x0 - x1

        y = (diff ** 2).sum() / len(diff)

        return y



    def backward(self, gy):

        x0, x1 = self.inputs

        diff = x0 - x1

        gx0 = gy * diff * (2. / len(diff))

        gx1 = -gx0

        return gx0, gx1





def mean_squared_error(x0, x1):

    return MeanSquaredError()(x0, x1)



class FConv2d(Function):

    def __init__(self, stride=1, pad=0):

        super().__init__()

        self.stride = pair(stride)

        self.pad = pair(pad)



    def forward(self, x, W, b):



        KH, KW = W.shape[2:]

        col = im2col_array(x, (KH, KW), self.stride, self.pad, to_matrix=False)



        y = np.tensordot(col, W, ((1, 2, 3), (1, 2, 3)))

        if b is not None:

            y += b

        y = np.rollaxis(y, 3, 1)

        # y = np.transpose(y, (0, 3, 1, 2))

        return y



    def backward(self, gy):

        x, W, b = self.inputs

        # ==== gx ====

        gx = deconv2d(gy, W, b=None, stride=self.stride, pad=self.pad,

                      outsize=(x.shape[2], x.shape[3]))

        # ==== gW ====

        gW = Conv2DGradW(self)(x, gy)

        # ==== gb ====

        gb = None

        if b.data is not None:

            gb = gy.sum(axis=(0, 2, 3))

        return gx, gW, gb





def conv2d(x, W, b=None, stride=1, pad=0):

    return FConv2d(stride, pad)(x, W, b)



class Deconv2d(Function):

    def __init__(self, stride=1, pad=0, outsize=None):

        super().__init__()

        self.stride = pair(stride)

        self.pad = pair(pad)

        self.outsize = outsize



    def forward(self, x, W, b):

        Weight = W

        SH, SW = self.stride

        PH, PW = self.pad

        C, OC, KH, KW = Weight.shape

        N, C, H, W = x.shape

        if self.outsize is None:

            out_h = get_deconv_outsize(H, KH, SH, PH)

            out_w = get_deconv_outsize(W, KW, SW, PW)

        else:

            out_h, out_w = pair(self.outsize)

        img_shape = (N, OC, out_h, out_w)



        gcol = np.tensordot(Weight, x, (0, 1))

        gcol = np.rollaxis(gcol, 3)

        y = col2im_array(gcol, img_shape, (KH, KW), self.stride, self.pad,

                         to_matrix=False)

        # b, k, h, w

        if b is not None:

            self.no_bias = True

            y += b.reshape((1, b.size, 1, 1))

        return y



    def backward(self, gy):

        x, W, b = self.inputs



        # ==== gx ====

        gx = conv2d(gy, W, b=None, stride=self.stride, pad=self.pad)

        # ==== gW ====

        f = Conv2DGradW(self)

        gW = f(gy, x)

        # ==== gb ====

        gb = None

        if b.data is not None:

            gb = gy.sum(axis=(0, 2, 3))

        return gx, gW, gb





def deconv2d(x, W, b=None, stride=1, pad=0, outsize=None):

    return Deconv2d(stride, pad, outsize)(x, W, b)

#Conv2dの逆伝播



class Conv2DGradW(Function):

    def __init__(self, conv2d):

        W = conv2d.inputs[1]

        kh, kw = W.shape[2:]

        self.kernel_size = (kh, kw)

        self.stride = conv2d.stride

        self.pad = conv2d.pad



    def forward(self, x, gy):

        col = im2col_array(x, self.kernel_size, self.stride, self.pad,

                           to_matrix=False)

        gW = np.tensordot(gy, col, ((0, 2, 3), (0, 4, 5)))

        return gW



    def backward(self, gys):

        x, gy = self.inputs

        gW, = self.outputs



        xh, xw = x.shape[2:]

        gx = deconv2d(gy, gW, stride=self.stride, pad=self.pad,

                      outsize=(xh, xw))

        ggy = conv2d(x, gW, stride=self.stride, pad=self.pad)

        return gx, ggy

# ※要検討



class Im2col(Function):

    def __init__(self, kernel_size, stride, pad, to_matrix):

        super().__init__()

        self.input_shape = None

        self.kernel_size = kernel_size

        self.stride = stride

        self.pad = pad

        self.to_matrix = to_matrix



    def forward(self, x):

        self.input_shape = x.shape

        y = im2col_array(x, self.kernel_size, self.stride, self.pad,

                         self.to_matrix)

        return y



    def backward(self, gy):

        gx = col2im(gy, self.input_shape, self.kernel_size, self.stride,

                    self.pad, self.to_matrix)

        return gx



def im2col(x, kernel_size, stride=1, pad=0, to_matrix=True):

    y = Im2col(kernel_size, stride, pad, to_matrix)(x)

    return y

# 入力データをフィルタと計算しやすい行列に変換する関数



class Col2im(Function):

    def __init__(self, input_shape, kernel_size, stride, pad, to_matrix):

        super().__init__()

        self.input_shape = input_shape

        self.kernel_size = kernel_size

        self.stride = stride

        self.pad = pad

        self.to_matrix = to_matrix



    def forward(self, x):

        y = col2im_array(x, self.input_shape, self.kernel_size, self.stride,

                         self.pad, self.to_matrix)

        return y



    def backward(self, gy):

        gx = im2col(gy, self.kernel_size, self.stride, self.pad,

                    self.to_matrix)

        return gx





def col2im(x, input_shape, kernel_size, stride=1, pad=0, to_matrix=True):

    return Col2im(input_shape, kernel_size, stride, pad, to_matrix)(x)

# im2colの逆伝播のためにある関数



class Pooling(Function):

    def __init__(self, kernel_size, stride=1, pad=0):

        super().__init__()

        self.kernel_size = kernel_size

        self.stride = stride

        self.pad = pad



    def forward(self, x):

        col = im2col_array(x, self.kernel_size, self.stride, self.pad,

                           to_matrix=False)



        N, C, KH, KW, OH, OW = col.shape

        col = col.reshape(N, C, KH * KW, OH, OW)

        self.indexes = col.argmax(axis=2)

        y = col.max(axis=2)

        return y



    def backward(self, gy):

        return Pooling2DGrad(self)(gy)





class Pooling2DGrad(Function):

    def __init__(self, mpool2d):

        self.mpool2d = mpool2d

        self.kernel_size = mpool2d.kernel_size

        self.stride = mpool2d.stride

        self.pad = mpool2d.pad

        self.input_shape = mpool2d.inputs[0].shape

        self.dtype = mpool2d.inputs[0].dtype

        self.indexes = mpool2d.indexes



    def forward(self, gy):



        N, C, OH, OW = gy.shape

        N, C, H, W = self.input_shape

        KH, KW = pair(self.kernel_size)



        gcol = np.zeros((N * C * OH * OW * KH * KW), dtype=self.dtype)



        indexes = (self.indexes.ravel()

                   + np.arange(0, self.indexes.size * KH * KW, KH * KW))

        

        gcol[indexes] = gy.ravel()

        gcol = gcol.reshape(N, C, OH, OW, KH, KW)

        gcol = np.swapaxes(gcol, 2, 4)

        gcol = np.swapaxes(gcol, 3, 5)



        gx = col2im_array(gcol, (N, C, H, W), self.kernel_size, self.stride,

                          self.pad, to_matrix=False)

        return gx



    def backward(self, ggx):

        f = Pooling2DWithIndexes(self.mpool2d)

        return f(ggx)





class Pooling2DWithIndexes(Function):

    def __init__(self, mpool2d):

        self.kernel_size = mpool2d.kernel_size

        self.stride = mpool2d.stride

        self.pad = mpool2d.pad

        self.input_shpae = mpool2d.inputs[0].shape

        self.dtype = mpool2d.inputs[0].dtype

        self.indexes = mpool2d.indexes



    def forward(self, x):

        col = im2col_array(x, self.kernel_size, self.stride, self.pad,

                           to_matrix=False)

        N, C, KH, KW, OH, OW = col.shape

        col = col.reshape(N, C, KH * KW, OH, OW)

        col = col.transpose(0, 1, 3, 4, 2).reshape(-1, KH * KW)

        indexes = self.indexes.ravel()

        col = col[np.arange(len(indexes)), indexes]

        return col.reshape(N, C, OH, OW)





def pooling(x, kernel_size, stride=1, pad=0):

    return Pooling(kernel_size, stride, pad)(x)

# ※要検討

"""

Embeddingを作ろうとして失敗した例(計算コストばかでかい)

class Embedding(Function):

    def __init__(self):

        self.grad = None

    def forward(self, x, W):

        x = W[x]

        if self.grad is None:

            self.grads = np.zeros_like(W)

        return x



    def backward(self, gy):

        idx = self.inputs[0].data

        gW = Variable(self.grads) 

        for i, word_id in enumerate(idx):

            gW.data[word_id] += gy.data[i]

        return sum(gy,axis=1), gW



def embedding(x, W):

    return Embedding()(x, W)

# embeddingレイヤ(自作) 入力はidxを指定する



class Embeddingdot(Function):

    def __init__(self, idx):

        self.idx = idx

        self.grad = None



    def forward(self, x, W):

        self.x_shape = x.shape

        target_W = W[self.idx]

        self.target_W = target_W 

        y = np.sum(target_W * x, axis=1)   

        return y



    def backward(self, gy):

        x, target_W = self.inputs[0], self.target_W

        gx = reshape_sum_backward(gy, self.x_shape, axis=1, keepdims=False)

        gy = broadcast_to(gx, self.x_shape)

        if self.grad is None:

            self.grad = np.zeros(self.inputs[1].shape)

        gW = self.grad

        for i, idx in enumerate(self.idx):

            gW[idx] += (gy.data[i] * x.data[i])

        return gy * target_W, Variable(gW)



def embeddingdot(x, W, idx):

    return Embeddingdot(idx)(x, W)

"""













        





    

def goldstein(x, y):

    z = (1+(x+y+1)**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2))*(30+(2*x-3*y)**2*(18-32*x+12*x**2+48*y-36*x*y+27*y**2))

    return z







#----------------------------------------------------------------------------------------

# layer commons

#----------------------------------------------------------------------------------------



class Model(Layer):

    def plot(self, *inputs, to_file="model.png"):

        y = self.forward(*inputs)

        return plot_dot_graph(y, verbose=True, to_file=to_file)

# Modelクラスを定義、これにより計算グラフを作ることができる



class Linear(Layer):

    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):

    # 出力サイズを指定する必要がある

        super().__init__()

        # Layerクラスのinitを実行

        self.in_size = in_size

        self.out_size = out_size

        self.dtype = dtype

        self.W = Parameter(None, name="W")

        if in_size is not None:

            self._init_W()

        # 各引数の項目を設定 Wを設定

        if nobias:

            self.b = None

        else:

            self.b = Parameter(np.zeros(out_size, dtype=dtype), name="b")

        # bを設定 形状は出力と同じ



    def _init_W(self):

        I, O = self.in_size, self.out_size

        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)

        self.W.data = W_data

    # Wの値を設定、形状は[in_size, out_size]



    def forward(self, x):

        if self.W.data is None:

            self.in_size = x.shape[1]

            self._init_W()

        y = linear(x, self.W, self.b)

        return y

# 全結合レイヤ



class Twolayernet(Model):

    def __init__(self, hidden_size, out_size):

        super().__init__()

        self.l1 = Linear(hidden_size)

        self.l2 = Linear(out_size)



    def forward(self, x):

        y = sigmoid(self.l1(x))

        y = self.l2(y)

        return y

# 二層の全結合レイヤを保持するレイヤを定義



class Conv2d(Layer):

    def __init__(self, out_channels, kernel_size, stride=1, pad=0, nobias=False, dtype=np.float32, in_channels=None):

        super().__init__()

        self.inchannels = in_channels

        self.out_channels = out_channels

        self.kernel_size = kernel_size

        self.stride = stride

        self.pad = pad

        self.dtype = dtype

        self.W = Parameter(None, name="W")

        if in_channels is not None:

            self._init_W()



        if nobias:

            self.b = None

        else:

            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name="b")

    

    def _init_W(self):

        C, OC = self.in_channels, self.out_channels

        KH, KW = pair(self.kernel_size)

        scale = np.sqrt(1 / (C * KH * KW))

        W_data = np.random.randn(OC, C, KH, KW).astype(self.dtype) * scale

        self.W.data = W_data



    def forward(self, x):

        if self.W.data is None:

            self.in_channels = x.shape[1]

            self._init_W()

        

        y = conv2d(x, self.W, self.b, self.stride, self.pad)

        return y



class MLP(Model):

    def __init__(self, fc_output_size, activation=sigmoid):

    # 出力サイズをリストで指定することで多層の全結合レイヤを一つにまとめる

        super().__init__()

        self.activation = activation

        self.layers = []



        for i, out_size in enumerate(fc_output_size):

            layer = Linear(out_size)

            setattr(self, "l" + str(i), layer)

            self.layers.append(layer)

        # レイヤとレイヤのvalueをMLPに追加し最後にそれらをself.layersに追加

    def forward(self, x):

        for l in self.layers[:-1]:

            x = self.activation(l(x))

        # 最後のレイヤ以外をactivationして出力

        return self.layers[-1](x)

# 全結合レイヤを任意の数つなげるモデルのクラス



class VGG16(Model):

    WEIGHT_PATH = "https://github.com/koki0702/dezero-models/releases/download/v0.1/vgg16.npz"

    def __init__(self, pretrained=True):

        super().__init__()

        self.conv1_1 = Conv2d(64, kernel_size=3, stride=1, pad=1)

        self.conv1_2 = Conv2d(64, kernel_size=3, stride=1, pad=1)

        self.conv2_1 = Conv2d(128, kernel_size=3, stride=1, pad=1)

        self.conv2_2 = Conv2d(128, kernel_size=3, stride=1, pad=1)

        self.conv3_1 = Conv2d(256, kernel_size=3, stride=1, pad=1)

        self.conv3_2 = Conv2d(256, kernel_size=3, stride=1, pad=1)

        self.conv3_3 = Conv2d(256, kernel_size=3, stride=1, pad=1)

        self.conv4_1 = Conv2d(512, kernel_size=3, stride=1, pad=1)

        self.conv4_2 = Conv2d(512, kernel_size=3, stride=1, pad=1)

        self.conv4_3 = Conv2d(512, kernel_size=3, stride=1, pad=1)

        self.conv5_1 = Conv2d(512, kernel_size=3, stride=1, pad=1)

        self.conv5_2 = Conv2d(512, kernel_size=3, stride=1, pad=1)

        self.conv5_3 = Conv2d(512, kernel_size=3, stride=1, pad=1)

        self.fc6 = Linear(4096)

        self.fc7 = Linear(4096)

        self.fc8 = Linear(1000)

        if pretrained:

            weight_path = dezero.utils.get_file(VGG16.WEIGHT_PATH)

            self.load_weights(weight_path)

    

    def forward(self, x):

        x = relu(self.conv1_1(x))

        x = relu(self.conv1_2(x))

        x = pooling(x, 2, 2)

        x = relu(self.conv2_1(x))

        x = relu(self.conv2_2(x))

        x = pooling(x, 2, 2)

        x = relu(self.conv3_1(x))

        x = relu(self.conv3_2(x))

        x = relu(self.conv3_3(x))

        x = pooling(x, 2, 2)

        x = relu(self.conv4_1(x))

        x = relu(self.conv4_2(x))

        x = relu(self.conv4_3(x))

        x = pooling(x, 2, 2)

        x = relu(self.conv5_1(x))

        x = relu(self.conv5_2(x))

        x = relu(self.conv5_3(x))

        x = pooling(x, 2, 2)

        x = reshape(x, (x.shape[0], -1))

        # 4階テンソルを2階テンソルに変換

        x = dropout(relu(self.fc6(x)))

        x = dropout(relu(self.fc7(x)))

        x = self.fc8(x)

        return x



    @staticmethod

    def preprocess(image, size=(224, 224), dtype=np.float32):

        image = image.convert('RGB')

        if size:

            image = image.resize(size)

        image = np.asarray(image, dtype=dtype)

        image = image[:, :, ::-1]

        image -= np.array([103.939, 116.779, 123.68], dtype=dtype)

        # 固定値を差し引く

        image = image.transpose((2, 0, 1))

        # BGRの順にする

        return image

    # preprocessによって画像を幅高さ224に整形する　imagenetの学習をした時と同じ前処理らしい ※中身は分かってない

# VGG16



class Rnn(Layer):

    def __init__(self, hidden_size, in_size=None):

        super().__init__()

        self.x2h = Linear(hidden_size, in_size=in_size)

        self.h2h = Linear(hidden_size, in_size=in_size, nobias=True)

        self.h = None

    

    def reset_state(self):

        self.h = None

    

    def forward(self, x):

        if self.h is None:

            h_new = tanh(self.x2h(x))

        else:

            h_new = tanh(self.x2h(x) + self.h2h(self.h))

        self.h = h_new



        return h_new



class Simplernn(Model):

    def __init__(self, hidden_size, out_size):

        super().__init__()

        self.rnn = Rnn(hidden_size)

        self.fc = Linear(out_size)



    def reset_state(self):

        self.rnn.reset_state()



    def forward(self, x):

        h = self.rnn(x)

        y = self.fc(h)

        return y



class Lstm(Layer):

    def __init__(self, hidden_size, in_size=None):

        super().__init__()

        H, I = hidden_size, in_size

        self.x2f = Linear(H, in_size=I)

        self.x2i = Linear(H, in_size=I)

        self.x2o = Linear(H, in_size=I)

        self.x2u = Linear(H, in_size=I)

        self.h2f = Linear(H, in_size=H, nobias=True)

        self.h2i = Linear(H, in_size=H, nobias=True)

        self.h2o = Linear(H, in_size=H, nobias=True)

        self.h2u = Linear(H, in_size=H, nobias=True)

        self.reset_state()



    def reset_state(self):

        self.h = None

        self.c = None

    

    def forward(self, x):

        if self.h is None:

            f = sigmoid(self.x2f(x))

            i = sigmoid(self.x2i(x))

            o = sigmoid(self.x2o(x))

            u = tanh(self.x2u(x))

        else:

            f = sigmoid(self.x2f(x) + self.h2f(self.h))

            i = sigmoid(self.x2i(x) + self.h2i(self.h))

            o = sigmoid(self.x2o(x) + self.h2o(self.h))

            u = tanh(self.x2u(x) + self.h2u(self.h))



        if self.c is None:

            c_new = (i * u)

        else:

            c_new = (f * self.c) + (i + u)



        h_new = o * tanh(c_new)

        self.h, self.c = h_new, c_new

        return h_new



class Betterrnn(Model):

    def __init__(self, hidden_size, out_size):

        super().__init__()

        self.rnn = Lstm(hidden_size)

        self.fc = Linear(out_size)



    def reset_state(self):

        self.rnn.reset_state()



    def forward(self, x):

        y = self.rnn(x)

        y = self.fc(y)

        return y





class Embeddinglayer(Layer):

    def __init__(self, vocab_size, hidden_size, dtype=np.float32):

        super().__init__()

        self.W = Parameter(np.random.randn(vocab_size, hidden_size).astype(dtype) / np.sqrt(hidden_size), name="W")

    def __call__(self, x):

        y = self.W[x]

        return y





class Negativesamplingloss(Layer):

    def __init__(self, hidden_size, corpuses, vocab_size, power=0.75, sample_size=5, dtype=np.float32):

        super().__init__()

        self.sampler = Word_sampler(corpuses, power, sample_size)

        self.sample_size = sample_size

        self.embed = Embeddinglayer(vocab_size, hidden_size, dtype)

    def forward(self, x, t):

        negative_sample = self.sampler.get_negative_sample(t)

        correct = self.embed(t)

        y0 = sum(correct * x, axis=1)

        y = sigmoid_cross_entropy(y0, 1.0)

        

        for i in range(self.sample_size):

            bad = self.embed(negative_sample[i])

            y0 = sum(bad * x, axis=1)

            y += sigmoid_cross_entropy(y0, 0.0)

        y /= (self.sample_size + 1)

    

        return y







class Word2vec(Model):

    def __init__(self, corpuses, id_to_word, hidden_size, power=0.75, sample_size=5, dtype=np.float32):

        super().__init__()

        self.embed = Embeddinglayer(len(id_to_word), hidden_size, dtype)

        self.negativesampling = Negativesamplingloss(hidden_size, corpuses, len(id_to_word), power, sample_size, dtype)



    def forward(self, x, t):

        x0, x1 = self.embed(x[:,0]), self.embed(x[:,1])

        x = x0 + x1

        x *= 0.5

        y = self.negativesampling(x, t)

        return y







    

























        







#----------------------------------------------------------------------------------------

# dataset

#----------------------------------------------------------------------------------------

class Dataset:

    def __init__(self, train=True, transform=None, target_transform=None):

        self.train = train

        self.data = None

        self.label = None

        self.transform = transform

        self.target_transform = target_transform

        if transform is None:

            self.transform = lambda x: x

        if target_transform is None:

            self.target_transform = lambda x: x



        self.prepare()



    def __getitem__(self, index):

        assert np.isscalar(index)

        #indexは整数のみ対応 スライスには対応せず

        if self.label is None:

            return self.transform(self.data[index]), None

        #教師なし学習

        else:

            return self.transform(self.data[index]), self.target_transform(self.label[index])

        #教師あり学習

    

    def __len__(self):

        return len(self.data)

    

    def prepare(self):

        pass



# 最初にデータを全て読み込むと膨大なメモリを消費する場合があるのでDatasetクラス内でデータの管理を行うように設定する



        

            



        









#----------------------------------------------------------------------------------------

# Dataloader

#----------------------------------------------------------------------------------------

# Dataloaderを用いることで、自動的にシャッフルを行いイテレータを用いることでバッチ数の数だけデータを取り出してくれる



class Dataloader:

    def __init__(self, dataset, batch_size, shuffle=True):

        self.dataset = dataset

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.data_size = len(dataset)

        self.max_iter = math.ceil(self.data_size / self.batch_size)



        self.reset()

    # max_iterはデータを一周するのに何回推論を行う必要があるか

    # shuffleはテスト時にはFalseに設定する方が良い

    

    def reset(self):

        self.iteration = 0

        if self.shuffle:

            self.index = np.random.permutation(self.data_size)

        else:

            self.index = np.arange(self.data_size)

    # resetによってデータをシャッフル

    

    def __iter__(self):

        return self

    # イテレータが呼ばれたときに自分自身を返す

    

    def __next__(self):

        if self.iteration >= self.max_iter:

            self.reset()

            raise StopIteration

        

        i, batch_size = self.iteration, self.batch_size

        batch_index = self.index[i * batch_size:(i + 1) * batch_size]

        batch = [self.dataset[int(j)] for j in batch_index]

        x = np.array([example[0] for example in batch])

        t = np.array([example[1] for example in batch])

        self.iteration += 1

        return x, t

    # iterが最大でない場合にbatchの数だけx,tを用意し返す



    def next(self):

        return self.__next__()

    

    # 実際 for x,t in {Dataloader}のように使うとmax_iterの数だけデータを取り出すことができる







class Seqdataloader(Dataloader):

    def __init__(self, dataset, batch_size, ):

        super().__init__(dataset, batch_size, shuffle=None)

    

    def __next__(self):

        if self.iteration >= self.max_iter:

            self.reset()

            raise StopIteration



        jump = self.data_size // self.batch_size

        batch_index = [(i * jump + self.iteration) % self.data_size for i in range(self.batch_size)]

        batch = [self.dataset[i] for i in batch_index]

        x = np.array([example[0] for example in batch])

        t = np.array([example[1] for example in batch])

        self.iteration += 1

        return x, t















#----------------------------------------------------------------------------------------

# optimizer

#----------------------------------------------------------------------------------------

class Optimizer:

# optimizerの基底クラス

    def __init__(self):

        self.target = None

        self.hooks = []



    def setup(self, target):

        self.target = target

        return self

    

    def update(self):

        params = [p for p in self.target.params() if p.grad is not None]

        # targetのparamsを取り出す

        for f in self.hooks:

            f(params)

        for param in params:

            self.update_one(param)

        



    def update_one(self, param):

        raise NotImplementedError()



    def add_hook(self, f):

        self.hook.append(f)

    



class SGD(Optimizer):

    def __init__(self, lr=0.01):

        super().__init__()

        self.lr = lr

    

    def update_one(self, param):

        param.data -= self.lr * param.grad.data

# SGDクラス W = W - lr * grad



class MomentumSGD(Optimizer):

    def __init__(self, lr=0.01, momentum=0.9):

        super().__init__()

        self.lr = lr

        self.momentum = momentum

        self.vs = {}



    def update_one(self, param):

        v_key = id(param)

        if v_key not in self.vs:

            self.vs[v_key] = np.zeros_like(param.data)

        v = self.vs[v_key]

        v *= self.momentum

        v -= self.lr * self.grad.data

        param.data += v

# Momentumクラス

# v ← αv - lr * grad

# W ← W + v



class Adam(Optimizer):

    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):

        super().__init__()

        self.t = 0

        self.alpha = alpha

        self.beta1 = beta1

        self.beta2 = beta2

        self.eps = eps

        self.ms = {}

        self.vs = {}



    def update(self, *args, **kwargs):

        self.t += 1

        super().update(*args, **kwargs)



    @property

    def lr(self):

        fix1 = 1. - math.pow(self.beta1, self.t)

        fix2 = 1. - math.pow(self.beta2, self.t)

        return self.alpha * math.sqrt(fix2) / fix1



    def update_one(self, param):

        key = id(param)

        if key not in self.ms:

            self.ms[key] = np.zeros_like(param.data)

            self.vs[key] = np.zeros_like(param.data)



        m, v = self.ms[key], self.vs[key]

        beta1, beta2, eps = self.beta1, self.beta2, self.eps

        grad = param.grad.data

        m += (1 - beta1) * (grad - m)

        v += (1 - beta2) * (grad * grad - v)

        param.data -= self.lr * m / (np.sqrt(v) + eps)

#Adam　※中身はよくわかっていない



    









#----------------------------------------------------------------------------------------

# Variable overloads

#----------------------------------------------------------------------------------------

Variable.__mul__ = mul

Variable.__add__ = add

Variable.__rmul__ = mul

Variable.__radd__ = add

Variable.__neg__ = neg

Variable.__sub__ = sub

Variable.__rsub__ = rsub

Variable.__truediv__ = div

Variable.__rtruediv__ = rdiv

Variable.__pow__ = pow

Variable.__getitem__ = get_item







#----------------------------------------------------------------------------------------

# utils

#----------------------------------------------------------------------------------------



def numerical_diff(f, x, eps=1e-4):

# 入力はf=Function関数、x=Variable変数 eps=h

    x0 = Variable(x.data - eps)

    x1 = Variable(x.data + eps)

    y0 = f(x0)

    y1 = f(x1)

    return (y1.data - y0.data)/(2 * eps)

# 数値微分



def as_array(x):

    if np.isscalar(x):

        return np.array(x)

    return x

#整数や浮動小数点型をndarrayインスタンスに変更する関数



def as_variable(obj):

    if isinstance(obj, Variable):

        return obj

    return Variable(obj)

#ndarrayインスタンスをVariableに変換する



def _dot_var(v, verbose=False):

    dot_var = '{} [label="{}", color=orange, style=filled]\n'

    name = " " if v.name is None else v.name

    if verbose and v.data is not None:

        if v.name is not None:

            name += ": "

        name += str(v.shape) + " " + str(v.dtype)

    return dot_var.format(id(v), name)

# var関数のグラフを作る関数　verbose=Trueの時 varの形とdtypeを表示する



def _dot_func(f):

    dot_func = '{} [label = "{}", color=lightblue, style=filled, shape=box]\n'

    txt = dot_func.format(id(f), f.__class__.__name__)

    # functionのidと名前を渡す

    dot_edge = '{} -> {}\n'

    for x in f.inputs:

        txt += dot_edge.format(id(x), id(f))

    for y in f.outputs:

        txt += dot_edge.format(id(f), id(y()))

    return txt

# functionのグラフを作り、varとの関係性を構築する



def get_dot_graph(output, verbose=True):

    txt = ""

    funcs =[]

    seen_set = set()

    def add_func(f):            

        if f not in seen_set:

            funcs.append(f)

            seen_set.add(f)

    add_func(output.creator)

    txt += _dot_var(output, verbose)

    while funcs:

        func = funcs.pop()

        txt += _dot_func(func)

        for x in func.inputs:

            txt += _dot_var(x, verbose)

            if x.creator is not None:

                add_func(x.creator)

    return "digraph g {\n" + txt + "}"

# 各関数変数のdot言語をまとめる



def plot_dot_graph(output, verbose=True, to_file="graph.png"):

    dot_graph = get_dot_graph(output, verbose)

    tmp_dir = os.path.join(os.path.expanduser("~"), ".dezero")

    if not os.path.exists(tmp_dir):

        os.mkdir(tmp_dir)

    graph_path = os.path.join(tmp_dir, "tmp_graph.dot")



    with open(graph_path, "w") as f:

        f.write(dot_graph)

    extension = os.path.splitext(to_file)[1][1:]

    cmd = f"dot {graph_path} -T {extension} -o {to_file}"

    subprocess.run(cmd, shell=True)

# get_dot_graphからdot言語を読み込みto_file上に計算グラフを保存する





def reshape_sum_backward(gy, x_shape, axis, keepdims):

    ndim = len(x_shape)

    tupled_axis = axis

    if axis is None:

        tupled_axis = None

    elif not isinstance(axis, tuple):

        tupled_axis = (axis,)



    if not (ndim == 0 or tupled_axis is None or keepdims):

        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]

        shape = list(gy.shape)

        for a in sorted(actual_axis):

            shape.insert(a, 1)

    else:

        shape = gy.shape



    gy = gy.reshape(shape)  # reshape

    return gy

# axisとkeepdimsによって得られたsumに対して元のデータに戻す際整形する関数　※中身はよくわからない



def def_sum_to(x, shape):

    ndim = len(shape)

    lead = x.ndim - ndim

    lead_axis = tuple(range(lead))



    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])

    y = x.sum(lead_axis + axis, keepdims=True)

    if lead > 0:

        y = y.squeeze(lead_axis)

    return y

# shapeに合うようにxを足し合わせる関数 ※中身はよくわからない



def logsumexp(x, axis=1):

    m = x.max(axis=axis, keepdims=True)

    y = x - m

    np.exp(y, out=y)

    # exp(x-C)

    s = y.sum(axis=axis, keepdims=True)

    # Σexp(x-C)

    np.log(s, out=s)

    # log Σexp(x-C)

    # outを使うとメモリ効率が良い

    m += s

    # C + log Σexp(x-C)

    return m

# 要検討





def accuracy(y, t):

    y, t = as_variable(y), as_variable(t)

    # y,tをVar変数として定義

    pred = y.data.argmax(axis=1).reshape(t.shape)

    # yの予測結果の最大のインデックスを求め.tの次元に変更する

    result = (t.data == pred)

    # True or False * データ長 の配列になる

    acc = result.mean()

    # Trueの割合を求めることができる

    return Variable(as_array(acc))

# 正解率を表示する yとtを引数に持つ



def as_numpy(x):

    if isinstance(x, Variable):

        x = x.data

    if np.isscalar(x):

        return np.array(x)

    elif isinstance(x, np.ndarray):

        return x

    return cupy.asnumpy(x)

# （主にcupyのndarrayを）numpyのndarrayに変換する関数



def dropout(x, dropout_ratio=0.5):

    x = as_variable(x)

    if Config.train:

        mask = np.random.rand(*x.shape) > dropout_ratio

        scale = np.array(1.0 * dropout_ratio).astype(x.dtype)

        y = x * mask / scale

        return y

    else:

        return x

# dropout関数 dropoutを用いると入力に対し一定の確率で値を0にする scaleで割り出力を弱めることでテスト時と学習時で同じスケールで扱えるようにする



def get_conv_outsize(input_size, kernel_size, stride, pad):

    return (input_size + pad * 2 - kernel_size) // stride + 1

# 入力とpadding、フィルタのサイズ、strideから出力のサイズを計算する関数



def get_deconv_outsize(size, kernel_size, stride, pad):

    return stride * (size - 1) + kernel_size - 2 * pad

# ※要検討



def pair(x):

    if isinstance(x, int):

        return (x, x)

    elif isinstance(x, tuple):

        assert len(x) == 2

        return x

    else:

        raise ValueError

# 値が一つの場合は二つにして二つの場合はそのまま返す関数



def im2col_array(img, kernel_size, stride, pad, to_matrix=True):



    N, C, H, W = img.shape

    KH, KW = pair(kernel_size)

    SH, SW = pair(stride)

    PH, PW = pair(pad)

    OH = get_conv_outsize(H, KH, SH, PH)

    OW = get_conv_outsize(W, KW, SW, PW)



    if gpu_enable:

        col = _im2col_gpu(img, kernel_size, stride, pad)

    else:

        img = np.pad(img,

                     ((0, 0), (0, 0), (PH, PH + SH - 1), (PW, PW + SW - 1)),

                     mode='constant', constant_values=(0,))

        col = np.ndarray((N, C, KH, KW, OH, OW), dtype=img.dtype)



        for j in range(KH):

            j_lim = j + SH * OH

            for i in range(KW):

                i_lim = i + SW * OW

                col[:, :, j, i, :, :] = img[:, :, j:j_lim:SH, i:i_lim:SW]



    if to_matrix:

        col = col.transpose((0, 4, 5, 1, 2, 3)).reshape((N * OH * OW, -1))



    return col

# 入力データをフィルタに対して計算しやすい行列に変換する関数　※中身がわからない.....



def col2im_array(col, img_shape, kernel_size, stride, pad, to_matrix=True):

    N, C, H, W = img_shape

    KH, KW = pair(kernel_size)

    SH, SW = pair(stride)

    PH, PW = pair(pad)

    OH = get_conv_outsize(H, KH, SH, PH)

    OW = get_conv_outsize(W, KW, SW, PW)



    if to_matrix:

        col = col.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)



    

    if gpu_enable:

        img = _col2im_gpu(col, SH, SW, PH, PW, H, W)

        return img

    else:

        img = np.zeros((N, C, H + 2 * PH + SH - 1, W + 2 * PW + SW - 1),

                       dtype=col.dtype)

        for j in range(KH):

            j_lim = j + SH * OH

            for i in range(KW):

                i_lim = i + SW * OW

                img[:, :, j:j_lim:SH, i:i_lim:SW] += col[:, :, j, i, :, :]

        return img[:, :, PH:H + PH, PW:W + PW]

# im2colの逆伝播 ※中身がわからない



def _im2col_gpu(img, kernel_size, stride, pad):

    """im2col function for GPU.

    This code is ported from Chainer:

    https://github.com/chainer/chainer/blob/v6.4.0/chainer/utils/conv.py

    """

    n, c, h, w = img.shape

    kh, kw = pair(kernel_size)

    sy, sx = pair(stride)

    ph, pw = pair(pad)

    out_h = get_conv_outsize(h, kh, sy, ph)

    out_w = get_conv_outsize(w, kw, sx, pw)

    dy, dx = 1, 1

    col = np.empty((n, c, kh, kw, out_h, out_w), dtype=img.dtype)



    np.ElementwiseKernel(

        'raw T img, int32 h, int32 w, int32 out_h, int32 out_w,'

        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'

        'int32 dy, int32 dx',

        'T col',

        '''

           int c0 = i / (kh * kw * out_h * out_w);

           int ky = i / (kw * out_h * out_w) % kh;

           int kx = i / (out_h * out_w) % kw;

           int out_y = i / out_w % out_h;

           int out_x = i % out_w;

           int in_y = ky * dy + out_y * sy - ph;

           int in_x = kx * dx + out_x * sx - pw;

           if (in_y >= 0 && in_y < h && in_x >= 0 && in_x < w) {

             col = img[in_x + w * (in_y + h * c0)];

           } else {

             col = 0;

           }

        ''',

        'im2col')(img.reduced_view(),

                  h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dy, dx, col)



    return col



def _col2im_gpu(col, sy, sx, ph, pw, h, w):

    """col2im function for GPU.

    This code is ported from Chainer:

    https://github.com/chainer/chainer/blob/v6.4.0/chainer/utils/conv.py

    """

    n, c, kh, kw, out_h, out_w = col.shape

    dx, dy = 1, 1

    img = np.empty((n, c, h, w), dtype=col.dtype)



    np.ElementwiseKernel(

        'raw T col, int32 h, int32 w, int32 out_h, int32 out_w,'

        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'

        'int32 dx, int32 dy',

        'T img',

        '''

           int c0 = i / (h * w);

           int y  = i / w % h;

           int x  = i % w;

           T val = 0;

           for (int ky = 0; ky < kh; ++ky) {

             int out_y = (y + ph - ky * dy);

             if (0 > out_y || out_y >= out_h * sy) continue;

             if (out_y % sy != 0) continue;

             out_y /= sy;

             for (int kx = 0; kx < kw; ++kx) {

               int out_x = (x + pw - kx * dx);

               if (0 > out_x || out_x >= out_w * sx) continue;

               if (out_x % sx != 0) continue;

               out_x /= sx;

               int k = out_y + out_h * (kx + kw * (ky + kh * c0));

               val = val + col[out_x + out_w * k];

             }

           }

           img = val;

        ''',

        'col2im')(col.reduced_view(),

                  h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dx, dy, img)

    return img

# ※要検討



def create_corpuses(dataset):

    wakati = MeCab.Tagger("-Owakati")

    data = []

    for i in range(len(dataset)):

        tmp = wakati.parse(dataset[i])

        tmp.replace("\n", "")

        tmp = tmp.split()

        tmp.insert(0, "<eos>")

        tmp.append("<end>")

        data.append(tmp)

    word_to_id = {}

    id_to_word = {}

    corpuses = []

    for context in data:

        for word in context:

            if word not in word_to_id:

                new_id = len(word_to_id)

                word_to_id[word] = new_id

                id_to_word[new_id] = word

        corpuses.append([word_to_id[word] for word in context])

    return corpuses,id_to_word

# 文章のリストを入力するとcorpusにしてくれる関数



def create_simpleword2vec_data(corpuses):

    x = []

    for corpus in corpuses:

        for i in range(1,len(corpus)-2):

            x.append([[corpus[i-1],corpus[i+1]],corpus[i]])

    return x

# window_sizeが1の場合のword2vecの学習データを生成する関数





class Word_sampler:

    def __init__(self, corpuses, power, sample_size):

        self.sample_size = sample_size

        self.vocab_size = None

        self.word_p = None

        counts = collections.Counter()

        for corpus in corpuses:

            for word_id in corpus:

                counts[word_id] += 1

        vocab_size = len(counts)

        self.vocab_size = vocab_size



        self.word_p = np.zeros(vocab_size)

        for i in range(vocab_size):

            self.word_p[i] = counts[i]



        self.word_p = np.power(self.word_p, power)

        self.word_p /= np.sum(self.word_p)



    def get_negative_sample(self, target):

        batch_size = target.shape[0]





        negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)

        for i in range(batch_size):

                p = self.word_p.copy()

                target_idx = target[i]

                p[target_idx] = 0

                p /= p.sum()

                negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size,  p=p)

        negative_sample = negative_sample.T

        return negative_sample

#negativesampleを作るクラス　引数にcorpusとpowerを持つ







#----------------------------------------------------------------------------------------

# 実装

#----------------------------------------------------------------------------------------

batch_size= 1000

max_epoch = 20

hidden_size = 512



dataset = pd.read_excel("JEC_basic_sentence_v1-3.xls",header=None)

corpuses,id_to_word = create_corpuses(dataset[1])

datasets = create_simpleword2vec_data(corpuses)

train_data = Dataloader(datasets,batch_size=batch_size)

model = Word2vec(corpuses,id_to_word,hidden_size)

optimizer = Adam().setup(model)

for epoch in range(max_epoch):

    avg_loss = 0

    for x, t in train_data:

        loss = model.forward(x, t)

        model.cleargrads()

        loss.backward()

        optimizer.update()

        avg_loss += loss




    avg_loss = avg_loss / len(t)

    print(f"| epoch {epoch+1} | loss {avg_loss.data}")

model.save_weights("word2vec.npz")





