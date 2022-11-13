import numpy as np
import weakref, contextlib, os, subprocess

#----------------------------------------------------------------------------------------
# Config
#----------------------------------------------------------------------------------------
class Config:
    enable_backcrop = True




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

    def __init_W(self):
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data
    # Wの値を設定、形状は[in_size, out_size]

    def forward(self, x):
        if self.W.data is None:
            self.in_size = x.shape[1]
            self.__init_W()
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
        if self.x0_shape != self.x1.shape:
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
    as_variable(x1)
    return Div()(x0, x1)
# x0 / x1

def rdiv(x0, x1):
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
        self.shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y
    
    def backward(self, gy):
        gy = reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy)
        return gx

def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)
# 行列の値を合計する関数　backwardでは元のデータの形状にgyがブロードキャストされる
# axisで指定した方向に沿って和を求められる keepdimsは次元数を保つ


class Broadcast_to(Function):
    def forward(self, x, shape):
        self.x_shape = x.shape
        y = np.broadcast(x. shape)
        return y
    
    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx

def broadcast_to(x, shape):
    return Broadcast_to()(x, shape)
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
            



    
def goldstein(x, y):
    z = (1+(x+y+1)**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2))*(30+(2*x-3*y)**2*(18-32*x+12*x**2+48*y-36*x*y+27*y**2))
    return z


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

#----------------------------------------------------------------------------------------
# 実装
#----------------------------------------------------------------------------------------

x = Variable(np.random.randn(5,10),name="x")
model = Twolayernet(100,10)
model.plot(x)

