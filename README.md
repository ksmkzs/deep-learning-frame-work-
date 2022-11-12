# deep-learning-frame-work
TensorflowやPytorchなどの外部フレームワークを用いずにdeeplearningのフレームワークを作成しようとしている　2021年11月12日時点で作成途中

オライリー社の出版している『ゼロから作るDeepLearning③ フレームワーク編』を読み、多少の変更を加えながらフレームワークを作成している。

現在は全結合のモデルを再現可能であり、実行することで本書で用意されている「MNISTデータセット(http://yann.lecun.com/exdb/mnist/)」の画像分類をするディープラーニングモデルの学習が実行され、my_mlp.npz上にハイパーパラメータが保存される。

max_epoch = データを何周学習させるか
batch_size = 一回の処理に何個のデータを使用する羽化
hidden_size = 隠れ層のデータ数
accuracy = データの精度(100%が最大)

また、ReLU関数とSigmoid関数を自由に設定できるほか、Adam、AdaGrad、Momentiumなどの最適化関数も利用でき、dropoutも利用可能。


今後はCNNやRNNを実装する手筈を整えることで、より汎用性の高いフレームワークを実装予定。


