# データセットの準備

本プロジェクトは [SUN RGB-D](https://rgbd.cs.princeton.edu/) を使用します。データセットの配置先は環境変数 `YOLOX_DATADIR` で指定できます。

```shell
export YOLOX_DATADIR=/path/to/your/datasets
```

`YOLOX_DATADIR` が未設定の場合は、カレントディレクトリからの相対パス `./datasets` が既定値になります。

## ディレクトリ構成

```
$YOLOX_DATADIR/
└── SUNRGBD/
    ├── train_anno_10.json   # 学習用アノテーション（全データの 1/10 サンプリング）
    ├── test_anno_10.json    # 評価用アノテーション
    └── ...                  # JSON の img_path が参照する RGB 画像ファイル群
```

使用するアノテーションファイルは実験設定（`yolox/exp/yolox3D_base.py` の `Exp.train_ann` / `Exp.val_ann`）で指定します。既定では検証サイクルを短縮するため 1/10 サンプリングした `*_anno_10.json` を参照します。

> **画像パスの解決について**
> JSON の `img_path` は **`$YOLOX_DATADIR` からの相対パス**として解決されます（`SUNRGBDDataset.load_image`）。したがって画像を `$YOLOX_DATADIR/SUNRGBD/` 以下に置く場合、`img_path` は `SUNRGBD/kv1/.../image/0000103.jpg` のように `SUNRGBD/` を含めた形で記述してください。
>
> なお `cache_type="disk"` を指定した際のキャッシュパスのみ `SUNRGBD/` を重ねて付与する実装になっており、RAMキャッシュ（既定）または非キャッシュでの実行を推奨します。

## アノテーション JSON のフォーマット

トップレベルは **画像IDをキーとする辞書**です。COCO 形式ではなく、本プロジェクト固有の形式である点に注意してください。

```json
{
  "<image_id>": {
    "img_path": "SUNRGBD/kv1/NYUdata/NYU0001/image/NYU0001.jpg",
    "width": 730,
    "height": 530,
    "intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "extrinsics": [[r11, r12, r13, t1],
                   [r21, r22, r23, t2],
                   [r31, r32, r33, t3]],
    "objects": [
      {
        "name": "chair",
        "polygon": [[[x0, y0, z0], [x1, y1, z1], "... 全8頂点 ..."]],
        "3Dbox": [cx, cy, cz, w, h, l, yaw]
      }
    ]
  }
}
```

| キー | 内容 |
| --- | --- |
| `img_path` | RGB画像への相対パス（`$YOLOX_DATADIR` 起点） |
| `width` / `height` | 元画像の幅・高さ。入力サイズへのリサイズ係数 `r = min(640/height, 640/width)` の算出に使用 |
| `intrinsics` | カメラ内部パラメータ（3×3）。評価時に3Dボックス中心を画像へ投影する際に使用 |
| `extrinsics` | カメラ外部パラメータ（3×4 の `[R\|t]`）。読み込み時に4×4化して逆行列を保持 |
| `objects[].name` | クラス名。下記21クラスに含まれない名前は自動的に `others` へ丸められます |
| `objects[].polygon` | `polygon[0]` に直方体の**8頂点**を `[x, y, z]` で格納。固定トポロジの6面体メッシュとして読み込まれます |
| `objects[].3Dbox` | GT の3Dボックス `[cx, cy, cz, w, h, l, yaw]`（7要素）。AP@0.25 / AP@0.5 の評価に使用 |

### 頂点座標の扱い

`polygon[0]` の8頂点は、以下の順に処理されます。

1. リサイズ係数 `r` を乗算
2. 立方体領域 `[-3.9, -3.9, 0.1] 〜 [3.9, 3.9, 7.9]` へクランプ
3. オフセット `[4, 4, 0]` を加算し、各軸が `[0.1, 7.9]` の範囲に収まるよう平行移動

その後、固定の面インデックス（6面 × 4頂点）を適用して直方体メッシュを構成します。この座標空間がモデル側の $80^3$ 格子へマッピングされます。

## クラス定義

`yolox/data/datasets/sunrgbd_classes.py` の `SUNRGBD_CLASSES_21` を使用します（`others` と `background` を含む21クラス）。

```
bathtub, bed, bookshelf, box, chair, counter, desk, door, dresser,
garbage bin, lamp, monitor, night stand, pillow, sink, sofa, table,
tv, toilet, others, background
```

38クラス版の `SUNRGBD_CLASSES_38` も定義済みですが、現行の学習・評価では21クラス版を使用しています。

---

## 補足: COCO データセット（フォーク元 YOLOX の2D検出用）

フォーク元 YOLOX の2D物体検出を動かす場合は、以下の構成で COCO を配置します。

```
$YOLOX_DATADIR/
└── COCO/
    ├── annotations/
    │   └── instances_{train,val}2017.json
    └── {train,val}2017/
        # 対応する JSON に記載された画像ファイル群
```

2014年版のデータセットも利用できます。詳細は [COCO detection](https://cocodataset.org/#download) を参照してください。
