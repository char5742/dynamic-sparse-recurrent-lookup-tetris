const PROJECT_ROOT = normpath(joinpath(@__DIR__,".."))
const col = 10  # 10 columns
const row = 20  # 20 rows
const LEFT_ROTATION::Int8 = 1
const FALL_THRESHOLD = 60 # 自由落下の閾値
const MAX_GROUND_ACTION_COUNT = 15 # 接地後に行える行動の上限回数
const MAX_PUT_DELAY_COUNT = 30 # 設置までの猶予フレーム数