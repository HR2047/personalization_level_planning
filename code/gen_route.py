import numpy as np
import pandas as pd
import numpy.random as random
import scipy as sp
from pandas import Series,DataFrame
from gurobipy import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import gurobipy as gu
from sampling_ratio import sampling_ratio

def gen_route(start_point, end_point, timebudget):
    df = pd.read_csv("../data/common_data.csv")
    user_score = pd.read_csv(f"../data/userdata/user_score.csv", header=None)
    df['personalize score'] = user_score.iloc[0]
    
    personalize_level = sampling_ratio()
        
    pop_ratio = personalize_level[0][0]
    pref_ratio = personalize_level[1][0]
    poppref_ratio = personalize_level[2][0]

    # StandardScalerを使用したZスコア正規化
    scaler = StandardScaler()

    df['personalize score'] = scaler.fit_transform(df[['personalize score']])
    df['personalize score'] += 1

    # 両方考慮した評価値
    df['mix'] = df[['score', 'personalize score']].mean(axis=1)
        
    duration_matrix = pd.read_csv("../data/duration_modified.csv")

    # 最適化モデルの定義
    model = gu.Model("p_merihari")

    #エリア数
    N = df.shape[0]-1
    s = start_point
    g = end_point

    X={}
    Y={}
    # 0:非個人化 1:個人化 2:両方
    I = [0,1,2] 
    J = [0,1,2]

    # 変数を定義
    for x in range(N+1):
        for i in range(len(I)):
            X[x,i] = model.addVar(name=f"(x[{x},{i}]",vtype="B",lb=0,ub=1) 

    for y in range(N+1):
        for j in range(len(J)):
            Y[y,j] = model.addVar(name=f"(y[{y},{j}]",vtype="B",lb=0,ub=1)
    model.update()

    c = {} #コースの評価値
    basis_list = ['score', 'personalize score', 'mix']
    for x in range(N+1):
        for i, basis in enumerate(basis_list):
            c[x,i] = df[(df["pid"]==x)][basis]

    A = {} #移動を示すバイナリ値
    for x in range(N+1):
        for i in I:
            for y in range(N+1):
                for j in J:
                    if x != y:
                        A[x,i,y,j] = model.addVar(name = f"A({X[x,i],Y[y,j]})", vtype="B")
    model.update()

    #最適化式
    model.setObjective(gu.quicksum(A[x,i,y,j]*c[x,i] for x in range(N+1) for i in I for y in range(N+1) for j in J if x!=y and x!=g and y!=s),gu.GRB.MAXIMIZE)
    model.update()

    #　条件式(8)の登録：スタートポイントとエンドポイント
    model.addConstr(gu.quicksum(A[s,i,y,j] for i in I for y in range(N+1)for j in J if y!=s) == 1) # from s = 1
    model.addConstr(gu.quicksum(A[g,i,y,j] for y in range(N+1)  for i in I for j in J if y!=g) == 0) # from g = 0
    model.addConstr(gu.quicksum(A[x,i,g,j] for x in range(N+1)  for i in I for j in J if x!=g) == 1) # to g = 1
    model.addConstr(gu.quicksum(A[x,i,s,j] for i in I for x in range(N+1)for j in J if x!=s) == 0) # to s = 0
    model.update()

    # 条件式(3)の登録；経路の接続を保証、2度同じエリアに行かない
    #w:経路途中のエリア
    w = [model.addVar(name = "%s"%(i), vtype="I", lb=1,ub=N) for i in range(N+1) if i!=s and i!=g] 
    model.update()

    for w in range(N+1):
        if w == s or w == g:
            continue
        for k in I:
            model.addConstr(gu.quicksum(A[x,i,w,k] for x in range(N+1) for i in I if x!=w and x!=g )  == gu.quicksum(A[w,k,y,j] for y in range(1,N+1) for j in J if y!=w ) )
            model.addConstr(gu.quicksum(A[x,i,w,k] for x in range(N+1)for i in I if x!=w and x!=g )  <= 1)
            model.addConstr(gu.quicksum(A[w,k,y,j] for y in range(N+1) for j in J if y!=w and y!=s )  <= 1)
    model.update()


    # 条件式(4) (MTZ制約：sub-tourの排除)
    u={} #補助変数
    for x in range(N+1):
        if x == s:
            continue
        u[x] = model.addVar(name="u[{x}]", lb=1, ub=(N+1),vtype="I")
    model.update()
            
    for x in range(N+1):
        if x == s:
            continue
        for i in I:
            for y in range(N+1):
                if y == s:
                    continue
                for j in J:
                    if x != y:
                        model.addConstr((u[x] + 1 - (N)*(1 - A[x,i,y,j]))<= u[y])
    model.update()

    #条件式(5):時間予算を超えない
    time = {} #コースの所要時間
    stay_time={}
    for y in range(N+1):
            stay_time[y] = float(df[df["pid"]==y]["stay_time"].iloc[0])

    for x in range(N+1):
            for y in range(N+1):
                    if x!=y:
                        time[x,y] = duration_matrix.iloc[x,y] + stay_time[x]

    model.addConstr(gu.quicksum((time[x,y]* A[x,i,y,j])for x in range(N+1)for i in I  for y in range(N+1) for j in J if x!=y and x!=g and y!=s ) <= timebudget)
    model.update()

    #条件式(6):各エリアで１コース以内が選ばれる
    for x in range(N+1):
            model.addConstr(gu.quicksum(A[x,i,y,j] for i in I for y in range(N+1) for j in J if x!=y) <=1)
    model.update()

    for y in range(N+1):
            model.addConstr(gu.quicksum(A[x,i,y,j] for x in range(N+1)for i in I  for j in J if x!=y) <=1)
    model.update()

    #追加の条件式 : 個人化レベルの割合
    num_course = gu.quicksum(A[x, i, y, j] for x in range(N+1) for i in I for y in range(N+1) for j in J if x != y)
    num_0 = gu.quicksum(A[x, 0, y, j] for x in range(N+1) for y in range(N+1) for j in J if x != y and x!=g and y!=s)
    num_1 = gu.quicksum(A[x, 1, y, j] for x in range(N+1) for y in range(N+1) for j in J if x != y and x!=g and y!=s)
    num_2 = gu.quicksum(A[x, 2, y, j] for x in range(N+1) for y in range(N+1) for j in J if x != y and x!=g and y!=s)

    epsilon = 1.0  # 許容誤差
    model.addConstr(num_0 >= pop_ratio * num_course - epsilon)
    #model.addConstr(num_0 <= pop_ratio * num_course + epsilon)
    model.addConstr(num_1 >= pref_ratio * num_course - epsilon)
    #model.addConstr(num_1 <= pref_ratio * num_course + epsilon)
    model.addConstr(num_2 >= poppref_ratio * num_course - epsilon)
    #model.addConstr(num_2 <= poppref_ratio * num_course + epsilon)

    model.update()

    model.setParam('OutputFlag', 0)
    model.optimize()

    try:
        score = model.ObjVal

    # no solution
    except AttributeError as e:
        print("no solution")
        # return 

    #調査用
    count =0
    count_pop =0
    count_pref =0
    count_poppref =0
    test_time = 0.00
    test_score=0
    for x in range(N+1):
        for i in I:
            for y in range(N+1):
                for j in J:
                    if x!=y:
                        if A[x,i,y,j].X==1:
                            count=count+1
                            if i==0:
                                count_pop=count_pop+1
                            elif i==1:
                                count_pref=count_pref+1
                            elif i==2:
                                count_poppref=count_poppref+1
                            test_time += time[x,y]
                            test_score += float(c[x,i])

    print("訪問エリア数",count)
    print("Timebudget",timebudget,"h")
    print("合計旅行時間",test_time)
    print("旅程のスコア",test_score)

    #調査用
    route = pd.DataFrame(columns=['x', 'y', 'i'])

    for x in range(N+1):
        for i in I:
            for y in range(N+1):
                for j in J:
                    if x!=y and A[x, i, y, j].X == 1:
                        route = route.append({'x': x, 'y': y, 'i':i}, ignore_index=True)
    x = s
    d = s

    route_ans = []
    level_ans = []
    route_ans.append(x)

    while d != g:
        rows = route.loc[route['x'] == x]
        if not rows.empty:
            d = rows['y'].values[0]
            e = rows['i'].values[0]
            route_ans.append(d)
            level_ans.append(e)
            x = d
        else:
            print("error")
            # return
        
    name = pd.read_csv("../data/name.csv")

    for i in route_ans:
        if i != g:
            poi_name = name.loc[name['pid'] == i, 'jpn name'].values[0]
            basis = route.loc[route['x'] == i, 'i'].values[0]
            print(f"{poi_name}({basis})", end='→')
        else:
            poi_name = name.loc[name['pid'] == i, 'jpn name'].values[0]
            print(f"{poi_name}")

    pop_ratio_route = count_pop / count
    pref_ratio_route = count_pref / count
    poppref_ratio_route = count_poppref / count

    print("個人化レベル(設定値)　　", pop_ratio, pref_ratio, poppref_ratio)
    print("個人化レベル(生成ルート)", pop_ratio_route, pref_ratio_route, poppref_ratio_route)
