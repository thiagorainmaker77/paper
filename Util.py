import pandas as pd



def build_df(name, ln):
    nome_cls = [ str(name ) +'_0',  str(name ) +'_1', str(name ) +'_2', str(name ) +'_3',  str(name ) +'_4', str(name ) +'_5' ]
    df = pd.DataFrame(columns=nome_cls)

    for i in ln:
        df = df.append({
            str(name ) +'_0': i[0],
            str(name ) +'_1': i[1],
            str(name ) +'_2': i[2],
            str(name ) +'_3': i[3],
            str(name ) +'_4': i[4],
            str(name ) +'_5': i[5],
        }, ignore_index=True)


    return df