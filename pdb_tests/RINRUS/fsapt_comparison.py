import pandas as pd

fsapt_data = """
Frag1     Frag2         Elst     Exch    IndAB    IndBA     Disp    EDisp    Total 
A:203     A:7:ARG   -140.728   18.406   -4.160   -7.047   -7.339    0.000 -140.867 
A:203     A:7:MC      -2.895    0.000   -0.012   -0.256   -0.039    0.000   -3.201 
A:203     A:8:GLY     -1.692    0.000   -0.026   -0.126   -0.017    0.000   -1.860 
A:203     A:8:MC       8.352    0.008    0.163   -0.441   -0.096    0.000    7.986 
A:203     A:9:ALA      2.752    0.060    0.037   -0.958   -0.315    0.000    1.575 
A:203     A:9:MC      -1.742   -0.000   -0.010   -0.232   -0.032    0.000   -2.016 
A:203     A:78:GLU    84.100    1.858   -3.723   -0.462   -1.667    0.000   80.106 
A:203     A:78:MC     -4.129    0.000    0.200   -0.094   -0.028    0.000   -4.051 
A:203     A:79:MET     1.155   -0.000   -0.018   -0.375   -0.103    0.000    0.659 
A:203     A:81:VAL     0.295    0.005    0.005   -0.374   -0.109    0.000   -0.178 
A:203     A:84:GLY     1.098   -0.000   -0.010   -0.081   -0.019    0.000    0.988 
A:203     A:84:MC     -4.962   -0.000    0.007   -0.153   -0.029    0.000   -5.136 
A:203     A:85:LEU     0.834    0.001    0.008   -0.488   -0.108    0.000    0.247 
A:203     A:88:CYS     4.248    0.001    0.034   -0.526   -0.103    0.000    3.654 
A:203     A:90:ARG  -107.239    9.337   -0.576   -5.205   -6.119    0.000 -109.802 
A:203     A:108:TYR  -12.724    8.221   -1.156   -7.690   -3.981    0.000  -17.331 
A:203     A:115:LEU   -1.113    9.873   -0.074   -3.926   -6.532    0.000   -1.771 
A:203     A:115:MC     7.930    0.075    0.018   -0.723   -0.579    0.000    6.722 
A:203     A:210:HOH  -18.557   10.379   -1.989   -1.874   -3.440    0.000  -15.481 
A:203     A:235:HOH    0.763    0.001    0.176   -0.088   -0.037    0.000    0.815 
A:203     A:249:HOH   -0.038    0.000    0.000   -0.118   -0.071    0.000   -0.227 
A:203     C:47:LEU     0.933    0.000   -0.009   -0.488   -0.096    0.000    0.340 
A:203     C:56:VAL    -0.089    0.000    0.062   -0.472   -0.079    0.000   -0.578 
A:203     C:56:MC     -6.891    0.005   -0.071   -0.418   -0.123    0.000   -7.499 
A:203     C:57:PHE     3.642    5.218   -0.041   -3.397   -7.123    0.000   -1.700 
A:203     C:57:MC     13.240    0.024    0.353   -0.837   -0.343    0.000   12.436 
A:203     C:58:PRO    -1.648    0.001   -0.013   -0.513   -0.120    0.000   -2.294 
A:203     C:58:MC     -5.547    0.015    0.212   -0.833   -0.345    0.000   -6.499 
A:203     C:59:ALA    -0.262    4.148   -0.114   -1.744   -3.908    0.000   -1.881 
A:203     C:59:MC     -0.739    2.758    0.055   -2.599   -2.022    0.000   -2.549 
A:203     C:60:LYS   -66.323    1.782   -0.379   -3.658   -1.744    0.000  -70.321 
A:203     C:60:MC     -7.264    0.007    0.064   -0.808   -0.185    0.000   -8.187 
A:203     C:61:MC     -6.201    0.000    0.021   -0.304   -0.038    0.000   -6.522 
A:203     C:62:VAL     0.426    0.000   -0.025   -0.625   -0.108    0.000   -0.332 
A:203     C:62:MC     -5.031    0.000   -0.026   -0.303   -0.036    0.000   -5.397 
A:203     C:63:ARG  -126.334    9.205   -2.518   -5.767   -4.783    0.000 -130.197 
A:203     C:71:MC      8.884   -0.000    0.037   -0.209   -0.032    0.000    8.681 
A:203     C:72:PRO     0.180   -0.000    0.031   -0.271   -0.040    0.000   -0.100 
A:203     C:72:MC     -7.890    0.001   -0.003   -0.356   -0.124    0.000   -8.371 
A:203     C:73:VAL     0.362    0.113   -0.001   -1.074   -0.855    0.000   -1.455 
A:203     C:73:MC      8.954    1.799    0.093   -0.596   -1.545    0.000    8.705 
A:203     C:74:THR    10.518    4.205   -0.451   -1.160   -2.823    0.000   10.289 
A:203     C:74:MC     -9.282    1.533    0.527   -0.859   -1.237    0.000   -9.318 
A:203     C:75:CYS    -1.931    2.441    0.068   -2.046   -2.435    0.000   -3.902 
A:203     C:75:MC      0.850    0.001   -0.199   -0.222   -0.133    0.000    0.296 
A:203     C:232:HOH    0.779    0.001    0.017   -0.259   -0.119    0.000    0.419 
A:203     C:236:HOH    1.763    0.000    0.079   -0.104   -0.031    0.000    1.706 
A:203     cap_ats     -4.806   -0.000    0.017   -0.539   -0.108    0.000   -5.435 
"""

def parse_fsapt_data(data):
    """Parse the fsapt data from a string."""
    lines = data.strip().split('\n')
    header = lines[0].split()
    records = [line.split() for line in lines[1:]]
    
    # Convert to DataFrame
    df = pd.DataFrame(records, columns=header)
    
    # Convert numeric columns to appropriate types
    numeric_cols = ['Elst', 'Exch', 'IndAB', 'IndBA', 'Disp', 'EDisp', 'Total']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    return df


def main():
    # Parse the fsapt data
    # sleep 2 seaonds
    import time
    time.sleep(2)
    df = parse_fsapt_data(fsapt_data)
    df['source'] = 'fsapt'
    df['abs(total)'] = df['Total'].abs()
    df.sort_values(by='abs(total)', ascending=False, inplace=True)
    df['fA-fB'] = df['Frag1'] + '-' + df['Frag2']
    pd.set_option('display.max_rows', None)  # Show all rows
    # Display the DataFrame
    print("FSAPT DataFrame:")
    print(df)
    df_ap2 = pd.read_pickle("./apnet2_results.pkl")
    # df_ap2['source'] = 'ap2'
    print("\nAPNet2 DataFrame:")
    print(df_ap2)
    # The 'fA-fB' column is used for merging; however, not exactly the same
    # across both DataFrames...
    df_merged = pd.merge(df, df_ap2, on='fA-fB', how='outer', suffixes=('_fsapt', '_ap2'))
    print("\nMerged DataFrame (Limited):")
    df_merged['Total_diff'] = df_merged['total'] - df_merged['Total']
    print(df_merged[['fA-fB', 'total', 'Total', 'Total_diff', 'source']])
    return

if __name__ == "__main__":
    main()

