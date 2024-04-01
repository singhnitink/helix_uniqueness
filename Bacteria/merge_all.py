import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
plt.style.use('classic')
#proteins="5MDW"
parameter="Bacteria"
proteins="1HUE	2GB7	6KYS	2WZM	5EB0	4W1T	2NPN	7NZ0	6ARI	1G87	5VG1	3FGC	7ARM	1CHU	6SSI	2FE6	1U07	3T61	3NBU	6QV1	1QA6	1RPN	6J0E	1M6N	6CF3	3HHX	6EJF	2XDF	6MOU	2WFK	3KOP	4D4D	4S2V	5KQU	4D9I	2Z7B	5MDW	3OID	6KTK	3CE6	4RV4	6DTQ	3MKK	4MTD	3ENW	1XXD	2RML	1JIK	5U7H	1ATG	2FJT	2DRJ	2FKN	3BZC	5LIQ	2CGK	4R1H	3SHO	6X6Y	3WRE	3U2Q	2WLG	3V4G	6HPB	4MIM	6HQW	5BTC	2WPN	5XNZ	5B2O	1P4N	6PD2	6RO2	5LZE	1EM7	5ZSX	3QBU	5EYR	4YER	2GE4	6ZE0	5DXZ	4AT2	4HYJ	4Q14	5FGN	6YG8	7C9M	6XSF	1FJJ	1YDO	4X9E	1DUA	4XT6	5U3H	1CGT	2OXN	6IOA	4UAC	5N8H	3GUZ	6RI7	6FFW	1O3S	4E69	4P6C	2FGG	2ZYM	3LCH	6IBV	4UXZ	4TRO	1OUR	4XTU	6UBR	6C39	1I8O	2PA0	5TGF	1NFS	6MLX	1JXA	4R8X	1AHJ	3PFT	7D6C	2WN4	6JUO	3T4Z	5BP3	7C71	2L17	2OYK	2SNI	7JT3	3NR7	7OCK	6U5Z	6Q97	1S0E	6IV0	4BMQ	1RQK	7MQD	3ZQY	4ZRB	2JNE	6FMV	4OW8	1O8I	4XXV	2Q1S	5DN5	6JZV	2BX7	1KYI	1J1V	5KKO	5ELF	3ZQX	4ZGL	1K2X	4Y1G	4OK2	4N91	2IJ0	7A6D	6TQD	6P0T	1DOY	2WTX	6NAQ	2E67	2CO1	2WGB	6WA1	1T8H	3WFC	3PZ5	3UU9	6YGX	1ESP	3M4W	1T0K	4F8Y	3B8J	5CPM	6HQQ	4EIA	4BAX	4V7K	5LCC	5KIA	4URH	4ZSZ	4AIQ	6U7I	4FDB	3KCU	2XXP	6SDR	6N0F	6N8A	4IAR	5DX7	6OOK	1POP	3RN9	7DWQ	6VBM	3I86	6H4F	4V9P	1AV2	2GPJ	2DG6	1M65	5ISW	1UX8	7JNS	3MWC	6M4V	3MDQ	6RB7	6DOY	5ESU	3HYX	4U9Q	3C0Q	2DET	6HYZ	1O5P	4J3O	1WZX	7AM0	1XLJ	3X1G	6T7M	4E0P	5TP4	1ETN	2JD0	2L0L	4A8R	4TUC	5Z1A	1R9C	5F0Z	5NPE	3S0Y	1SJP	6PT7	1IHZ	6D1K	5LKQ	6DYJ	2OV9	3NKE	4F0R	6LL1	6LU1	2Z6J	4Q2L	6T1E	6WO6	1O3R	5X9I	3VYN	1KP4	5ZPT	3JSZ	6GTP	2G2U	3OZU	6NNN	1U9N	4MV1	6TBB	2D42	1VRV	2AWD	5KRY	6IXX	1JD7	4HO1	2M57	3SQN	6NDZ	3TR5	7CBN	4LWL	1JAZ	5ZDO	4HVX	4BFY	2KO1	7RAH	4QME	1P7Y	5XDU	4CSD	2PBP	4ODG	7LCM	3JTR	1MUC	4NBX	1F9G	2QM8	5QAA	7C83	6SYY	3UBL	1WC1	4GI5	1ZUN	1UIY	3NIP	6PAR	2GKN	6J9L	4QXL	5TPU	2EHQ	6PQE	4PH8	1YAI	4V02	3OKS	3K4I	5EE2	1DFH	7KD1	1VHX	3I23	3TQI	3UCS	6R1X	2G0B	3NH3	5B6E	5FTX	1IOK	1LRK	1BSB	4CL0	6IFO	2A5G	4DIO	3V5V	3GYZ	1BUZ	5UUE	4MME	4M4W	6NNZ	4XPX	5A0G	3HMC	6WB4	3BWV	5JFF	1IXU	5AFA	6QHX	4ZVE	4JMX	3RWL	4B0C	2Q1T	4ZHW	3QZ5	4KDP	1I9H	2G03	3IFZ	7LTR	4C1D	2BP0	5ODE	4JQC	6FVQ	2YGX	2J62	4H7A	3PKW	1VR8	1E5M	4QR0	5X1J	6G8B	4NLL	7JWG	6A37	4KD3	2VGU	3IVE	4QYZ	2C9X	7EWO	1OIL	4ON0	6EX4	5HFT	6TXL	5D8Q	5Z3K	3QS2	6FZE	2KZN	3DNH	4XNI	5UI6	7KHA	1NLR	3URC	2Q69	2UU9	4XYQ	3CGH	4B0T	1YYO	6IOC	2QFI	6VPV	5BY3	1O87	4JEB	3M5U	7AFR	1H9T	5DJI	5NPY	3GUQ	4EB0	5MDP	1O0C	1F0L	1AK9	6GOC	7D6T	3KWR	6YKO	4K9R	4YB1	6SE9	6UFI	2IHV	4HKY	1I6D	6JFR	3KKF	6WM7	4ACK	3N0M	1AKL	6XXA	2Y1O	2YL1	5XD7	3P6R	3V6P	3G98	6GQ5	2ZOI	3VK5	6RKC	6LQ8	1Z77	3B57	3NWI	4LJ3	4IWT	1C6S	6VV3	3ACW	3ETS	6SXN	5Z3R	2X5Z	1RLM	6RXJ	2UYK	2KYA	4M5N	1T6A	3C57	3LSR	4P7X	3FTJ	7NTX	1USY	1R5I	6YTJ	4UTG	1XXO	5VRV	5K0T	1PA4	4KO1	1E6U	6WJE	5NY7	3GKM	2Z1X	6FRL	6LXS	7EGN	4JUU	1FFL	5M11	2Q2N	2PIL	2IGZ	6EYR	4B3H	1AGX	6JW8	2Y6L	6WD3	4H8Q	1P5X	5AOP	2FMI	1UTA	3ZPL	6N0I	4NNH	1RZQ	3PQJ	5IGJ	3QY8	7B22	4HA1	2GG5	2FEM	6A6G	2RAD	2HLT	4RK1	1N3I	4JYX	4G2T	3BZW	4XS5	1QPN	1DZO	2JLN	6XRR	6QI2	4ZZE	2K0Y	1XAI	1JFR	6ECV	3R79	2YH5	4TM1	6LVT	6IM3	3TLB	6B1Y	3IHG	2EF0	5ZLP	1QTM	3FDD	6ESU	6QZ2	5GLB	2ZXG	6R0S	6VW0	2JOK	4Q0V	3LYC	1XX6	5U3F	5DPD	2JPF	2A84	3OW7	6R27	2AJQ	7CB8	2M4V	4ONW	2UVJ	2RQX	3I3B	3SR1	6V03	3T5M	3VEB	4F7K	7OA5	6T7E	4JES	4R8U	4NPB	3GJA	1NY6	3DKQ	7BVD	3LV0	3ZOF	1LJU	4P5B	3O5C	4ZXU	1G7X	5B66	7NLY	6ND7	6H6W	5B4V	1RIY	1R0H	7K44	4JA3	2OYR	6ABA	3PH1	6QR7	3C01	3BW8	6WT5	2PUB	2IGA	2G7G	6HPA	3PJ0	6NOR	3LXH	5NNV	5IJ3	3L62	3HMY	7KQR	1PWU	1R94	3QZA	1CY8	1MTL	5XK9	2V1P	3FS9	2CDY	2OGA	4F8Z	1U7I	6UJE	2SNM	3DLL	4H40	2IEX	1UBO	4JI8	1YTW	1PF5	4EM0	5AMV	4NS1	1MAE	4H8H	1CB8	2VT1	5WBC	2FU6	5ZIK	3KE3	3V3T	1MQE	2B3B	3HLZ	6FXO	5T4C	2HP7	1XVF	6IIV	1FG7	1VSB	3G05	2JJM	3IA4	4NQ4	1KO8	3FD9	2LS5	7KZU	5KD8	6JIR	4YTB	1HTO	4FIN	4ZJX	3LR4	3HCY	1N9M	4DQ0	6AU0	5CL1	1HM2	3HYC	3U6D	4UIC	4NX8	5O15	1UKA	4D7R	1M2W	6ILX	2A10	3ZK7	5UWB	6DDP	4KEN	3RI6	2GQQ	4NAX	3X20	1OPX	2INB	1EE8	3TMB	2LCG	6GDF	2DEF	6FV8	2OV7	4Y96	3MF8	3L31	3BQE	1O6Q	6QOJ	5DN6	7KJD	4L4W	2ORI	5Y9G	5VEG	5F7C	3FI9	1FR1	6KLK	4HCX	1CUR	6FFV	6WCG	5D0G	3U4C	2GZW	5KDB	3RDS	5YV1	2JVV	5VI0	3NNR	1W5Z	5VWR	3GEK	3GH7	6VCI	6MOY	2BB8	4GAM	3P2Q	6QVP	6XSY	5A5Z	3J4J	6RH0	2Q3L	5XOU	3DUK	6R0Z	3EY7	2K29	5CJM	4C7Z	3FZQ	2PFC	1W5C	1JC4	2WCO	6VJP	2HXX	3L7X	6AMX	4CJZ	3L6T	3TYD	7NBN	2Q3W	2Y9K	3OOE	5XUS	3GEM	5LG5	1WEK	1E9M	3B1C	3PSA	5QLZ	2QW7	1WU0	5WGG	3VL3	1GIM	6RZD	2WIT	3UD5	1A87	2CNY	5UAJ	5XLH	6N6W	2FBH	6K9S	2DEG	2ICI	7OZ3	4QLM	6MPA	4IPT	6W3S	2ISM	3D1O	5ABR	3KE5	6J8V	3OWR	6X7O	6F96	3CJY	1Z3Z	6O55	6UR7	2VHL	4HUN	6AUL	5ZCZ	6AJN	6PFN	2GAE	2HZA	6QGX	3TD2	6HRA	1XJS	7JND	2C97	2JKL	6QO9	2XCR	2DG8	4I3I	2W79	3VCP	6QWI	1JVS	6BAC	7NL5	1S7I	1CTN	1TVF	1YQP	3A51	5DIP	3T4W	6LS0	2GGC	2A0F	1MUD	3GXH	6HHX	3P32	3GA2	6E17	6WYN	1P3C	4WBS	6Q1C	4G7S	5BKD	4QIT	3A38	3H33	6YNU	1Y1R	6RBY	1TUO	1TWJ	4KTS	6OH7	5V8U	4WGJ	6B8B	5KX9	1TGY	5XD4	3VTI	4OPY	3WOD	2A8Z	3D1F	3ZXQ	7CXU	6L4L	4H83	3T80	4X62	5FAO	3FEW	6AW3	1GQ6	1ZVW	1VEO	5CIZ	6XPG	6HZR	4BAJ	2J45	2VY9	2WCV	2XDI	2ODI	4HN8	4WXB	6C9W	3LA9	5GXV	4FJ6	3SON	3LSN	4H5F	2E3D	5JT9	1A7T	2CHH	1NKI	4KNC	5WVM	4HNV	3GVD	7M5F	4DZA	4MVA	3H19	1HAU	4PSY	3KBS	6ITL	3MDU	4KKL	1KH7	1EJX	3OX9	5UUS	1KMO	6G9G	3IMK	3ONR	3DWZ	5V74	6H9P	1DCZ	7D80	1G4E	7AUB	6XXD	1XG4	6E1X	4L1K	1OTX	3VYG	1B65	2IWB	6DOP	5WDL	7O0N	4KSA	3GA9	2OEC	4P2V	4APB	1A23	3K13	2BT9	5ZGO	3MV0	2W1N	4ZQP	4F60	4MB5	4H1D	4ZC1	6ENO	6BLD	"
col_names=['Number','Chain','AA-Name','Structure']
#***************************************
PDBnames=[]
df = pd.DataFrame()
for name in proteins.split():
    #print(name)
    data = pd.read_csv("../../../bacteria/"+name+"_secst_data.dat",names=col_names, delimiter=" ")
    each_aa=data.iloc[:,2]
    length=len(data.iloc[:,0])
    if length!=0:
        i=0
        ##Counting each aa in the particular secondary structure
        A1=0
        R1=0
        N1=0
        D1=0
        C1=0
        Q1=0
        E1=0
        G1=0
        H1=0
        I1=0
        L1=0
        K1=0
        M1=0
        F1=0
        P1=0
        S1=0
        T1=0
        W1=0
        Y1=0
        V1=0
        while i < length:
            if each_aa[i]=='ALA':
                A1+=1
            elif each_aa[i]=='ARG':
                R1+=1
            elif each_aa[i]=='ASN':
                N1+=1
            elif each_aa[i]=='ASP':
                D1+=1
            elif each_aa[i]=='CYS':
                C1+=1
            elif each_aa[i]=='GLN':
                Q1+=1
            elif each_aa[i]=='GLU':
                E1+=1
            elif each_aa[i]=='GLY':
                G1+=1
            elif each_aa[i]=='HIS':
                H1+=1
            elif each_aa[i]=='ILE':
                I1+=1
            elif each_aa[i]=='LEU':
                L1+=1
            elif each_aa[i]=='LYS':
                K1+=1
            elif each_aa[i]=='MET':
                M1+=1
            elif each_aa[i]=='PHE':
                F1+=1
            elif each_aa[i]=='PRO':
                P1+=1
            elif each_aa[i]=='SER':
                S1+=1
            elif each_aa[i]=='THR':
                T1+=1
            elif each_aa[i]=='TRP':
                W1+=1
            elif each_aa[i]=='TYR':
                Y1+=1
            elif each_aa[i]=='VAL':
                V1+=1
            i+=1
        d = {'columns': ['G','A','P','V','L','I','M','F','Y','W','S','T','C','N','Q','K','H','R','D','E'],
             'data': [[G1/length,A1/length,P1/length,V1/length,L1/length,I1/length,M1/length,F1/length,Y1/length
                       ,W1/length,S1/length,T1/length,C1/length,N1/length,Q1/length,K1/length,H1/length,R1/length,D1/length,E1/length]],
             'index': [name]}
        df1 = pd.DataFrame(d['data'], columns=d['columns'], index=d['index'])
        df=df.append(df1)
df.to_csv("BACTERIA-all_aacontent.csv")
###
col_names1=['Name','G','A','P','V','L','I','M','F','Y','W','S','T','C','N','Q','K','H','R','D','E']
df2 = pd.read_csv("BACTERIA-all_aacontent.csv",names=col_names1, header=0)
names=df2.iloc[:,0]
#print(names)
#####
#xlist=['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
x=np.arange(0,len(names),1)
A_mean=df2.loc[:,"A"].mean()
R_mean=df2.loc[:,"R"].mean()
N_mean=df2.loc[:,"N"].mean()
D_mean=df2.loc[:,"D"].mean()
C_mean=df2.loc[:,"C"].mean()
Q_mean=df2.loc[:,"Q"].mean()
E_mean=df2.loc[:,"E"].mean()
G_mean=df2.loc[:,"G"].mean()
H_mean=df2.loc[:,"H"].mean()
I_mean=df2.loc[:,"I"].mean()
L_mean=df2.loc[:,"L"].mean()
K_mean=df2.loc[:,"K"].mean()
M_mean=df2.loc[:,"M"].mean()
F_mean=df2.loc[:,"F"].mean()
P_mean=df2.loc[:,"P"].mean()
S_mean=df2.loc[:,"S"].mean()
T_mean=df2.loc[:,"T"].mean()
W_mean=df2.loc[:,"W"].mean()
Y_mean=df2.loc[:,"Y"].mean()
V_mean=df2.loc[:,"V"].mean()
dx = {'columns': ['G','A','P','V','L','I','M','F','Y','W','S','T','C','N','Q','K','H','R','D','E'],
     'data': [[G_mean,A_mean,P_mean,V_mean,L_mean,I_mean,M_mean,F_mean,Y_mean,W_mean,S_mean,T_mean,C_mean,N_mean,Q_mean,K_mean,H_mean,R_mean,D_mean,E_mean]], 'index': [1]}
dfx = pd.DataFrame(dx['data'], columns=dx['columns'], index=dx['index'])
dfx.columns.names = ['Amino-acid']
dfx.to_csv("mean-all_aacontent.csv")
row = dfx.iloc[0]
plt.figure(figsize=(15,7),facecolor='white')
ax=row.plot(kind='bar', color='orange')
plt.legend([parameter])
plt.title(parameter+"-Amino-acid propensity", fontweight='bold')
plt.ylabel("Mean propensity")
plt.yticks(np.arange(0,0.4,0.04))
plt.xticks(rotation=0)
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
plotfile=parameter+'-amino-acid_ALL.png'  ##**CHANGE HERE
plt.savefig(plotfile, dpi=150, bbox_inches='tight')
#plt.show()
