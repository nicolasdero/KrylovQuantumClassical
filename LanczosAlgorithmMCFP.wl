(* ::Package:: *)

JxJyJz[j_]:=Module[
(*This function returns the x,y,and z components of the angular momentum operator in the spin-j representation. The expressions are exact and involve square roots. Floating-point precision should be set outside this function.*)
{d,JP,JM,Mlist,MPlist,MMlist,Jx,Jy,Jz},{
d=2j+1;
JP=ConstantArray[0,{d,d}];
JM=ConstantArray[0,{d,d}];
Mlist=Table[i,{i,j,-j,-1}];
MPlist=Mlist[[2;;]];
MMlist=Mlist[[;;-2]];
For[
i=1,i<=Length[MPlist],i++,
JP[[i,i+1]]=Sqrt[j(j+1)-MPlist[[i]](MPlist[[i]]+1)];
JM[[i+1,i]]=Sqrt[j(j+1)-MMlist[[i]](MMlist[[i]]-1)];
];
Jx=(JP+JM)/2;
Jy=(JP-JM)/(2I);
Jz=DiagonalMatrix[Mlist];
{Jx,Jy,Jz}
}][[1]]

Commutator[A_,B_]:=A . B-B . A

IPMC[eigenvectorsList_,eigenvectorsSize_,A_,B_,p_]:= Module[
(*Computes the commutator of two matrices, with numeric precision p.*)
{APrec = SetPrecision[A,p],BPrec = SetPrecision[B,p]},
Total[With[{Av = APrec . #,Bv = BPrec . #},Conjugate[Av] . Bv]& /@ eigenvectorsList] /eigenvectorsSize 
];

LanczosAlgoMC[H_,\[Theta]0_,d_,nList_,nSize_,\[CurlyEpsilon]_,p_:256]:=Module[
(*This function implements the original Lanczos algorithm. Given a Hamiltonian operator and an initial (seed) operator,it computes the corresponding Lanczos coefficients and Krylov basis elements. While the original algorithm may suffer from numerical instabilities and Lanczos coefficients never reaching exactly zero, using arbitrary precision p helps to mitigate these issues.*)
{A,b,norm0,K},
a=Table[0,{d^2-d+1}];
b=Table[0,{d^2-d+1}];
K=0;

KB1=\[Theta]0/Sqrt[IPMC[nList,nSize,\[Theta]0,\[Theta]0,p]];
a[[1]]=IPMC[nList,nSize,KB1,Commutator[H,KB1],p];
A1=Commutator[H,KB1]-a[[1]]KB1;
b[[1]]=0;
b[[2]]=Sqrt[IPMC[nList,nSize,A1,A1,p]];
If[Re[b[[2]]]<=\[CurlyEpsilon],
K=1;
Return[{K,{b[[1]],b[[2]]}}],
KB2=A1/b[[2]];
C2=Commutator[H,KB2];
a[[2]]=IPMC[nList,nSize,KB2,C2,p];
Do[
A2=Commutator[H,KB2]-a[[i]]KB2-b[[i]] KB1;
b[[i+1]]=Sqrt[IPMC[nList,nSize,A2,A2,p]];
K=i;

If[Re[b[[i+1]]]<=\[CurlyEpsilon],
Break[],
KB1=KB2;
KB2=A2/b[[i+1]];
C2=Commutator[H,KB2];
a[[i+1]]=IPMC[nList,nSize,KB2,C2,p];
]
,{i,2,d^2-d}];
];
Return[{K,Re[b],Re[a]}]];

\[Lambda] = SetPrecision[lambdaVal, p];

{sx,sy,sz} = JxJyJz[L];
sx = SparseArray[SetPrecision[sx/L, p]];
sy = SparseArray[SetPrecision[sy/L, p]];
sz = SparseArray[SetPrecision[sz/L, p]];

Id = IdentityMatrix[2L+1];
HFP = -(1+\[Lambda])(KroneckerProduct[sz,Id]+KroneckerProduct[Id,sz])-4(1-\[Lambda])KroneckerProduct[sx,sx];
HFP = SparseArray[SetPrecision[HFP, p]];
{EFP, nFP} = Eigensystem[HFP];

idx = Flatten[Position[EFP,_?(Between[{energy-deltaE/2, energy+deltaE/2}])]]
EWindow=EFP[[idx]];
nWindow=nFP[[idx]];
sizeWindow=Length[EWindow];

lengthIC = Length[ic];
constructIC[{coeff_, type1_, exp1_, type2_, exp2_}]:= Module[
{op1,op2},
op1=Switch[type1, 0, Id, 1, sx, 2, sy, 3, sz];
op2=Switch[type2, 0, Id, 1, sx, 2, sy, 3, sz];
coeff*KroneckerProduct[MatrixPower[op1, exp1], MatrixPower[op2, exp2]]
]
initialOpTransit = Total[Table[constructIC[ic[[i]]], {i, 1, lengthIC}]];
If[onePoint == 1, initialOp = initialOpTransit - IPMC[nWindow, sizeWindow, IdentityMatrix[(2 L+1)^2], initialOpTransit, p] IdentityMatrix[(2 L+1)^2], initialOp = initialOpTransit]

{KdimMC, LanczosMC, aMC} = LanczosAlgoMC[HFP, initialOp, (2L+1)^2, nWindow, sizeWindow, 10^(-50), p];
{KdimMC, LanczosMC[[;;KdimMC + 1]], aMC[[;;KdimMC + 1]]}
