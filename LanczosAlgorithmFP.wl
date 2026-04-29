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

IP[A_,B_,d_]:=Module[{Aflatten=Flatten[A],Bflatten=Flatten[B]},
(1/d) ConjugateTranspose[Aflatten] . Bflatten]

LanczosAlgo[H_,\[Theta]0_,d_,\[CurlyEpsilon]_,p_]:=Module[
{A,b,norm0,K},
b=Table[0,{d^2-d+2}];
K=0;

KB1=\[Theta]0/Sqrt[IP[\[Theta]0,\[Theta]0,d]];
A1=Commutator[H,KB1];
b[[1]]=SetPrecision[0,p];
b[[2]]=Sqrt[IP[A1,A1,d]];
If[Re[b[[2]]]<=\[CurlyEpsilon],
K=1;
Return[{K,{b[[1]],b[[2]]}}],
KB2=A1/b[[2]];
Do[
A2=Commutator[H,KB2]-b[[i]] KB1;
b[[i+1]]=Sqrt[IP[A2,A2,d]];
K=i;

If[Re[b[[i+1]]]<=\[CurlyEpsilon],
Break[],
KB1=KB2;
KB2=A2/b[[i+1]];
]
,{i,2,d^2-d+1}];
];
Return[{K,b}]]

\[Lambda] = SetPrecision[lambdaVal, p];

{sx,sy,sz} = JxJyJz[L];
sx = SetPrecision[sx/L, p];
sy = SetPrecision[sy/L, p];
sz = SetPrecision[sz/L, p];

Id = IdentityMatrix[2L+1];
HFP = -(1+\[Lambda])(KroneckerProduct[sz,Id]+KroneckerProduct[Id,sz])-4(1-\[Lambda])KroneckerProduct[sx,sx];
HFP = SetPrecision[HFP, p];

lengthIC = Length[ic];
constructIC[{coeff_, type1_, exp1_, type2_, exp2_}]:= Module[
{op1,op2},
op1=Switch[type1, 0, Id, 1, sx, 2, sy, 3, sz];
op2=Switch[type2, 0, Id, 1, sx, 2, sy, 3, sz];
coeff*KroneckerProduct[MatrixPower[op1, exp1], MatrixPower[op2, exp2]]
]

initialOp = Total[Table[constructIC[ic[[i]]], {i, 1, lengthIC}]];

{Kdim, Lanczos} = LanczosAlgo[HFP, initialOp, (2L+1)^2, 10^(-50), p];
{Kdim, Lanczos[[;;Kdim+1]]}
