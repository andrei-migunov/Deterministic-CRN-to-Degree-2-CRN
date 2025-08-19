# Deterministic-CRN-to-Degree-2-CRN
Some comparisons demonstrating a greedy conversion of a deterministic mass-action ODE system into an equivalent mass-action ODE system of degree 2. 'Equivalent' here means that both systems compute the same real number in the limit. Specifically, if the input system has variables x_1,...,x_n, then the output system has the same variables with the same values, but with their ODEs' expressions having lower degree. This means introducing auxiliary variables.

Comment and uncomment `testn()` calls in `comparisons.py` to see two algorithms applied to an ODE system. The first - due to Carothers, et. al. (2005) uses repeated variable substitution to reduce the degree of the input system, which can have monomials of arbitrary high degree. Ours, a greedy algorithm, tries to 'recycle' variables that have been introduced into the system when decomposing terms. Sometimes that works great. Sometimes it's worse!

Keep in mind that there are *lots* of other possible decomposition strategies. Here, the heuristic used is as follows:

Anytime we encounter a monomial m whose degree is higher than 2, we rewrite it according to the following heuristic. 

1. If there is any way to write m as m = w*v where w and v are auxiliary variables that are both already implemented, then do so. (Introduces 0 new variables)
2. If there is a way to write m as m = w*v where w is already defined in the system and v is not, and the degree of v is no more than half the degree of m, then do so. (Introduces 1 new variable)
3. Otherwise, write m as m = w*v ensuring that w and v both have degree half of the degree of m (i.e. - distribute degree of m evenly among the two new variables) and introduce both into the system. (Introduces 2 new variables)


