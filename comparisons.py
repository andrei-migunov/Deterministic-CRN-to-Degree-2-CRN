from carothers2005 import *
import importlib  
from decompose_CRN import *
from St0_Fns import *
from itertools import combinations
from math import prod
from Tools.plotting_etc import *
import time


'''The tests below compare the method of Carothers, et. al. (2005) (Obs. 1) to a greedy strategy for decomposing a high-degree polynomial ODE system A
into a polynomial system B whose primary variables have the same definition and behavior as those of A, but whose monomials all have degree no more than 2. 

For example, if x' = ... + x**2y**8 appears in A, then in B it must equal something like x' = ... + v*y, where v = x**2 y**7 is a newly introduced variable, whose own ODE must now also be defined, since we are choosing to include v in the system. Thus, each choice of implementation has downstream effects in terms of which new variables must be implemented.

The main idea is that one should try to be careful and selective in how one introduces new variables. For example, we could have rewritten x**2 y**8, instead, as 
x * w where w = x y**8. We might be able to `recycle' this new variable if another monomial elsewhere in the system can also use it. 

One greedy approach goes as follows:

Anytime we encounter a monomial m whose degree is higher than 2, we rewrite it according to the following heuristic. 

1. If there is any way to write m as m = w*v where w and v are auxiliary variables that are both already implemented, then do so. (Introduces 0 new variables)
2. If there is a way to write m as m = w*v where w is already defined in the system and v is not, and the degree of v is no more than half the degree of m, then do so. (Introduces 1 new variable)
3. Otherwise, write m as m = w*v ensuring that w and v both have degree half of the degree of m (i.e. - distribute degree of m evenly among the two new variables) and introduce both into the system. (Introduces 2 new variables)


One can show that the above heuristic is guaranteed to *terminate* as a result of insist on splitting the degree evenly - it will not continue introducing new variables forever. However, it is certainly not optimal. The examples below demonstrate that sometimes this heuristic saves us a lot of variables compared to the Carothers method. Sometimes it actually performs worse! One can run *both* methods and choose the better outcome, or explore the huge variety of other decomposition strategies and heuristics. '''

# Define canonical variable symbols
x1, x2, x3, x4 = sp.symbols('x1 x2 x3 x4')

def system_decomp_test(input_system, vars):
    if not crn_implementable(input_system):
        raise ValueError(f'Input system is not CRN-implementable, so decomposition will not be performed.')

    #IDENTIFIES THE SYSTEM PEAKS AND MAXIMUM MONOMIAL DEGREE
    peak_terms = peaks(input_system,vars)
    max_deg = max([sum(term) for term in peak_terms])

    print("###############")
    print(f'Peaks (highest degree incomparable monomials in the input system, as vectors): {peak_terms}')
    print(f'Degree of highest monomial in the input system: {max_deg}')
    print(f'The number of elements in the union of the lattices at the peaks (multiple peaks typically overlap somewhat): {lattice_union_size(peak_terms)}')
    print(f'The size of the ENTIRE simplex [ v in N^{len(vars)} | sum(v) <= {max_deg}] in which all variable-finding takes place (defined by the highest degree): {max_deg**len(vars)}')
    print("###############")
    #COMPUTES THE STANDARD DECOMPOSITION USING CAROTHERS (2012)
    
    print(f'Computing standard decomposition via Carothers algorithm...')
    t1 = time.time()
    carothers_system = carothers_observation1_ode_system(input_system, vars)
    t2 = time.time()
    print(f'Done in {t2-t1} seconds.')
    if not crn_implementable(carothers_system):
        raise ValueError(f'Something went wrong - Carothers system is not CRN-implementable. Terminating.')
    print(f'Number of auxiliary variables introduced during Carothers: {len(carothers_system.items()) - len(input_system.items())}')
    print("###############")

    #COMPUTES THE GREEDY DECOMPOSITION
    print(f'Computing decomposition via greedy method...')
    greedy_method = DecompWithBD(input_system)
    t1 = time.time()
    greedy_method.decomp()
    t2 = time.time()
    print(f'Done in {t2-t1} seconds.')
    print(f'Number of auxiliary variables introduced during greedy decomp: {len(greedy_method.system.keys()) -len(input_system.keys())}') 

    print("###############")

#32(greedy) vs 122 (naive)
def test1():
    odes_input = {
        x1: x2 * x3**2 - x1**3 * x2 **5 * x3 ** 6,
        x2: x1 * x3,
        x3: x1**2 * x3
    }


    print("Input system:\n")
    print(format_dict(odes_input))

    system_decomp_test(odes_input, [x1,x2,x3])

# 44 (greedy) vs 44 (naive)
def test2():
    odes_input2 = {
        x1: x2 * x3**2 + x1**3 * x2 **2 * x3 ** 3,
        x2: x1 * x3,
        x3: x1**2 * x3
    }
    print("Input system:\n")
    print(format_dict(odes_input2))
    system_decomp_test(odes_input2, [x1,x2,x3])


# 345 (greedy) vs (296) (naive) - note that this means that the greedy algoritm leaves the lattice defined by the peaks 
#It does not leave the simplex, but that is a coincidence of the particular greedy strategy. Other strategies may go outside the simplex.
def test3():
    odes_input3 = {
    x1: x2 * x3**2 * x4*5 + x1**3 * x2 **5 * x3 ** 11 - x2**2 * x1,
    x2: x1 * x3*x4 - x2**2 * x4**3 - x2**4 * x3 * x1**2,
    x3: x1**2 * x3 - x3**4 * x1 * x2**3,
    x4: -3*x4**5 + x2**2 * x3**2 * x1
}
    print("Input system:\n")
    print(format_dict(odes_input3))
    system_decomp_test(odes_input3, [x1,x2,x3,x4])


test1()
test2()
test3()

