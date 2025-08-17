import sympy as sp
from sympy import expand, Symbol, Add, Mul
from collections import deque
from St1_Fns import *

def carothers_observation1_ode_system(odes, variables):
    vmap = {}
    flattened_odes = {}
    vdefs = {}
    visited = set()

    def mon_to_symbol(mon):
        if mon not in vmap:
            vmap[mon] = Symbol(f"v_{list(mon)}")
        return vmap[mon]
    
    def v_to_pow(vvar):
        return [int(x) for x in str(vvar)[2:].strip('][').split(',')]

    def term_to_monomial(term):
        powers = [0] * len(variables)
        if isinstance(term, Mul):
            factors = term.args
        else:
            factors = [term]
        for f in factors:
            if f in variables:
                idx = variables.index(f)
                powers[idx] += 1
            elif isinstance(f, sp.Pow):
                base, exp = f.args
                if base in variables:
                    idx = variables.index(base)
                    powers[idx] += int(exp)
        return tuple(powers)
    
    def absorb_vvars_and_vars_into_pow(term):
            vterms = []
            monomial = 1
            
            for factor in term.args if term.is_Mul else [term]:
                if factor.is_Pow and factor.base.is_Symbol and factor.base.name.startswith('v_'):
                    if factor.exp != 1:
                        vterms.extend([factor.base] * factor.exp)
                    else:
                        vterms.append(factor.base)
                elif factor.is_Symbol and factor.name.startswith('v_'):
                    vterms.append(factor)
                else:
                    monomial *= factor
            
            # Convert v-variables to power lists
            v_pows = [v_to_pow(v) for v in vterms]
            
            # Convert the remaining monomial to a power list
            mon_pows = mon_to_pow(variables, monomial)
            
            # Add all power lists together element-wise
            pows = [sum(x) for x in zip(*v_pows, mon_pows)]
            
            return pows

    def powminus(pow1,pow2):
        ret = []
        for i in range(len(pow1)):
            ret.append(pow1[i]-pow2[i])
        return ret
    
    def crn_safe_negative(y, coeff, product):
        """Rewrites -coeff * product as -y * alpha where y * alpha = product."""
        total_mon = absorb_vvars_and_vars_into_pow(product)
        y_mon = v_to_pow(y)
        residual_mon = powminus(total_mon,y_mon)
        alpha = mon_to_symbol(tuple(residual_mon)) if sum(residual_mon) > 1 else sp.Mul(*[v**d for v, d in zip(variables, residual_mon)])
        return coeff * y * alpha, alpha  # coeff is already negative

    # Step 1: build flattened x_i' ODEs
    worklist = deque()
    for var in variables:
        expr = expand(odes[var])
        result = 0
        terms = expr.args if isinstance(expr, Add) else [expr]
        for t in terms:
            coeff, t_expr = t.as_coeff_Mul() if isinstance(t, Mul) else (1, t)
            mon = term_to_monomial(t_expr)
            if sum(mon) <= 1:
                result += coeff * t_expr
            else:
                if coeff > 0:
                    v = mon_to_symbol(mon)
                    result += coeff * v
                    worklist.append(mon)
                else:
                    idx = variables.index(var)
                    new_mon = list(mon)
                    new_mon[idx] -= 1
                    new_mon = tuple(new_mon)
                    v_y = mon_to_symbol(new_mon)
                    residual_mon = tuple(a - b for a, b in zip(mon, term_to_monomial(var)))
                    alpha = mon_to_symbol(residual_mon) if sum(residual_mon) > 1 else sp.Mul(*[v**d for v, d in zip(variables, residual_mon)])
                    result += coeff * var * alpha
                    worklist.append(new_mon)
        flattened_odes[var] = sp.simplify(result)

    # Step 2: build v-variable ODEs
    while worklist:
        mon = worklist.popleft()
        if mon in visited or sum(mon) <= 1:
            continue
        visited.add(mon)
        v = mon_to_symbol(mon)
        rhs = 0
        for i in range(len(variables)):
            if mon[i] > 0:
                T_drop = list(mon)
                T_drop[i] -= 1
                T_drop = tuple(T_drop)
                v_reduced = mon_to_symbol(T_drop)
                if T_drop not in visited:
                    worklist.append(T_drop)

                xi_prime = flattened_odes[variables[i]]
                xi_expr = expand(xi_prime)
                for term in (xi_expr.args if isinstance(xi_expr, Add) else [xi_expr]):
                    coeff, residual = term.as_coeff_Mul()
                    if coeff == 0:
                        continue
                    product = v_reduced * residual
                    if coeff > 0:
                        rhs += coeff * product
                    else:
                        # enforce CRN-safe: rewrite -a * product as -v * alpha
                        newterm, newvar = crn_safe_negative(v, coeff, product)
                        if not newvar in worklist: #and not newvar in variables:
                            worklist.append(tuple(v_to_pow(newvar)))
                        rhs += newterm
        vdefs[v] = sp.simplify(rhs)

    return {**flattened_odes, **vdefs}

def peaks(input_sys, variables):
    """
    Return the set of 'peak' monomials from a system.
    A monomial m1 is a peak if no other monomial m2 in the system satisfies m1 < m2 componentwise.
    """

    def term_to_monomial(term):
        powers = [0] * len(variables)
        if isinstance(term, Mul):
            factors = term.args
        else:
            factors = [term]
        for f in factors:
            if f in variables:
                idx = variables.index(f)
                powers[idx] += 1
            elif isinstance(f, sp.Pow):
                base, exp = f.args
                if base in variables:
                    idx = variables.index(base)
                    powers[idx] += int(exp)
        return tuple(powers)

    # Extract all monomials from all expressions
    monomials = set()
    for expr in input_sys.values():
        expr = expand(expr)
        terms = expr.args if isinstance(expr, Add) else [expr]
        for t in terms:
            _, t_expr = (t.as_coeff_Mul() if isinstance(t, Mul) else (1, t))
            mon = term_to_monomial(t_expr)
            monomials.add(mon)

    # Remove dominated monomials
    peaks = set()
    for m in monomials:
        if not any(
            all(mi <= mj for mi, mj in zip(m, m2)) and m != m2
            for m2 in monomials
        ):
            peaks.add(m)

    return peaks

