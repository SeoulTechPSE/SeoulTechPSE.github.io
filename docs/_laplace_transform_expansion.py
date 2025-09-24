# Process Systems Lab., SeoulTech
# April 15, 2021 Authored by Kee-Youn Yoo
# April 4, 2024 Modifed for sympy version 1.12

import sympy
from sympy import integrate, laplace_transform
from functools import reduce

def subs_(e, s1, s2):

    if isinstance(e, sympy.LaplaceTransform):
        s_ = e.args[2]
        return e.subs(s_, s).subs(s1, s2).subs(s, s_)

    if isinstance(e, (sympy.Add, sympy.Mul, 
                      sympy.Derivative, sympy.Integral, sympy.Subs)):
        tp = type(e)      
        return tp(*[subs_(arg, s1, s2) for arg in e.args])

    return e


def laplace_transform_(*e, **f):
    
    t_ = e[1]
    
    if isinstance(e[0], (int, float)):
        return laplace_transform(*e, **f)[0]

    k = len(e[0].args)
    
    terms = []
    for i in range(k):
        if  k == 1:
            terms.append(e[0])
        else:
            if isinstance(e[0], (sympy.Mul, sympy.Derivative, sympy.Integral)):
                terms.append(e[0])
                break
            else:
                terms.append(e[0].args[i])
    
    m = len(terms)
    if m == 0:
        return laplace_transform(*e, **f)[0]
    
    Leq = sympy.Float('0')
    for i in range(m):

        flag = 0
        l = len(terms[i].args) 
        if l == 1:
            terms__ = terms[i]
        else:
            terms__ = sympy.Integer('1')
            for j in range(l):
                if isinstance(terms[i], (sympy.Derivative, sympy.Integral)):
                    terms__ = terms[i]
                    break
                else: 
                    if isinstance(terms[i].args[j], sympy.exp):
                        a = terms[i].args[j].args[0].args
                        if len(a) == 2:
                            flag = a[0]
                        else:
                            flag = a[0] *a[2]                                                         
                    else:
                        terms__ *= terms[i].args[j]

        Leq_ = laplace_transform_expansion(laplace_transform(terms__, e[1], e[2], **f)[0])

        if flag != 0: 
            Leq_ = Leq_.subs(e[2], e[2] -flag)

        Leq += Leq_

    return Leq.doit()


def laplace_transform_expansion(e):
    """
    Evaluate the laplace transforms of derivatives, integrals, and composites of functions
    """       
    
    if isinstance(e, sympy.LaplaceTransform):
        
        ex, t, s = e.args
        
        # Preliminaries --------------------------

        if len(ex.args) == 1: 
           
            c = []
            for arg in ex.args[0].args:
                if arg != t: c.append(arg)
                    
            if len(c) == 0:
                return e
            else:
                d = reduce(lambda x, y: x *y, c)
                #return (sympy.LaplaceTransform(ex.subs(d *t, t), t, s/d) /d)
                return (sympy.LaplaceTransform(ex.subs(d *t, t), t, s))
               
        if isinstance(ex.args[0], sympy.Pow): 
            ex = sympy.simplify(ex)
            
        ex0 = ex.args[0]           
        if not isinstance(ex, sympy.Integral):
            ex1 = reduce(lambda x, y: x *y, ex.args[1:])
           
        # -----------------------------------------            
      
        if isinstance(ex, sympy.Derivative):

            n = ex1.args[1]           
            return ((s**n) *sympy.LaplaceTransform(ex0, t, s)
                    -sum([s**(n -i) *sympy.diff(ex0, t, i -1).subs(t, 0) for i in range(1, n +1)]))
        
        elif isinstance(ex, sympy.Integral):        
            
            if len(ex.args[1]) == 3:

                tau, t0, t = ex.args[-1]
                if t0 != 0: return e                
                       
                if len(ex0.args) == 2:
               
                    f, g = ex0.args[0], ex0.args[1]
                
                    if f.args[0] == tau and g.args[0] == t -tau:           
                        return (sympy.LaplaceTransform(f, tau, s).subs(tau, t) 
                               *sympy.LaplaceTransform(g, t -tau, s)).subs(t -tau, t)
                    elif f.args[0] == t -tau and g.args[0] == tau:
                        return (sympy.LaplaceTransform(f, t -tau, s).subs(t -tau, t) 
                               *sympy.LaplaceTransform(g, tau, s)).subs(tau, t)
                    else:
                        return e
                    
                else:
                    n = len(ex.args) -2
                    if n > 0:
                        for i in range(n):
                            tau_, t0_, t_ = ex.args[i +1]
                            ex0 = integrate(ex0, (tau_, 0, t_))
                        ex0 = ex0.subs(tau_, t).subs(t_, t)
                    else:
                        ex0 = ex0.subs(tau, t)
                        
                    return (laplace_transform_expansion(sympy.LaplaceTransform(ex0, t, s)) /s)
            else:
                return e

        elif isinstance(ex0, sympy.exp):         

            c = []
            for arg in ex0.args[0].args:
                if arg != t: c.append(arg)

            d = reduce(lambda x, y: x *y, c)
                       
            if ex0 == sympy.exp(d *t):
                return (laplace_transform_expansion(sympy.LaplaceTransform(ex1, t, s)).subs(s, s -d))
            else:
                return e                
        
        elif isinstance(ex0, sympy.Pow) or ex0 == t:  
            
            if ex0 == t:
                n = 1
            else:
                n = ex0.args[1]
                if not n.is_integer: return e
                      
            c = laplace_transform_expansion(sympy.LaplaceTransform(ex1, t, s))
            
            if isinstance(c, sympy.Add):
                for i in range(len(c.args)):
                    for j in range(len(c.args[i].args)):
                        if isinstance(c.args[i].args[j], sympy.LaplaceTransform):
                            d = c.args[i].args[j].args[-1]
            elif isinstance(c, sympy.Mul):
                for i in range(len(c.args)):
                    if isinstance(c.args[i], sympy.LaplaceTransform):
                        d = c.args[i].args[-1]                   
            elif isinstance(c, sympy.LaplaceTransform):
                d = c.args[-1]
            # else:
            #     return ((-1)**n *sympy.diff(c, (s, n)))
            
            #return ((-1)**n *sympy.diff(c.subs(d, s), (s, n)).subs(s, d))
            s_ = sympy.Symbol('s')

            return (-1)**n *sympy.diff(c.subs(d, s_), (s_, n)).subs(s_, d)

        elif isinstance(ex0, (sympy.Derivative, sympy.Integral)):
                       
            if isinstance(ex1, sympy.exp):
                
                c = []
                for arg in ex1.args[0].args:
                    if arg != t: c.append(arg)

                d = reduce(lambda x, y: x *y, c)

                return (laplace_transform_expansion(sympy.LaplaceTransform(ex0, t, s).subs(s, s -d)))       
            
        elif isinstance(ex0, sympy.Heaviside):          
            
            t, m_a = ex0.args[0].args
            
            if ex1.args[0] == t +m_a:
                f = ex1.subs(t +m_a, t)
                return (sympy.exp(m_a *s) *sympy.LaplaceTransform(f, t, s))
            elif ex1.args[0] == t:
                f = ex1.subs(t, t -m_a)
                return (sympy.exp(m_a *s) *sympy.LaplaceTransform(f, t, s))
            else:
                return e
        
    if isinstance(e, (sympy.Add, sympy.Mul, 
       sympy.Derivative, sympy.Integral, sympy.Subs)):
        tp = type(e)      
        return tp(*[laplace_transform_expansion(arg) for arg in e.args])

    return e