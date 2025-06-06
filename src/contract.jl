using TensorKit: permute
using PEPSKit: EnlargedCorner

function contract_enlarged_corner(c1, t1, t4, (a, b))
    t4c1 = t4 * c1
    t1p = permute(t1, ((1,), (2, 3, 4)))

    #  C1---T1-6
    #  |    ||
    #  |    45
    #  |
    #  T4=2,3
    #  |
    #  1
    t4c1t1 = t4c1 * t1p
    t4c1t1p = permute(t4c1t1, ((1, 3, 5, 6), (2, 4)))
    ap = permute(a, ((5, 2), (1, 3, 4)))
    t4c1t1a = t4c1t1p * ap

    #  C1----T1-4
    #  |     ||
    #  |     |3
    #  |   5 |        1 2
    #  |    \|         \|
    #  T4----A-6      5-A*-3
    #  | \2  |          |
    #  1     7          4
    t4c1t1ap = permute(t4c1t1a, ((1, 4, 6, 7), (2, 3, 5)))
    bdagp = permute(b, ((3, 4), (5, 2, 1)))'
    cnw = t4c1t1ap * bdagp

    #  C1-T1-2 ---->4
    #  |  ||
    #  T4=AA*=3,5->5,6
    #  |  ||
    #  1  46
    #  1  23
    cnwp = permute(cnw, ((1, 4, 6), (2, 3, 5)))
    return cnwp
end

function contract_enlarged_corner(Q::EnlargedCorner)
    return contract_enlarged_corner(Q.C, Q.E_2, Q.E_1, Q.A)
end
