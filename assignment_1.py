import sympy as sp

# Fill in with value from previous iteration
omega1 = -0.2102
omega2 = 0.2036
b = -0.0279

cat_a_x = [2, 3, 2, 4, 5, 6, 7, 7, 8, 8]
cat_a_y = [1, 2, 3, 3, 4, 5, 4, 6, 6, 7]
cat_b_x = [1, 2, 3, 4, 5, 6, 4, 7, 5, 6]
cat_b_y = [7, 6, 8, 7, 9, 8, 10, 9, 10, 11]

grad_omega1 = 1/20*(sum(1/(1+sp.exp(-(omega1*x+omega2*y+b)))*x for x,y in zip(cat_a_x, cat_a_y))+
                    sum((1/(1+sp.exp(-(omega1*x+omega2*y+b)))-1)*x for x,y in zip(cat_b_x, cat_b_y)))
grad_omega2 = 1/20*(sum(1/(1+sp.exp(-(omega1*x+omega2*y+b)))*y for x,y in zip(cat_a_x, cat_a_y))+
                    sum((1/(1+sp.exp(-(omega1*x+omega2*y+b)))-1)*y for x,y in zip(cat_b_x, cat_b_y)))
grad_b = 1/20*(sum(1/(1+sp.exp(-(omega1*x+omega2*y+b))) for x,y in zip(cat_a_x, cat_a_y))+
                    sum((1/(1+sp.exp(-(omega1*x+omega2*y+b)))-1) for x,y in zip(cat_b_x, cat_b_y)))

print(grad_omega1)
print(grad_omega2)
print(grad_b)