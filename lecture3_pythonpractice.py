import numpy as np

# Check range exercise
def check_range(x):
    if x >= 100:
        print('Large')
    elif x >= 50 and x < 100:
        print('Medium')
    elif x >= 20 and x < 50:
        print('Small')
    else:
        print('Tiny')

# Calculate the sum of numbers from 1 to 100 exercise
def sum_100():
    sum = 0
    for i in range(1, 101):
        sum += i
    print(sum)

# Function exercise
def quadratic_fun(x):
    return 3*x**2 +5*x +10

# Find minmum number exercise
def find_min():
    num_list = [24, 45, 12, 67, 83, 34, 21, 59, 77, 90]
    for i  in range(len(num_list)):
        if i == 0:
            min = num_list[i]
        elif num_list[i] < min:
            min = num_list[i]
    print(min)

# Determine the frequency of names exercise
def name_frequency():
    names = ["Alice","Bob","Alice","David","Carol","Bob","Alice","Claine"]
    name_count = {}
    for name in names:
        if name in name_count:
            name_count[name] += 1
        else:
            name_count[name] = 1
    print('Name Frequencies: '+str(name_count))
    print('The most frequent name is: '+max(name_count, key=name_count.get))

# Find inverse of 4 linear functions exercise
def inverse_function():
    # Construct the matrix
    A = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])

    # Calculate the inverse of the matrix
    A_inv = np.linalg.inv(A)

    print(A_inv)

# Checking
check_range(75)
sum_100()
print(quadratic_fun(2.6))
find_min()
name_frequency()
inverse_function()