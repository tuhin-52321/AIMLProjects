#Write a function to calculate cube root of a number
def cube_root(n, k):
    #Loop from 1 to n
    for i in range(1,n+1):
        #Check if i*i*i is equal to n
        if i*i*i == n:
            #Return i
            return i
        #Check if i*i*i is greater than n
        elif i*i*i > n:
            #Now we know that the number is not a perfect cube
            #The cube root will be between i-1 and i
            #let's try to find the decimal upto precision k
            #Initialize low and high
            low = i-1
            high = i
            #Loop till low is less than high
            loss = round(i*i*i - n, k)
            while loss > 0:
                #Find mid
                guess = (low+high)/2
                #Check if mid*mid*mid is equal to n
                if guess*guess*guess == n:
                    #Return mid
                    return guess
                #Check if mid*mid*mid is less than n
                elif guess*guess*guess < n:
                    #Update low
                    low = guess
                    #Update loss
                    loss = round(n - guess*guess*guess, k)
                #Else update high
                else:
                    high = guess
                    #Update loss
                    loss = round(guess*guess*guess - n, k)
            #Return the guess
            return guess
    #Return -1
    return -1

#Call the function with value 27
print(cube_root(27))