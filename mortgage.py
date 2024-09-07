p = input("What is the amount borrowed?")
r = input("What is the annual interest rate - express this as a decimal such as 0.07 for 7%?")

# place your code here after this line
principal = float(p)
annual_rate = float(r)
periodic_rate = annual_rate / 12
number_of_payments = 12 * 30
A = principal * periodic_rate * (1+periodic_rate) ** number_of_payments \
    / ((1+periodic_rate) ** number_of_payments - 1)
payment_amount = int(A*100) / 100
print(payment_amount)
