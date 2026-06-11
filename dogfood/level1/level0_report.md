# Classification evaluation

```
n = 2800    classes = 10    accuracy = 0.437    macro-F1 = 0.453

class                  prec recall     f1  support
checking_savings_acc  0.305  0.482  0.374      280
credit_card           0.489  0.236  0.318      280
credit_reporting      0.628  0.489  0.550      280
debt_collection       0.536  0.482  0.508      280
money_transfer_or_se  0.206  0.546  0.299      280
mortgage              0.786  0.525  0.630      280
payday_or_personal_l  0.451  0.311  0.368      280
prepaid_card          0.618  0.411  0.494      280
student_loan          0.511  0.496  0.504      280
vehicle_loan_or_leas  0.630  0.389  0.481      280

confusion matrix (rows = expected, cols = predicted):

            checking_s credit_car credit_rep debt_colle money_tran   mortgage payday_or_ prepaid_ca student_lo vehicle_lo
  checking_s        135         20          3          3        104          1          1          9          1          3
  credit_car         69         66         11         16         60          7         12         23         10          6
  credit_rep         13          9        137         50         55          0          1          1         11          3
  debt_colle          5          7         30        135         63          3         13          1         10         13
  money_tran         98          4          1          2        153          0          3         13          2          4
    mortgage          8          3          3          4         42        147         20          1         44          8
  payday_or_         24         14          8         10         63         12         87         19         29         14
  prepaid_ca         76          5          1          3         75          0          3        115          1          1
  student_lo          2          2          8          5         69         11         30          2        139         12
  vehicle_lo         12          5         16         24         58          6         23          2         25        109
```
