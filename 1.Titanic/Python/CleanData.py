#attach the training data header
# 00 ''         01 'PassengerId' 02 'Survived'  03 'Pclass'  04 'Name' 05 'Sex'
# 06 'Age'      07 'SibSp'       08 'Parch'     09 'Ticket'  10 'Fare' 11 'Cabin'
# 12 'Embarked' 13 'Gender'      14 'Embark']

#attach the testing data header
# 01 PassengerId 02 Pclass     03 Name     04 Sex
# 05 Age         06 SibSp      07 Parch    08 Ticket
# 09 Fare        10 Cabin      11 Embarked

#calculate the average age 5 6
def calculate_average_age(sample_matrix,index_of_age):


    totalAge = 0
    totalCount = 0

    for row in sample_matrix:
        if(row[index_of_age] != "NA"):
            totalAge+=(float(row[index_of_age]))
            totalCount+=1

    averageAge = int(totalAge/totalCount)

    for row in sample_matrix:
        if(row[index_of_age] == "NA"):
            row[index_of_age] = averageAge
