###
#Script to load and save the Titanic dataset to a CSV file
###

library(titanic)
titanic_data <- data("titanic_train")

# Save the dataset as a CSV file
write.csv(titanic_train, "/home/lukes/Documents/Portfolio/Titanic-Python-R/data/titanic_train.csv", row.names = FALSE)